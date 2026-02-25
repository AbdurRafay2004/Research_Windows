r"""
Script to fix Hugging Face cache issues in the notebook.
Updates Cell 2 and Cell 13 to ensure all cache files go to Y:\Research_Windows\huggingface_cache
"""

import json

# Read the notebook
notebook_path = 'densecap-grit-gritlm-fusion (1).ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# ============================================================================
# Fix Cell 2 - Set ALL cache environment variables BEFORE any imports
# ============================================================================
new_cell_2_source = '''# Cell 2: Import all necessary libraries

# Set Hugging Face cache to project directory (avoid filling C drive)
# IMPORTANT: Set these BEFORE any imports!
import os

# Create cache directory
cache_dir = r'Y:\\Research_Windows\\huggingface_cache'
os.makedirs(cache_dir, exist_ok=True)
offload_dir = r'Y:\\Research_Windows\\huggingface_cache\\offload'
os.makedirs(offload_dir, exist_ok=True)

# Set ALL cache environment variables
os.environ['HF_HOME'] = cache_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = cache_dir
os.environ['HF_METRICS_CACHE'] = cache_dir
os.environ['ACCELERATE_CACHE'] = cache_dir
os.environ['TORCH_HOME'] = cache_dir

# Now import other libraries
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# Transformers & HuggingFace
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# Detectron2
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, Instances

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"HF Cache: {cache_dir}")'''

# Update Cell 2 (index 1)
notebook['cells'][1]['source'] = new_cell_2_source

# ============================================================================
# Fix Cell 13 - Add cache_dir and offload_folder to model loading
# ============================================================================
new_cell_13_source = '''# Cell 13: Load GritLM and inject cross-attention adapters
# With Windows compatibility: falls back to fp16 if bitsandbytes fails

from transformers import AutoConfig
import copy

# Cache directory for model downloads
CACHE_DIR = r'Y:\\Research_Windows\\huggingface_cache'
OFFLOAD_DIR = r'Y:\\Research_Windows\\huggingface_cache\\offload'

# Check bitsandbytes availability on Windows
BNB_AVAILABLE = False
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
    print('bitsandbytes available - using 4-bit quantization')
except Exception as e:
    print(f'bitsandbytes not available ({e})')
    print('   Falling back to fp16 loading (uses more VRAM)')


class GritLMWithVision(nn.Module):
    """
    GritLM-7B with cross-attention adapters for visual conditioning.
    Memory-efficient version using 4-bit quantization (or fp16 fallback).
    """
    
    def __init__(
        self,
        model_name: str = "GritLM/GritLM-7B",
        num_cross_attn_layers: int = 8,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        
        if BNB_AVAILABLE:
            print(f'Loading {model_name} with 4-bit quantization...')
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=CACHE_DIR,
                offload_folder=OFFLOAD_DIR,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        else:
            print(f'Loading {model_name} with fp16...')
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=CACHE_DIR,
                offload_folder=OFFLOAD_DIR,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        
        self.config = self.llm.config
        hidden_size = self.config.hidden_size
        num_heads = self.config.num_attention_heads
        num_layers = self.config.num_hidden_layers
        
        if gradient_checkpointing:
            self.llm.gradient_checkpointing_enable()
        
        # Add cross-attention adapters to selected layers
        self.cross_attn_adapters = nn.ModuleList()
        self.cross_attn_layer_indices = []
        
        layer_interval = num_layers // num_cross_attn_layers
        for i in range(num_cross_attn_layers):
            layer_idx = i * layer_interval + layer_interval // 2
            self.cross_attn_layer_indices.append(layer_idx)
            self.cross_attn_adapters.append(
                CrossAttentionAdapter(hidden_size, num_heads)
            )
        
        print(f'Loaded GritLM with cross-attention at layers: {self.cross_attn_layer_indices}')
        
        # Freeze base LLM, only train adapters
        for param in self.llm.parameters():
            param.requires_grad = False
        
        for adapter in self.cross_attn_adapters:
            for param in adapter.parameters():
                param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f'Trainable parameters: {trainable_params:,} / {total_params:,} '
              f'({100 * trainable_params / total_params:.2f}%)')
    
    def forward(
        self,
        input_ids,
        visual_tokens,
        attention_mask=None,
        labels=None,
    ):
        inputs_embeds = self.llm.model.embed_tokens(input_ids)
        
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        
        return outputs


print('GritLMWithVision class defined')'''

# Update Cell 13 (index 12)
notebook['cells'][12]['source'] = new_cell_13_source

# Save the modified notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("=" * 60)
print("NOTEBOOK UPDATED SUCCESSFULLY")
print("=" * 60)
print("\nChanges made:")
print("  1. Cell 2: Added all cache environment variables before imports")
print("     - HF_HOME, TRANSFORMERS_CACHE, ACCELERATE_CACHE, etc.")
print("     - Created cache and offload directories")
print("\n  2. Cell 13: Added cache_dir and offload_folder to model loading")
print("     - cache_dir=Y:\\Research_Windows\\huggingface_cache")
print("     - offload_folder=Y:\\Research_Windows\\huggingface_cache\\offload")
print("\nAll Hugging Face models will now download to Y:\\Research_Windows\\huggingface_cache")
print(f"\nNotebook saved: {notebook_path}")
