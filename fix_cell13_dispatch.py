"""
Script to fix Cell 13 in the notebook by adding the dispatch_model monkey-patch
for bitsandbytes 4-bit quantization compatibility.
"""

import json

NOTEBOOK_PATH = r'Y:\Research_Windows\densecap-grit-gritlm-fusion (1).ipynb'

# New Cell 13 source code with the fix
NEW_CELL_13_SOURCE = '''# Cell 13: Load GritLM and inject cross-attention adapters
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
            
            # FIX: Monkey-patch dispatch_model to force hook-based dispatch
            # This avoids the .to(device) call that transformers blocks for bnb models
            import accelerate.big_modeling as _accel_bm
            import transformers.modeling_utils as _tf_mu
            _orig_dispatch_accel = _accel_bm.dispatch_model
            _orig_dispatch_tf = _tf_mu.dispatch_model
            
            def _patched_dispatch(model, *args, **kwargs):
                kwargs['force_hooks'] = True
                return _orig_dispatch_accel(model, *args, **kwargs)
            
            _accel_bm.dispatch_model = _patched_dispatch
            _tf_mu.dispatch_model = _patched_dispatch
            
            try:
                self.llm = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=CACHE_DIR,
                    offload_folder=OFFLOAD_DIR,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    max_memory={0: "15GB"},  # Limit GPU memory usage
                )
            finally:
                # Restore original dispatch_model
                _accel_bm.dispatch_model = _orig_dispatch_accel
                _tf_mu.dispatch_model = _orig_dispatch_tf
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

def main():
    # Load notebook
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find Cell 13 (the one with GritLMWithVision class)
    cell_13_index = None
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'class GritLMWithVision(nn.Module):' in source and 'Cell 13:' in source:
                cell_13_index = i
                break
    
    if cell_13_index is None:
        print("ERROR: Could not find Cell 13 with GritLMWithVision class")
        return
    
    print(f"Found Cell 13 at index {cell_13_index}")
    
    # Update the cell source
    notebook['cells'][cell_13_index]['source'] = NEW_CELL_13_SOURCE.split('\n')
    # Add newline to each line except the last
    notebook['cells'][cell_13_index]['source'] = [
        line + '\n' for line in NEW_CELL_13_SOURCE.split('\n')[:-1]
    ] + [NEW_CELL_13_SOURCE.split('\n')[-1]]
    
    # Clear outputs for this cell
    notebook['cells'][cell_13_index]['outputs'] = []
    notebook['cells'][cell_13_index]['execution_count'] = None
    
    # Save notebook
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print("[OK] Cell 13 updated successfully with dispatch_model fix")
    print("\nChanges made:")
    print("  1. Added monkey-patch for dispatch_model (accelerate + transformers)")
    print("  2. Added try/finally to restore original functions")
    print("  3. Added max_memory={0: '15GB'} to prevent OOM")

if __name__ == '__main__':
    main()
