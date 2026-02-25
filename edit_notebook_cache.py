"""
Script to update Cell 2 in the notebook to use Y:\Research_Windows\huggingface_cache
"""

import json

# Read the notebook
notebook_path = 'densecap-grit-gritlm-fusion (1).ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Fix Cell 2 source - replace the entire source with corrected version
new_cell_2_source = [
    "# Cell 2: Import all necessary libraries\n",
    "\n",
    "# Set Hugging Face cache to project directory (avoid filling C drive)\n",
    "import os\n",
    "os.environ['HF_HOME'] = r'Y:\\Research_Windows\\huggingface_cache'\n",
    "os.environ['HUGGINGFACE_HUB_CACHE'] = r'Y:\\Research_Windows\\huggingface_cache'\n",
    "\n",
    "import json\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "from PIL import Image\n",
    "import numpy as np\n",
    "from pathlib import Path\n",
    "from typing import Dict, List, Tuple, Optional\n",
    "from dataclasses import dataclass\n",
    "import logging\n",
    "from tqdm.auto import tqdm\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# Transformers & HuggingFace\n",
    "from transformers import (\n",
    "    AutoTokenizer, \n",
    "    AutoModelForCausalLM,\n",
    "    BitsAndBytesConfig\n",
    ")\n",
    "\n",
    "# Detectron2\n",
    "import detectron2\n",
    "from detectron2.config import get_cfg\n",
    "from detectron2.engine import DefaultPredictor\n",
    "from detectron2.utils.visualizer import Visualizer\n",
    "from detectron2.data import MetadataCatalog\n",
    "from detectron2.structures import Boxes, Instances\n",
    "\n",
    "# Set up logging\n",
    "logging.basicConfig(level=logging.INFO)\n",
    "logger = logging.getLogger(__name__)\n",
    "\n",
    "# Check GPU\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "print(f\"✓ Using device: {device}\")\n",
    "print(f\"✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}\")\n",
    "print(f\"✓ CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\")"
]

# Update Cell 2 (index 1)
notebook['cells'][1]['source'] = new_cell_2_source

# Save the modified notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Updated Cell 2:")
print("  - Cache path set to Y:\\Research_Windows\\huggingface_cache")
print(f"\nNotebook saved: {notebook_path}")
