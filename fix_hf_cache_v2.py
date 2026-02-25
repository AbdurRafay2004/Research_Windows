r"""
Fix HuggingFace cache STILL going to C:\Users\abdur\.cache\huggingface

ROOT CAUSE:
  Cell 2 sets HF_HOME, HUGGINGFACE_HUB_CACHE, TRANSFORMERS_CACHE
  BUT is missing two critical variables:
    1. HF_HUB_CACHE  - the CURRENT env var for hub downloads (the old name 
       HUGGINGFACE_HUB_CACHE is deprecated and ignored by newer huggingface_hub)
    2. HF_MODULES_CACHE - controls where custom code (modeling_gritlm7b.py etc.)
       is downloaded when trust_remote_code=True

  Also, HF_HUB_CACHE should point to cache_dir/hub (not cache_dir directly),
  matching HuggingFace's expected directory structure.

WHAT THIS SCRIPT DOES:
  Updates Cell 2 to set ALL required env vars with correct paths,
  and monkey-patches huggingface_hub.constants at runtime.

Usage:
  python fix_hf_cache_v2.py
  Then restart kernel and re-run all cells.
"""

import json

notebook_path = r'y:\Research_Windows\densecap-grit-gritlm-fusion (1).ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# ============================================================================
# Fix Cell 2 (index 1) - Set ALL cache env vars with CORRECT paths
# ============================================================================
new_cell_2_source = [
    "# Cell 2: Import all necessary libraries\n",
    "\n",
    "# ============================================================\n",
    "# CRITICAL: Set ALL HuggingFace cache env vars BEFORE imports\n",
    "# ============================================================\n",
    "import os\n",
    "\n",
    "# Base cache directory on Y: drive\n",
    "cache_dir = r'Y:\\Research_Windows\\huggingface_cache'\n",
    "os.makedirs(cache_dir, exist_ok=True)\n",
    "\n",
    "# Sub-directories matching HuggingFace's expected structure\n",
    "hub_cache = os.path.join(cache_dir, 'hub')\n",
    "modules_cache = os.path.join(cache_dir, 'modules')\n",
    "offload_dir = os.path.join(cache_dir, 'offload')\n",
    "os.makedirs(hub_cache, exist_ok=True)\n",
    "os.makedirs(modules_cache, exist_ok=True)\n",
    "os.makedirs(offload_dir, exist_ok=True)\n",
    "\n",
    "# Set ALL cache environment variables\n",
    "os.environ['HF_HOME'] = cache_dir                    # Master switch\n",
    "os.environ['HF_HUB_CACHE'] = hub_cache               # Where model repos are cloned (CURRENT var name)\n",
    "os.environ['HUGGINGFACE_HUB_CACHE'] = hub_cache       # Old name, kept for compat\n",
    "os.environ['TRANSFORMERS_CACHE'] = hub_cache           # Transformers-specific\n",
    "os.environ['HF_MODULES_CACHE'] = modules_cache        # Where trust_remote_code downloads go\n",
    "os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_dir, 'datasets')\n",
    "os.environ['HF_METRICS_CACHE'] = cache_dir\n",
    "os.environ['ACCELERATE_CACHE'] = cache_dir\n",
    "os.environ['TORCH_HOME'] = cache_dir\n",
    "\n",
    "# Now import other libraries\n",
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
    "# Force huggingface_hub to use our paths (belt + suspenders)\n",
    "try:\n",
    "    import huggingface_hub.constants as hf_constants\n",
    "    hf_constants.HF_HOME = cache_dir\n",
    "    hf_constants.HF_HUB_CACHE = hub_cache\n",
    "    hf_constants.HUGGINGFACE_HUB_CACHE = hub_cache\n",
    "    hf_constants.HF_MODULES_CACHE = modules_cache\n",
    "    if hasattr(hf_constants, 'default_home'):\n",
    "        hf_constants.default_home = cache_dir\n",
    "    if hasattr(hf_constants, 'default_cache_path'):\n",
    "        hf_constants.default_cache_path = hub_cache\n",
    "    print(f'\\u2713 Patched huggingface_hub.constants to use Y: drive')\n",
    "except Exception as e:\n",
    "    print(f'Warning: Could not patch huggingface_hub constants: {e}')\n",
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
    "print(f\"Using device: {device}\")\n",
    "print(f\"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}\")\n",
    "print(f\"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\")\n",
    "print(f\"HF Cache: {cache_dir}\")\n",
    "print(f\"HF Hub Cache: {hub_cache}\")\n",
    "print(f\"HF Modules Cache: {modules_cache}\")\n",
]

# Update Cell 2 (index 1)
notebook['cells'][1]['source'] = new_cell_2_source
# Clear old outputs so user re-runs cleanly
notebook['cells'][1]['outputs'] = []
notebook['cells'][1]['execution_count'] = None

# Save
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("=" * 60)
print("NOTEBOOK UPDATED SUCCESSFULLY")
print("=" * 60)
print()
print("Changes made to Cell 2:")
print("  1. Added HF_HUB_CACHE -> Y:\\Research_Windows\\huggingface_cache\\hub")
print("     (this is the CURRENT env var that huggingface_hub reads)")
print("  2. Added HF_MODULES_CACHE -> Y:\\Research_Windows\\huggingface_cache\\modules")
print("     (this controls where trust_remote_code=True downloads go)")
print("  3. Added runtime monkey-patch of huggingface_hub.constants")
print("     (ensures even already-imported modules respect the paths)")
print()
print("NEXT STEPS:")
print("  1. Restart the Jupyter kernel (Kernel -> Restart)")
print("  2. Re-run ALL cells from Cell 1 onwards")
print()
print(f"Notebook saved: {notebook_path}")
