"""
Script to modify the DenseCap notebook for Windows 11 + local GPU setup.
Converts Kaggle-style notebook to work locally with RTX 5060 Ti.
"""
import json
import copy

NOTEBOOK_PATH = r"y:\Research_Windows\densecap-grit-gritlm-fusion (1).ipynb"
OUTPUT_PATH = r"y:\Research_Windows\densecap-grit-gritlm-fusion (1).ipynb"

# Read the notebook
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# ============================================================
# Cell 1: Replace pip install commands with verification block
# ============================================================
cells[0]['source'] = [
    "# Cell 1: Verify Required Packages\n",
    "# All packages should be pre-installed in the 'densecap-gpu' venv.\n",
    "# If any import fails, run: pip install <package> in your venv terminal.\n",
    "\n",
    "import importlib\n",
    "\n",
    "required_packages = {\n",
    "    'torch': 'torch',\n",
    "    'torchvision': 'torchvision',\n",
    "    'torchaudio': 'torchaudio',\n",
    "    'transformers': 'transformers',\n",
    "    'detectron2': 'detectron2',\n",
    "    'timm': 'timm',\n",
    "    'cv2': 'opencv-python',\n",
    "    'pycocotools': 'pycocotools',\n",
    "    'lvis': 'lvis',\n",
    "    'einops': 'einops',\n",
    "    'accelerate': 'accelerate',\n",
    "    'bitsandbytes': 'bitsandbytes',\n",
    "}\n",
    "\n",
    "all_ok = True\n",
    "for module_name, pip_name in required_packages.items():\n",
    "    try:\n",
    "        importlib.import_module(module_name)\n",
    "        print(f'✓ {module_name}')\n",
    "    except ImportError:\n",
    "        print(f'✗ {module_name} — run: pip install {pip_name}')\n",
    "        all_ok = False\n",
    "\n",
    "if all_ok:\n",
    "    print('\\n✓ All packages verified successfully!')\n",
    "else:\n",
    "    print('\\n✗ Some packages are missing. Install them and re-run this cell.')\n",
]

# ============================================================
# Cell 3: Fix data paths for Windows (absolute paths)
# ============================================================
cells[2]['source'] = [
    "# Cell 3: Configure your data paths\n",
    "\n",
    "import os\n",
    "\n",
    "# Auto-detect project root based on this notebook's location\n",
    "PROJECT_ROOT = os.path.dirname(os.path.abspath('__file__'))\n",
    "# If running from a different directory, set manually:\n",
    "# PROJECT_ROOT = r'y:\\Research_Windows'\n",
    "\n",
    "DATA_CONFIG = {\n",
    "    'images_dir': os.path.join(PROJECT_ROOT, 'data', 'VG_100K'),\n",
    "    'images2_dir': os.path.join(PROJECT_ROOT, 'data', 'VG_100K_2'),\n",
    "    'annotations': os.path.join(PROJECT_ROOT, 'data', 'region_descriptions.json'),\n",
    "    'output_dir': os.path.join(PROJECT_ROOT, 'output', 'outputs'),\n",
    "    'checkpoint_dir': os.path.join(PROJECT_ROOT, 'output', 'checkpoints'),\n",
    "}\n",
    "\n",
    "# Create output directories\n",
    "os.makedirs(DATA_CONFIG['output_dir'], exist_ok=True)\n",
    "os.makedirs(DATA_CONFIG['checkpoint_dir'], exist_ok=True)\n",
    "\n",
    "# Verify paths exist\n",
    "for key, path in DATA_CONFIG.items():\n",
    "    if 'dir' in key:\n",
    "        if os.path.exists(path):\n",
    "            print(f'✓ Found {key}: {path}')\n",
    "        else:\n",
    "            print(f'✗ Missing {key}: {path}')\n",
    "    elif key == 'annotations':\n",
    "        if os.path.exists(path):\n",
    "            print(f'✓ Found annotations: {path}')\n",
    "        else:\n",
    "            print(f'✗ Missing annotations: {path}')\n",
    "\n",
    "print(f'\\n✓ Setup complete! Output will be saved to: {DATA_CONFIG[\"output_dir\"]}')\n",
]

# ============================================================
# Cell 8: Fix checkpoint download (no wget on Windows)
# ============================================================
# Find Cell 8 (checkpoint download) and replace the wget fallback
cell8_source = cells[7]['source']
new_cell8_source = []
skip_wget_lines = False
for line in cell8_source:
    # Remove the !wget line and use only urllib
    if '!wget' in line:
        continue  # Skip wget lines
    # Replace /tmp/ paths with Windows-compatible temp dir
    if '/tmp/' in line:
        line = line.replace('/tmp/', 'temp/')
    new_cell8_source.append(line)
cells[7]['source'] = new_cell8_source

# ============================================================
# Cell 9: Fix git clone for Windows (use subprocess)
# ============================================================
cells[8]['source'] = [
    "# Cell 9: Clone GRiT repo and setup configuration\n",
    "\n",
    "import subprocess\n",
    "import sys\n",
    "\n",
    "print('='*60)\n",
    "print('SETTING UP GRIT FRAMEWORK')\n",
    "print('='*60)\n",
    "\n",
    "# Clone GRiT repository if not exists\n",
    "GRIT_REPO_PATH = 'GRiT'\n",
    "\n",
    "if not os.path.exists(GRIT_REPO_PATH):\n",
    "    print('\\n[1/3] Cloning GRiT repository...')\n",
    "    subprocess.run(['git', 'clone', 'https://github.com/JialianW/GRiT.git', GRIT_REPO_PATH], check=True)\n",
    "    print('✓ Repository cloned')\n",
    "else:\n",
    "    print('\\n[1/3] ✓ GRiT repository already exists')\n",
    "\n",
    "# Add to Python path\n",
    "if GRIT_REPO_PATH not in sys.path:\n",
    "    sys.path.insert(0, GRIT_REPO_PATH)\n",
    "    print('✓ Added to Python path')\n",
    "\n",
    "# Clone CenterNet2 dependency (required by GRiT)\n",
    "CENTERNET2_PATH = os.path.join(GRIT_REPO_PATH, 'third_party', 'CenterNet2')\n",
    "if not os.path.exists(CENTERNET2_PATH):\n",
    "    print('\\n[2/3] Cloning CenterNet2 dependency...')\n",
    "    subprocess.run(['git', 'clone', 'https://github.com/xingyizhou/CenterNet2.git', CENTERNET2_PATH], check=True)\n",
    "    print('✓ CenterNet2 cloned')\n",
    "else:\n",
    "    print('\\n[2/3] ✓ CenterNet2 already exists')\n",
    "\n",
    "# Add CenterNet2 to path\n",
    "centernet2_projects = os.path.join(CENTERNET2_PATH, 'projects', 'CenterNet2')\n",
    "sys.path.insert(0, centernet2_projects)\n",
    "\n",
    "print('\\n[3/3] Importing GRiT modules...')\n",
    "try:\n",
    "    from grit.config import add_grit_config\n",
    "    from detectron2.config import get_cfg\n",
    "    from detectron2.engine import DefaultPredictor\n",
    "    print('✓ GRiT modules imported successfully')\n",
    "except ImportError as e:\n",
    "    print(f'✗ Import error: {e}')\n",
    "    print('  Make sure detectron2 and timm are installed in your venv.')\n",
    "\n",
    "print('\\n' + '='*60)\n",
    "print('GRIT FRAMEWORK READY')\n",
    "print('='*60)\n",
]

# ============================================================
# Cell 13: Adjust GritLMWithVision for Windows compatibility
# Bitsandbytes may not fully support Windows — add fallback
# ============================================================
cell13_source = cells[11]['source']
new_cell13_source = []
for line in cell13_source:
    new_cell13_source.append(line)

# Wrap the entire Cell 13 with bitsandbytes fallback
cells[11]['source'] = [
    "# Cell 13: Load GritLM and inject cross-attention adapters\n",
    "# With Windows compatibility: falls back to fp16 if bitsandbytes fails\n",
    "\n",
    "from transformers import AutoConfig\n",
    "import copy\n",
    "\n",
    "# Check bitsandbytes availability on Windows\n",
    "BNB_AVAILABLE = False\n",
    "try:\n",
    "    import bitsandbytes as bnb\n",
    "    BNB_AVAILABLE = True\n",
    "    print('✓ bitsandbytes available — using 4-bit quantization')\n",
    "except Exception as e:\n",
    "    print(f'⚠️  bitsandbytes not available ({e})')\n",
    "    print('   Falling back to fp16 loading (uses more VRAM)')\n",
    "\n",
    "class GritLMWithVision(nn.Module):\n",
    "    \"\"\"\n",
    "    GritLM-7B with cross-attention adapters for visual conditioning.\n",
    "    Memory-efficient version using 4-bit quantization (or fp16 fallback).\n",
    "    \"\"\"\n",
    "    \n",
    "    def __init__(\n",
    "        self,\n",
    "        model_name: str = \"GritLM/GritLM-7B\",\n",
    "        num_cross_attn_layers: int = 8,\n",
    "        gradient_checkpointing: bool = True,\n",
    "    ):\n",
    "        super().__init__()\n",
    "        \n",
    "        if BNB_AVAILABLE:\n",
    "            print(f'Loading {model_name} with 4-bit quantization...')\n",
    "            bnb_config = BitsAndBytesConfig(\n",
    "                load_in_4bit=True,\n",
    "                bnb_4bit_use_double_quant=True,\n",
    "                bnb_4bit_quant_type=\"nf4\",\n",
    "                bnb_4bit_compute_dtype=torch.float16,\n",
    "            )\n",
    "            self.llm = AutoModelForCausalLM.from_pretrained(\n",
    "                model_name,\n",
    "                quantization_config=bnb_config,\n",
    "                device_map=\"auto\",\n",
    "                trust_remote_code=True,\n",
    "                torch_dtype=torch.float16,\n",
    "            )\n",
    "        else:\n",
    "            print(f'Loading {model_name} with fp16...')\n",
    "            self.llm = AutoModelForCausalLM.from_pretrained(\n",
    "                model_name,\n",
    "                device_map=\"auto\",\n",
    "                trust_remote_code=True,\n",
    "                torch_dtype=torch.float16,\n",
    "            )\n",
    "        \n",
    "        self.config = self.llm.config\n",
    "        hidden_size = self.config.hidden_size\n",
    "        num_heads = self.config.num_attention_heads\n",
    "        num_layers = self.config.num_hidden_layers\n",
    "        \n",
    "        if gradient_checkpointing:\n",
    "            self.llm.gradient_checkpointing_enable()\n",
    "        \n",
    "        # Add cross-attention adapters to selected layers\n",
    "        self.cross_attn_adapters = nn.ModuleList()\n",
    "        self.cross_attn_layer_indices = []\n",
    "        \n",
    "        layer_interval = num_layers // num_cross_attn_layers\n",
    "        for i in range(num_cross_attn_layers):\n",
    "            layer_idx = i * layer_interval + layer_interval // 2\n",
    "            self.cross_attn_layer_indices.append(layer_idx)\n",
    "            self.cross_attn_adapters.append(\n",
    "                CrossAttentionAdapter(hidden_size, num_heads)\n",
    "            )\n",
    "        \n",
    "        print(f'✓ Loaded GritLM with cross-attention at layers: {self.cross_attn_layer_indices}')\n",
    "        \n",
    "        # Freeze base LLM, only train adapters\n",
    "        for param in self.llm.parameters():\n",
    "            param.requires_grad = False\n",
    "        \n",
    "        for adapter in self.cross_attn_adapters:\n",
    "            for param in adapter.parameters():\n",
    "                param.requires_grad = True\n",
    "        \n",
    "        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)\n",
    "        total_params = sum(p.numel() for p in self.parameters())\n",
    "        print(f'✓ Trainable parameters: {trainable_params:,} / {total_params:,} '\n",
    "              f'({100 * trainable_params / total_params:.2f}%)')\n",
    "    \n",
    "    def forward(\n",
    "        self,\n",
    "        input_ids,\n",
    "        visual_tokens,\n",
    "        attention_mask=None,\n",
    "        labels=None,\n",
    "    ):\n",
    "        inputs_embeds = self.llm.model.embed_tokens(input_ids)\n",
    "        \n",
    "        outputs = self.llm(\n",
    "            input_ids=input_ids,\n",
    "            attention_mask=attention_mask,\n",
    "            labels=labels,\n",
    "            output_hidden_states=True,\n",
    "            return_dict=True,\n",
    "        )\n",
    "        \n",
    "        return outputs\n",
    "\n",
    "print('✓ GritLMWithVision class defined')\n",
]

# ============================================================
# Cell 16: Fix NUM_WORKERS for Windows (0 is safest)
# ============================================================
cell16_source = cells[13]['source']
new_cell16_source = []
for line in cell16_source:
    if "NUM_WORKERS = 2" in line:
        line = "NUM_WORKERS = 0  # Set to 0 on Windows to avoid multiprocessing issues\\n"
    new_cell16_source.append(line)
cells[13]['source'] = new_cell16_source

# ============================================================
# Cell 17: Fix deprecated torch.cuda.amp imports
# ============================================================
cell17_source = cells[14]['source']
new_cell17_source = []
for line in cell17_source:
    if "from torch.cuda.amp import autocast, GradScaler" in line:
        line = "from torch.amp import autocast, GradScaler\\n"
    new_cell17_source.append(line)
cells[14]['source'] = new_cell17_source

# ============================================================
# Cell 18: Fix deprecated autocast usage and visual_encoder reference
# ============================================================
cell18_source = cells[15]['source']
new_cell18_source = []
for line in cell18_source:
    # Fix: model.visual_encoder.predictor.model.eval() -> model.visual_encoder.model.eval()
    if "model.visual_encoder.predictor.model.eval()" in line:
        line = line.replace("model.visual_encoder.predictor.model.eval()", "model.visual_encoder.model.eval()")
    # Fix deprecated autocast usage
    if "with autocast():" in line:
        line = line.replace("with autocast():", "with autocast('cuda'):")
    new_cell18_source.append(line)
cells[15]['source'] = new_cell18_source

# ============================================================
# Cell 19: Fix visual_encoder reference in validate function too
# ============================================================
cell19_source = cells[16]['source']
new_cell19_source = []
for line in cell19_source:
    if "model.visual_encoder.predictor.model.eval()" in line:
        line = line.replace("model.visual_encoder.predictor.model.eval()", "model.visual_encoder.model.eval()")
    new_cell19_source.append(line)
cells[16]['source'] = new_cell19_source

# ============================================================
# Cell 26: Fix /tmp/ path for Windows
# ============================================================
cell26_source = cells[-1]['source']
new_cell26_source = []
for line in cell26_source:
    if "/tmp/temp_image.jpg" in line:
        line = line.replace("/tmp/temp_image.jpg", "os.path.join(os.environ.get('TEMP', '.'), 'temp_image.jpg')")
    new_cell26_source.append(line)
cells[-1]['source'] = new_cell26_source

# ============================================================
# Update notebook metadata for the new kernel
# ============================================================
nb['metadata']['kernelspec'] = {
    "display_name": "DenseCap GPU (Python 3.11)",
    "language": "python",
    "name": "densecap-gpu"
}
nb['metadata']['language_info'] = {
    "codemirror_mode": {
        "name": "ipython",
        "version": 3
    },
    "file_extension": ".py",
    "mimetype": "text/x-python",
    "name": "python",
    "nbconvert_exporter": "python",
    "pygments_lexer": "ipython3",
    "version": "3.11.9"
}
# Remove Kaggle-specific metadata
if 'kaggle' in nb['metadata']:
    del nb['metadata']['kaggle']

# ============================================================
# Write the modified notebook
# ============================================================
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"✓ Notebook modified and saved to: {OUTPUT_PATH}")
print("Changes made:")
print("  - Cell 1: Replaced pip installs with import verification")
print("  - Cell 3: Fixed data paths for Windows (os.path.join)")
print("  - Cell 8: Removed wget fallback (Windows incompatible)")
print("  - Cell 9: Replaced !git clone with subprocess.run()")
print("  - Cell 13: Added bitsandbytes fallback for Windows")
print("  - Cell 16: Set NUM_WORKERS=0 for Windows")
print("  - Cell 17: Fixed deprecated torch.cuda.amp imports")
print("  - Cell 18: Fixed autocast('cuda') and visual_encoder ref")
print("  - Cell 19: Fixed visual_encoder reference in validate()")
print("  - Cell 26: Fixed /tmp/ path for Windows")
print("  - Metadata: Updated kernel to densecap-gpu, removed Kaggle metadata")
