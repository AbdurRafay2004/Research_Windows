# Changelog

## 2026-02-12

### Environment Setup
- Installed Python 3.11.9 via winget (system had 3.14/3.13 which are incompatible with ML stack)
- Created virtual environment at `.venv/` with Python 3.11.9
- Installed PyTorch 2.11.0.dev+cu128 nightly (required for RTX 5060 Ti Blackwell GPU)
- Built and installed detectron2 from source using MSVC Build Tools 2022
- Installed all remaining dependencies: transformers 4.36.0, timm, einops, accelerate, bitsandbytes, opencv, pycocotools, lvis, scikit-learn, nltk, matplotlib, jupyter, ipykernel
- Registered Jupyter kernel `densecap-gpu` ("DenseCap GPU (Python 3.11)")

### Notebook Modifications (via `modify_notebook.py`)
- Cell 1: Replaced `!pip install` commands with import verification block
- Cell 3: Changed hardcoded paths to `os.path.join()` with auto-detected project root
- Cell 8: Removed `!wget` fallback (not available on Windows)
- Cell 9: Replaced `!git clone` shell commands with `subprocess.run()` for Windows
- Cell 13: Added bitsandbytes availability check with fp16 fallback for Windows
- Cell 16: Set `NUM_WORKERS = 0` for Windows DataLoader compatibility
- Cell 17: Updated deprecated `from torch.cuda.amp import` to `from torch.amp import`
- Cell 18: Fixed `autocast()` to `autocast('cuda')`, fixed `visual_encoder.predictor.model.eval()` reference
- Cell 19: Fixed `visual_encoder.predictor.model.eval()` reference in validate function
- Cell 26: Fixed `/tmp/` path to use `os.environ['TEMP']` on Windows
- Metadata: Updated kernel to `densecap-gpu`, removed Kaggle-specific metadata

- replaced `region_id` with `id` at cell 4 and 5 because of keyerror `region_id` at cell 4

### New Files
- `requirements.txt` — Pinned dependency list
- `build_detectron2.bat` — MSVC batch file for building detectron2
- `modify_notebook.py` — Script that patched the notebook for Windows
- `README.md` — Project documentation with status and quick start
- `CHANGELOG.md` — This file

## 2026-02-25

### Path Fixes (via `fix_notebook_paths.py`)
- **Cell 3**: Fixed `PROJECT_ROOT` detection — replaced `os.path.abspath('__file__')` (string literal bug) with `os.getcwd()` fallback for Jupyter notebooks
- **Cell 8**: Fixed `MODEL_DIR` to use `os.path.join(PROJECT_ROOT, "pretrained_models")` instead of relative path
- **Cell 8**: Fixed `checkpoint_path` to use `os.path.join()` instead of f-string with forward slash
- **Cell 9**: Fixed `GRIT_REPO_PATH` to use `os.path.join(PROJECT_ROOT, 'GRiT')` instead of relative path
- **Cell 25 (Critical)**: Fixed `temp_path` — `os.path.join()` was incorrectly wrapped in quotes as a string literal instead of executable code

### New Files
- `fix_notebook_paths.py` — Script to fix path issues in the notebook

### Missing Class Definitions (via `add_missing_classes.py`)
- **Cell 12 (New)**: Added missing `CrossAttentionAdapter` class — cross-attention module for injecting visual features into LLM layers
- **Cell 12 (New)**: Added missing `VisualProjector` class — projects GRiT visual features (768 dim) to LLM space (4096 dim) with spatial compression (196 -> 32 tokens)
- These classes were referenced in Cell 13 and Cell 15 but were never defined, causing `NameError: name 'VisualProjector' is not defined`

### New Files
- `add_missing_classes.py` — Script to add missing class definitions to the notebook

### Hugging Face Cache Configuration
- **Cell 2**: Added `HF_HOME` and `HUGGINGFACE_HUB_CACHE` environment variables to redirect Hugging Face downloads from C: drive to `Y:\Research_Windows\huggingface_cache`
- This prevents the C: drive from filling up when downloading large models like GritLM-7B (~14 GB)

### GritLMWithVision Class Fix
- **Cell 13**: Removed duplicate `GritLMWithVision` class definition (Cell 13 was duplicated)
- **Cell 13**: Added Windows compatibility check for bitsandbytes — falls back to fp16 if 4-bit quantization fails
- Fixed `ValueError: '.to' is not supported for '4-bit' or '8-bit' bitsandbytes models` by ensuring model is loaded directly to correct device without `.to()` calls

### New Files
- `edit_notebook_cache.py` — Script to update Hugging Face cache path in notebook
- `fix_gritlm_error.py` — Script to fix duplicate GritLMWithVision class and bitsandbytes compatibility
