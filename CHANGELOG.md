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

### New Files
- `requirements.txt` — Pinned dependency list
- `build_detectron2.bat` — MSVC batch file for building detectron2
- `modify_notebook.py` — Script that patched the notebook for Windows
- `README.md` — Project documentation with status and quick start
- `CHANGELOG.md` — This file
