# DenseCap GRiT-GritLM Fusion

Dense Captioning model that fuses **GRiT** (ViT-B visual encoder) with **GritLM-7B** (LLM decoder + cross-attention adapters) for region-level image captioning on Visual Genome.

# Important files to look at

[Notebook](https://github.com/AbdurRafay2004/Research_Windows/blob/main/densecap-grit-gritlm-fusion%20(1).ipynb)

[Changelog](https://github.com/AbdurRafay2004/Research_Windows/blob/main/CHANGELOG.md)

[BestPractices](https://github.com/AbdurRafay2004/Research_Windows/blob/main/BestPractices.md)

[README](https://github.com/AbdurRafay2004/Research_Windows/blob/main/README.md)

[requirements.txt](https://github.com/AbdurRafay2004/Research_Windows/blob/main/requirements.txt)

[Image of the file structure](https://github.com/AbdurRafay2004/Research_Windows/blob/main/file_structure_image.png)

## Current Status

✅ **Environment Setup Complete** — Ready to train on local GPU.

| Component | Version |
|-----------|---------|
| Python | 3.11.9 (venv at `.venv/`) |
| PyTorch | 2.11.0.dev+cu128 (Blackwell) |
| detectron2 | Built from source (MSVC) |
| GPU | RTX 5060 Ti (16 GB) |
| Kernel | `densecap-gpu` |

## Quick Start

1. **Activate the virtual environment:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Open the notebook and select kernel `DenseCap GPU (Python 3.11)`**

3. **Run cells in order** — Cell 1 verifies all packages, Cell 2 checks GPU

## Project Structure

```
Research_Windows/
├── densecap-grit-gritlm-fusion (1).ipynb  # Main notebook (modified for Windows)
├── data/
│   ├── VG_100K/              # Visual Genome images (part 1)
│   ├── VG_100K_2/            # Visual Genome images (part 2)
│   └── region_descriptions.json  # Region caption annotations
├── output/
│   ├── outputs/              # Training outputs, plots
│   └── checkpoints/          # Model checkpoints
├── pretrained_models/        # Downloaded GRiT checkpoints
├── .venv/                    # Python 3.11 virtual environment
├── _detectron2_build/        # detectron2 source (editable install)
├── requirements.txt          # Pip dependencies
├── build_detectron2.bat      # MSVC build script for detectron2
├── modify_notebook.py        # Script that patched the notebook
└── README.md
```

## Immediate Next Steps

- [ ] Run Cell 1-3 to verify environment and data
- [ ] Run Cell 4-7 to load data and create datasets
- [ ] Run Cell 8-10 to download GRiT checkpoint and load model
- [ ] Run Cell 11-15 to initialize complete model pipeline
- [ ] Run Cell 16-20 to train the model
- [ ] Run Cell 21-26 for evaluation and inference

## Known Issues / Notes

- **bitsandbytes on Windows**: May not fully support 4-bit quantization on Windows. The notebook includes a fallback to fp16 if bitsandbytes fails. fp16 uses more VRAM (~14GB vs ~3.5GB for 4-bit).
- **NUM_WORKERS**: Set to 0 on Windows to avoid multiprocessing issues with DataLoader.
- **detectron2**: Built without custom CUDA extensions (CPU fallback). Core GPU operations still use PyTorch CUDA.
- **GRiT checkpoint**: ~500MB download, requires internet on first run.
- **GritLM-7B**: ~14GB download from HuggingFace on first run.
