# Best Practices

This document stores best practices and solutions to common issues encountered during development.

## Hugging Face Transformers + BitsAndBytes

### Issue: `.to()` not supported for 4-bit/8-bit models

**Error:**
```
ValueError: `.to` is not supported for `4-bit` or `8-bit` bitsandbytes models.
```

**Root Cause:**
- `transformers` 4.40.0+ blocks `.to()` calls on bitsandbytes quantized models
- `accelerate`'s `dispatch_model` function still attempts to call `.to()` when using `device_map="auto"`

**Solution:**
Monkey-patch `dispatch_model` to force `force_hooks=True` before loading the model:

```python
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
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        # ... other params
    )
finally:
    # Restore original functions
    _accel_bm.dispatch_model = _orig_dispatch_accel
    _tf_mu.dispatch_model = _orig_dispatch_tf
```

**Why it works:**
- `force_hooks=True` forces `accelerate` to use hook-based device dispatch
- This avoids the `.to(device)` call that transformers blocks for quantized models
- Must patch both `accelerate.big_modeling` AND `transformers.modeling_utils` because transformers imports `dispatch_model` directly into its own namespace

---

## Windows Development

### Path Handling

- Always use `os.path.join()` instead of hardcoded forward slashes or f-strings with `/`
- In Jupyter notebooks, use `os.getcwd()` instead of `os.path.abspath('__file__')` (the latter returns the literal string `'__file__'` in notebooks)

### DataLoader Workers

- Set `NUM_WORKERS = 0` in PyTorch DataLoaders on Windows to avoid multiprocessing issues

### Hugging Face Cache

- Set environment variables at the start of the notebook to redirect cache from C: drive:
  ```python
  import os
  cache_dir = r'Y:\Research_Windows\huggingface_cache'
  os.environ['HF_HOME'] = cache_dir
  os.environ['HF_HUB_CACHE'] = os.path.join(cache_dir, 'hub')
  os.environ['HF_MODULES_CACHE'] = os.path.join(cache_dir, 'modules')
  ```

---

## GPU Memory Management

### 4-bit Quantization on 16GB GPU

- Use `max_memory={0: "15GB"}` when loading large models to leave headroom for other operations
- Enable `gradient_checkpointing=True` to reduce memory during training
- Use `load_in_4bit=True` with `bnb_4bit_use_double_quant=True` for maximum memory savings

---

## Notebook Editing

Since `.ipynb` files are JSON and cannot be directly edited by the assistant, use Python scripts to modify them:

```python
import json

with open('notebook.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find and modify cells
for cell in notebook['cells']:
    if 'search_text' in ''.join(cell['source']):
        cell['source'] = new_source.split('\n')
        cell['source'] = [line + '\n' for line in new_source.split('\n')[:-1]] + [new_source.split('\n')[-1]]

with open('notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)
```
