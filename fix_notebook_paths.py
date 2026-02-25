"""
Script to fix path issues in densecap-grit-gritlm-fusion (1).ipynb

Issues fixed:
1. Line 162: '__file__' string literal instead of proper notebook path detection
2. Line 3546: os.path.join() incorrectly wrapped in quotes (critical bug)
3. Various relative paths improved with os.path.join() for cross-platform compatibility
"""

import json
import re

def fix_notebook_paths():
    notebook_path = 'densecap-grit-gritlm-fusion (1).ipynb'
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    fixes_made = []
    
    # Process each cell
    for cell_idx, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        
        source = cell['source']
        original_source = source.copy() if isinstance(source, list) else source
        
        # Convert to string for easier manipulation
        if isinstance(source, list):
            source_text = ''.join(source)
        else:
            source_text = source
        
        modified = False
        
        # Fix 1: PROJECT_ROOT with '__file__' string literal
        # Replace the incorrect usage with a proper approach for notebooks
        old_project_root = '''PROJECT_ROOT = os.path.dirname(os.path.abspath('__file__'))'''
        new_project_root = '''# For Jupyter notebooks, use current working directory
# or set manually if running from a different location
try:
    # Try to get notebook directory (works in some environments)
    import notebook
    PROJECT_ROOT = os.path.dirname(os.path.abspath(notebook.notebook_path))
except:
    # Fallback to current working directory
    PROJECT_ROOT = os.getcwd()'''
        
        if old_project_root in source_text:
            source_text = source_text.replace(old_project_root, new_project_root)
            fixes_made.append(f"Cell {cell_idx}: Fixed PROJECT_ROOT path detection")
            modified = True
        
        # Fix 2: Critical bug - temp_path with os.path.join inside quotes
        # Pattern: temp_path = 'os.path.join(...)'
        pattern = r'''temp_path\s*=\s*['\"]os\.path\.join\(os\.environ\.get\(['\"]TEMP['\"]\s*,\s*['\"]\.['\"]\)\s*,\s*['\"]temp_image\.jpg['\"]\)['\"]'''
        if re.search(pattern, source_text):
            source_text = re.sub(pattern, "temp_path = os.path.join(os.environ.get('TEMP', '.'), 'temp_image.jpg')", source_text)
            fixes_made.append(f"Cell {cell_idx}: Fixed critical bug - temp_path os.path.join was wrapped in quotes")
            modified = True
        
        # Fix 3: MODEL_DIR relative path (Cell 8)
        old_model_dir = 'MODEL_DIR = "pretrained_models"'
        new_model_dir = 'MODEL_DIR = os.path.join(PROJECT_ROOT, "pretrained_models")'
        
        if old_model_dir in source_text and new_model_dir not in source_text:
            source_text = source_text.replace(old_model_dir, new_model_dir)
            fixes_made.append(f"Cell {cell_idx}: Fixed MODEL_DIR to use PROJECT_ROOT")
            modified = True
        
        # Fix 4: GRIT_REPO_PATH relative path
        old_grit_repo = "GRIT_REPO_PATH = 'GRiT'"
        new_grit_repo = "GRIT_REPO_PATH = os.path.join(PROJECT_ROOT, 'GRiT')"
        
        if old_grit_repo in source_text and new_grit_repo not in source_text:
            source_text = source_text.replace(old_grit_repo, new_grit_repo)
            fixes_made.append(f"Cell {cell_idx}: Fixed GRIT_REPO_PATH to use PROJECT_ROOT")
            modified = True
        
        # Fix 5: checkpoint_path construction
        old_checkpoint_path = 'checkpoint_path = f"{MODEL_DIR}/{CHECKPOINT_NAME}.pth"'
        new_checkpoint_path = 'checkpoint_path = os.path.join(MODEL_DIR, f"{CHECKPOINT_NAME}.pth")'
        
        if old_checkpoint_path in source_text:
            source_text = source_text.replace(old_checkpoint_path, new_checkpoint_path)
            fixes_made.append(f"Cell {cell_idx}: Fixed checkpoint_path to use os.path.join")
            modified = True
        
        # Update the cell source if modified
        if modified:
            # Preserve the original format (list vs string)
            if isinstance(original_source, list):
                # Split back into lines, preserving line endings
                lines = source_text.split('\n')
                cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if len(lines) > 1 else [source_text]
            else:
                cell['source'] = source_text
    
    # Write the modified notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    # Print summary
    print("=" * 60)
    print("NOTEBOOK PATH FIXES")
    print("=" * 60)
    
    if fixes_made:
        print(f"\n[OK] Made {len(fixes_made)} fixes:\n")
        for fix in fixes_made:
            print(f"  - {fix}")
    else:
        print("\n[!] No fixes were needed - paths may already be correct or patterns not found")
    
    print("\n" + "=" * 60)
    
    return fixes_made

if __name__ == '__main__':
    fixes = fix_notebook_paths()
    print(f"\nTotal fixes applied: {len(fixes)}")
