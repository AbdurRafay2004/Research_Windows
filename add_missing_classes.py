"""
Script to add missing class definitions to the notebook:
- VisualProjector
- CrossAttentionAdapter

These classes are referenced in Cell 13 and Cell 15 but were never defined.
"""

import json

def add_missing_classes():
    notebook_path = 'densecap-grit-gritlm-fusion (1).ipynb'
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # The missing class definitions to add
    missing_classes_code = '''# Cell 12: Define Visual Projector and Cross-Attention Adapter

import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionAdapter(nn.Module):
    """
    Cross-attention adapter for injecting visual features into LLM layers.
    """
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
    def forward(self, text_hidden, visual_tokens):
        """
        Args:
            text_hidden: [batch, seq_len, hidden] - text embeddings
            visual_tokens: [batch, num_visual, hidden] - visual features
        
        Returns:
            adapted: [batch, seq_len, hidden] - text with visual conditioning
        """
        # Cross-attention: text attends to visual tokens
        residual = text_hidden
        text_hidden = self.norm1(text_hidden)
        
        # Cross-attention
        attn_output, _ = self.cross_attn(
            query=text_hidden,
            key=visual_tokens,
            value=visual_tokens
        )
        text_hidden = residual + attn_output
        
        # FFN
        residual = text_hidden
        text_hidden = self.norm2(text_hidden)
        text_hidden = residual + self.ffn(text_hidden)
        
        return text_hidden


class VisualProjector(nn.Module):
    """
    Projects visual features from GRiT (768 dim) to LLM space (4096 dim).
    Also compresses spatial tokens (196 -> 32).
    """
    def __init__(
        self, 
        visual_dim=768, 
        llm_dim=4096, 
        num_visual_tokens=32
    ):
        super().__init__()
        self.num_visual_tokens = num_visual_tokens
        
        # Spatial compression: 196 tokens -> 32 tokens
        self.spatial_compressor = nn.Sequential(
            nn.Linear(196, 128),
            nn.GELU(),
            nn.Linear(128, num_visual_tokens)
        )
        
        # Feature projection: 768 -> 4096
        self.feature_proj = nn.Sequential(
            nn.Linear(visual_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )
        
    def forward(self, visual_features):
        """
        Args:
            visual_features: [batch, 196, 768] - GRiT region features
        
        Returns:
            projected: [batch, 32, 4096] - LLM-ready visual tokens
        """
        batch_size = visual_features.shape[0]
        
        # Transpose for spatial compression: [batch, 768, 196]
        visual_features = visual_features.permute(0, 2, 1)
        
        # Compress spatial dimension: [batch, 768, 196] -> [batch, 768, 32]
        compressed = self.spatial_compressor(visual_features)
        
        # Transpose back: [batch, 32, 768]
        compressed = compressed.permute(0, 2, 1)
        
        # Project to LLM dimension: [batch, 32, 4096]
        projected = self.feature_proj(compressed)
        
        return projected


print("CrossAttentionAdapter defined")
print("VisualProjector defined")
'''

    # Find the position to insert (after Cell 11 - GRiTFeatureExtractor)
    insert_position = None
    for idx, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            if 'Cell 11:' in source or 'GRiTFeatureExtractor' in source:
                insert_position = idx + 1
                break
    
    if insert_position is None:
        # Find Cell 12 or Cell 13
        for idx, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
                if 'Cell 13:' in source:
                    insert_position = idx
                    break
    
    if insert_position is None:
        print("Could not find insertion point!")
        return False
    
    # Check if the cell already exists
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            if 'class VisualProjector' in source and 'class CrossAttentionAdapter' in source:
                print("Classes already exist in notebook")
                return True
    
    # Create new cell
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {
            "trusted": True
        },
        "outputs": [],
        "source": missing_classes_code.split('\n')
    }
    
    # Convert source to list format with newlines
    lines = missing_classes_code.split('\n')
    new_cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]
    
    # Insert the cell
    notebook['cells'].insert(insert_position, new_cell)
    
    # Write the modified notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"Added missing class definitions at position {insert_position}")
    return True

if __name__ == '__main__':
    success = add_missing_classes()
    if success:
        print("Notebook updated successfully")
    else:
        print("Failed to update notebook")
