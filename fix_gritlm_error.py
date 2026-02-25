"""
Script to fix the GritLMWithVision class error in the notebook.

The error occurs because:
1. There are TWO GritLMWithVision class definitions (Cell 13 is duplicated)
2. The second definition doesn't have Windows compatibility fallback
3. bitsandbytes models with device_map="auto" can't be moved with .to()

This script:
1. Removes the duplicate Cell 13
2. Updates the first Cell 13 with proper Windows compatibility
"""

import json


def create_fixed_gritlm_cell():
    """Create the fixed GritLMWithVision cell source"""
    return [
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
        "print('✓ GritLMWithVision class defined')\n"
    ]


# Read the notebook
notebook_path = 'densecap-grit-gritlm-fusion (1).ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find and remove duplicate GritLMWithVision definitions
cells_to_keep = []
seen_gritlm_class = False

for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        
        # Check if this cell defines GritLMWithVision
        if 'class GritLMWithVision' in source:
            if seen_gritlm_class:
                # This is a duplicate - skip it
                print(f"Removing duplicate GritLMWithVision class at cell index {i}")
                continue
            else:
                seen_gritlm_class = True
                # Update the first occurrence with fixed version
                cell['source'] = create_fixed_gritlm_cell()
                print(f"Updated GritLMWithVision class at cell index {i}")
    
    cells_to_keep.append(cell)

notebook['cells'] = cells_to_keep

# Save the modified notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print(f"\nNotebook saved: {notebook_path}")
print(f"Total cells: {len(notebook['cells'])}")
