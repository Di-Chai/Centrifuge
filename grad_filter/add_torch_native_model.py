#!/usr/bin/env python3
"""
Probe model autograd graph and update target_node_names.yaml.

This script:
1. Loads a model (randomly initialized, same architecture as from_pretrained)
2. Runs a forward pass to build the autograd graph
3. Collects all backward node names from the graph
4. Merges them into target_node_names.yaml (only appends new items)

After updating the yaml, run build.sh to regenerate code and recompile.

Usage:
    python add_torch_native_model.py --model_path TinyLlama/TinyLlama_v1.1
    python add_torch_native_model.py --model_path TinyLlama/TinyLlama_v1.1 --use_lora
    python add_torch_native_model.py --model_path TinyLlama/TinyLlama_v1.1 --attn_impl sdpa
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import torch
import shutil

# Ensure project root is in path for imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from arguments import parse_args
from model.utils import token_filter_loss
from transformers import AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


GRAD_FILTER_DIR = os.path.join(PROJECT_ROOT, "./")
YAML_PATH = os.path.join(GRAD_FILTER_DIR, "target_node_names.yaml")


def run_command(cmd, cwd=None):
    """Run a shell command and raise on failure."""
    print(f"[RUN] {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"Command failed with exit code {ret}: {cmd}")


def ensure_base_extension():
    """
    Ensure the base token_filter C++ extension is compiled.
    collect_node_names lives in the C++ extension, so we need at least
    a basic build before we can probe the graph.
    """
    try:
        from grad_filter import token_filter as tf
        _ = tf._C
        print("Base extension token_filter._C is available.")
        return True
    except ImportError as e:
        print(f"token_filter not importable ({e}). Building initial version...")
        run_command("python setup.py build_ext --inplace", cwd=GRAD_FILTER_DIR)
        # Verify again
        try:
            from grad_filter import token_filter as tf
            _ = tf._C
            print("Base extension built successfully.")
            return True
        except ImportError as e2:
            raise RuntimeError(f"Failed to build base extension: {e2}")


def align_seq_len(seq_len, align=8):
    """Align sequence length to satisfy Flash Attention constraints."""
    return ((seq_len + align - 1) // align) * align


def collect_node_names(args):
    """
    Load model, run forward pass, and collect all backward node names
    from the autograd graph.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load config (architecture parameters)
    print(f"Loading config from {args.model_path}...")
    config = AutoConfig.from_pretrained(args.model_path)

    # 2. Set attention implementation (must be done before model creation)
    if args.attn_impl == "eager":
        config._attn_implementation = args.attn_impl

    # 3. Create model from config (randomly initialized, same structure as from_pretrained)
    print("Creating model from config (randomly initialized weights)...")
    model = AutoModelForCausalLM.from_config(config)
    model = model.to(torch.bfloat16).to(device)
    model.train()  # Enable training-only nodes like NativeDropoutBackward0

    # 4. Apply LoRA if requested (changes the autograd graph)
    if args.use_lora:
        print("Applying LoRA...")
        lora_config = LoraConfig(
            r=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            lora_alpha=128
        )
        model = get_peft_model(model, lora_config)
        model = model.to(device)

    # 5. Prepare dummy inputs
    probe_seq_len = align_seq_len(args.context_length, align=8)
    if probe_seq_len != args.context_length:
        print(f"Aligned context_length for probing: {probe_seq_len} (original: {args.context_length})")

    input_ids = torch.randint(0, config.vocab_size, (args.batch_size, probe_seq_len)).to(device)
    attention_mask = torch.ones(args.batch_size, probe_seq_len).to(device)

    # 6. Forward pass
    print("Running forward pass...")
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        output = model(input_ids, attention_mask=attention_mask)
        logits = output.logits if hasattr(output, 'logits') else output

    # 7. Build a dummy loss that activates the full computation graph
    #    We use token_filter_loss (same as gen_node_tracing.py) to ensure
    #    the graph topology matches real training scenarios.
    print("Building dummy loss for graph collection...")
    loss, _ = token_filter_loss(
        inputs=input_ids,
        logits=logits,
        attention_mask=attention_mask,
        ref_loss=torch.randn(input_ids[:, 1:].size()).tolist(),
        dropping_strategy='fixed',
        drop_rate=0.5,
        is_left_padding=True,
        return_mask=True
    )

    # 8. Collect backward node names from the autograd graph
    print("Collecting backward node names from autograd graph...")
    from grad_filter import token_filter
    node_names = token_filter.ops.collect_node_names(loss)
    node_names = sorted(set(node_names))
    print(f"Found {len(node_names)} unique backward node types.")

    # Cleanup
    del model, output, logits, loss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return node_names


def write_yaml(node_names, backup=True):
    """Merge node names into target_node_names.yaml, only appending new items."""
    existing_names = []

    if os.path.exists(YAML_PATH):
        if backup:
            backup_path = YAML_PATH + ".bak"
            shutil.copy2(YAML_PATH, backup_path)
            print(f"Backed up existing yaml to {backup_path}")

        with open(YAML_PATH, "r") as f:
            data = yaml.safe_load(f) or {}
        existing_names = data.get("target_node_names", [])

    merged_names = sorted(set(existing_names) | set(node_names))
    added_count = len(merged_names) - len(existing_names)

    data = {"target_node_names": merged_names}
    with open(YAML_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"Written {len(merged_names)} node names to {YAML_PATH} ({added_count} newly added)")


def main():
    # ------------------------------------------------------------------
    # Parse arguments: reuse arguments.py parser and add build-specific args
    # ------------------------------------------------------------------
    base_parser = parse_args(include_deep_speed_args=False, return_parsed=False)

    # Override training-unfriendly defaults for graph probing
    base_parser.set_defaults(batch_size=1, context_length=128)

    base_parser.add_argument(
        "--no_backup",
        action="store_true",
        help="Do not backup existing target_node_names.yaml"
    )

    args = base_parser.parse_args()

    if getattr(args, 'model_path', None) is None:
        raise ValueError("--model_path is required")

    print(
        f"Configuration: model={args.model_path}, batch_size={args.batch_size}, "
        f"context_length={args.context_length}, attn_impl={args.attn_impl}, "
        f"use_lora={args.use_lora}"
    )

    # ------------------------------------------------------------------
    # Step 1: Ensure base extension is available
    # ------------------------------------------------------------------
    ensure_base_extension()

    # ------------------------------------------------------------------
    # Step 2: Probe model and collect backward node names
    # ------------------------------------------------------------------
    node_names = collect_node_names(args)

    # ------------------------------------------------------------------
    # Step 3: Write to YAML
    # ------------------------------------------------------------------
    write_yaml(node_names, backup=not args.no_backup)

    print("\ntarget_node_names.yaml updated.")


if __name__ == "__main__":
    main()
