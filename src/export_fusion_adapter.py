#!/usr/bin/env python3
"""Export a Stage-1 resumable checkpoint as adapter-only safetensors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Stage-1 adapter_step_*.pt")
    parser.add_argument("--output", required=True, help="Output .safetensors path")
    parser.add_argument("--metadata-output", default="", help="Optional JSON metadata path")
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    if "adapter" not in checkpoint:
        raise KeyError(f"Expected an 'adapter' state in {args.input}")
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in checkpoint["adapter"].items()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "safetensors",
        "source_step": checkpoint.get("step"),
        "source_epoch": checkpoint.get("epoch"),
        "tensor_count": len(state),
        "warning": "Check the UNI terms before redistributing an adapter trained from UNI outputs.",
    }
    save_file(state, str(output), metadata={k: str(v) for k, v in metadata.items() if v is not None})
    if args.metadata_output:
        metadata_path = Path(args.metadata_output)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
