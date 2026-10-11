#!/usr/bin/env python3
"""Fixed-set comparison for the completed UNI/Qwen stage-1 run.

The script evaluates the native Qwen image features and selected adapter
checkpoints on the same 512-record manifest.  It reports an answer-token
weighted causal NLL after reducing sums across all ranks.  The ``*_shuffled``
variant keeps the checkpoint fixed but pairs UNI features with another image
in the batch, which is a useful sanity check for whether the external branch
is carrying image-specific signal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler


RELEASE_ROOT = Path(__file__).resolve().parents[1]
CODE = RELEASE_ROOT / "src"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import train_uni_qwen35_stage1 as train_mod  # noqa: E402


def rank0_print(*args: Any, **kwargs: Any) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(*args, **kwargs, flush=True)


def parse_candidate(spec: str) -> tuple[str, str | None, bool]:
    # label=path or label=path:shuffled; the path itself may contain '=' only
    # in unusual local filenames, so split at the first delimiter.
    label, value = spec.split("=", 1)
    shuffled = value.endswith(":shuffled")
    if shuffled:
        value = value[: -len(":shuffled")]
    if value.lower() in {"native", "none", ""}:
        return label, None, shuffled
    return label, value, shuffled


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(batch)
    for key in ("input_ids", "attention_mask", "mm_token_type_ids", "labels", "pixel_values", "image_grid_thw"):
        out[key] = batch[key].to(device, non_blocking=True)
    return out


def answer_loss(
    qwen,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    labels: torch.Tensor,
    image_grid_thw: torch.Tensor,
    image_embeds: torch.Tensor,
) -> tuple[float, int]:
    """Run the frozen Qwen language path and return summed answer CE/count."""
    inputs_embeds = qwen.model.get_input_embeddings()(input_ids)
    image_mask, _ = qwen.model.get_placeholder_mask(
        input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
    )
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    position_ids = qwen.model.compute_3d_position_ids(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=None,
        mm_token_type_ids=mm_token_type_ids,
    )
    outputs = qwen.model.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=None,
        inputs_embeds=inputs_embeds,
        use_cache=False,
    )
    logits = qwen.lm_head(outputs.last_hidden_state)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid = int((shift_labels != -100).sum().item())
    loss_sum = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return float(loss_sum.float().item()), valid


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--qwen-path", required=True, help="Local Qwen3.5-9B directory")
    p.add_argument("--uni-path", required=True, help="Local UNI directory")
    p.add_argument("--annotation-file", required=True, help="Prepared fixed-evaluation manifest directory")
    p.add_argument("--image-root", required=True, help="Directory containing the Quilt images")
    p.add_argument("--checkpoint-dir", required=True, help="Directory used for default candidate paths")
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--candidate",
        action="append",
        dest="candidates",
        default=[],
        help="label=checkpoint.pt, label=native, or append :shuffled",
    )
    args = p.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16
    torch.set_float32_matmul_precision("high")

    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    rank0_print("loading Qwen")
    processor = AutoProcessor.from_pretrained(args.qwen_path, trust_remote_code=True, local_files_only=True)
    qwen = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.qwen_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    qwen.eval()
    qwen.requires_grad_(False)
    qwen.config.use_cache = False
    rank0_print("loading UNI")
    uni, uni_transform = train_mod.load_uni(args.uni_path, device, dtype)

    candidates = [parse_candidate(x) for x in args.candidates]
    if not candidates:
        ckpt_dir = Path(args.checkpoint_dir)
        candidates = [
            ("native", None, False),
            ("step5000", str(ckpt_dir / "adapter_step_0005000.pt"), False),
            ("step30000", str(ckpt_dir / "adapter_step_0030000.pt"), False),
            ("final", str(ckpt_dir / "adapter_step_0044643.pt"), False),
            ("final_shuffled", str(ckpt_dir / "adapter_step_0044643.pt"), True),
        ]

    adapters: dict[str, Any] = {}
    for label, path, _ in candidates:
        if path is None:
            continue
        adapter = train_mod.UNIFusionAdapter(
            uni_dim=1024,
            qwen_dim=int(qwen.config.text_config.hidden_size),
            adapter_dim=1024,
        ).to(device=device, dtype=torch.float32)
        state, metadata = train_mod.load_adapter_state(path)
        adapter.load_state_dict(state, strict=True)
        adapter.eval()
        adapters[label] = adapter
        rank0_print(
            f"loaded {label}: {path}; source_step={metadata.get('step', '?')}"
        )

    eval_set = train_mod.QuiltPretrainDataset(
        args.annotation_file, args.image_root, "eval512", val_fraction=0.01, max_samples=0
    )
    sampler = DistributedSampler(eval_set, shuffle=False) if distributed else None
    loader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=train_mod.QuiltCollator(processor),
        drop_last=False,
    )
    rank0_print(f"fixed eval records={len(eval_set)} local_batches={len(loader)} world_size={world_size}")

    sums = {label: [0.0, 0.0, 0.0] for label, _, _ in candidates}  # loss, answer tokens, examples
    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(loader):
            batch = move_batch(raw_batch, device)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            mm_token_type_ids = batch["mm_token_type_ids"]
            labels = batch["labels"]
            pixel_values = batch["pixel_values"]
            image_grid_thw = batch["image_grid_thw"]
            with torch.autocast(device_type="cuda", dtype=dtype):
                image_outputs = qwen.model.get_image_features(pixel_values, image_grid_thw, return_dict=True)
                native_groups = list(image_outputs.pooler_output)
                cls_tokens, patch_tokens = train_mod.extract_uni_tokens(
                    uni, uni_transform, raw_batch["image_paths"], device
                )
                for label, path, shuffled in candidates:
                    if path is None:
                        fused = torch.cat(native_groups, dim=0)
                    else:
                        cls = cls_tokens
                        patches = patch_tokens
                        if shuffled and cls.shape[0] > 1:
                            cls = torch.roll(cls, shifts=1, dims=0)
                            patches = torch.roll(patches, shifts=1, dims=0)
                        fused = adapters[label](native_groups, cls, patches)
                    loss_sum, token_count = answer_loss(
                        qwen,
                        input_ids,
                        attention_mask,
                        mm_token_type_ids,
                        labels,
                        image_grid_thw,
                        fused.to(dtype=torch.bfloat16),
                    )
                    sums[label][0] += loss_sum
                    sums[label][1] += token_count
                    sums[label][2] += len(raw_batch["ids"])
            if batch_idx == 0:
                rank0_print("completed first fixed-eval batch")

    if distributed:
        for label in sums:
            reduced = torch.tensor(sums[label], device=device, dtype=torch.float64)
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            sums[label] = [float(x) for x in reduced.tolist()]

    if rank == 0:
        results = {
            "eval_manifest": str(Path(args.annotation_file) / "eval512.jsonl"),
            "records": len(eval_set),
            "world_size": world_size,
            "candidates": {},
        }
        for label, values in sums.items():
            loss_sum, tokens, examples = values
            results["candidates"][label] = {
                "answer_nll": loss_sum / tokens if tokens else None,
                "answer_ppl": (torch.exp(torch.tensor(loss_sum / tokens)).item() if tokens else None),
                "loss_sum": loss_sum,
                "answer_tokens": int(tokens),
                "examples": int(examples),
            }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2), flush=True)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
