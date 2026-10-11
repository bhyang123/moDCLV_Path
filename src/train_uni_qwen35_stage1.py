#!/usr/bin/env python3
"""Stage-1 UNI -> Qwen3.5 visual-token fusion training.

The Qwen3.5 language model and vision tower are kept frozen.  UNI is also
kept frozen, and its CLS plus spatial patch tokens are passed through a small
cross-attention/MLP adapter.  The adapter produces a residual for each native
Qwen visual token, so Qwen's own image-token count and multimodal RoPE remain
unchanged.  The objective is causal next-token loss on the assistant answer.

This script deliberately uses the Transformers model API instead of a
LLaMA-Factory YAML: injecting a second visual encoder requires the fused image
embeddings to be inserted before Qwen3.5's language model forward pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms


def rank0_print(*args: Any, **kwargs: Any) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(*args, **kwargs, flush=True)


def set_seed(seed: int, rank: int = 0) -> None:
    seed = int(seed) + int(rank)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def image_content(value: str, path: str) -> list[dict[str, str]]:
    """Turn the Quilt ``<image>`` marker into Qwen processor content blocks."""
    blocks: list[dict[str, str]] = []
    pieces = value.split("<image>")
    for i, piece in enumerate(pieces):
        if piece:
            blocks.append({"type": "text", "text": piece})
        if i != len(pieces) - 1:
            blocks.append({"type": "image", "image": path})
    if not blocks:
        blocks.append({"type": "image", "image": path})
    return blocks


def find_assistant_prefix(input_ids: torch.Tensor, tokenizer) -> int:
    """Return the first answer-token position for loss masking.

    Qwen's chat template emits ``<|im_start|>assistant`` before the answer.
    Matching the actual token sequence is robust to the template's optional
    ``<think>`` wrapper and to an expanded image placeholder.
    """
    ids = input_ids.tolist()
    start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    role = tokenizer.encode("assistant", add_special_tokens=False)
    nl = tokenizer.encode("\n", add_special_tokens=False)
    patterns = [[start, *role, *nl], [start, *role]]
    for pattern in patterns:
        n = len(pattern)
        for i in range(len(ids) - n + 1):
            if ids[i : i + n] == pattern:
                return i + n
    # A template change should not silently train on the prompt.  The final
    # assistant marker is still a safe fallback for a one-turn record.
    candidates = [i for i, token in enumerate(ids) if token == start]
    if len(candidates) >= 2:
        return candidates[-1] + 1
    raise ValueError("Could not locate the assistant marker in Qwen3.5 input_ids")


class QuiltPretrainDataset(Dataset):
    def __init__(
        self,
        annotation_file: str,
        image_root: str,
        split: str,
        val_fraction: float,
        max_samples: int = 0,
    ) -> None:
        annotation_path = Path(annotation_file)
        prepared_manifest = annotation_path.is_dir()
        if annotation_path.is_dir():
            # ``prepare_uni_quilt.py`` emits JSONL manifests with a source
            # grouped split.  Convert them back to the two-turn structure used
            # by the generic JSON reader below.
            manifest_name = "validation" if split == "val" else split
            manifest_path = annotation_path / f"{manifest_name}.jsonl"
            rows = []
            with manifest_path.open("r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    rows.append(
                        {
                            "id": record["id"],
                            "image": record["image"],
                            "conversations": [
                                {"from": "human", "value": record["question"]},
                                {"from": "gpt", "value": record["answer"]},
                            ],
                        }
                    )
            # A prepared directory has already been split by source group.
        else:
            with annotation_path.open("r", encoding="utf-8") as f:
                rows = json.load(f)
        if not isinstance(rows, list):
            raise TypeError(f"Expected a JSON list in {annotation_file}")
        self.rows: list[dict[str, Any]] = []
        self.image_root = Path(image_root)
        missing = 0
        for row in rows:
            image = str(row["image"])
            path = self.image_root / image
            if not path.is_file():
                missing += 1
                continue
            # Split by image name so repeated captions for one image cannot
            # land in both train and validation sets.
            if not prepared_manifest:
                bucket = int(hashlib.sha1(image.encode("utf-8")).hexdigest()[:8], 16) / 2**32
                is_val = bucket < val_fraction
                if (split == "val") != is_val:
                    continue
            if len(row.get("conversations", [])) < 2:
                continue
            self.rows.append(row)
        if max_samples > 0:
            self.rows = self.rows[:max_samples]
        self.missing = missing
        self.split = split

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        human = str(row["conversations"][0]["value"])
        answer = str(row["conversations"][1]["value"])
        path = str(self.image_root / str(row["image"]))
        messages = [
            {"role": "user", "content": image_content(human, path)},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        return {"id": str(row.get("id", idx)), "image_path": path, "messages": messages}


class QuiltCollator:
    def __init__(self, processor) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        encoded: list[dict[str, torch.Tensor]] = []
        ids: list[str] = []
        paths: list[str] = []
        for example in examples:
            item = self.processor.apply_chat_template(
                example["messages"],
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
            item = {
                k: v.squeeze(0)
                if torch.is_tensor(v) and v.ndim > 0 and k not in {"pixel_values", "image_grid_thw"}
                else v
                for k, v in item.items()
            }
            # ``pixel_values`` is already [num_patches, patch_dim], while all
            # sequence fields are [1, length] for a single record.
            input_ids = item["input_ids"]
            if input_ids.ndim != 1:
                input_ids = input_ids.squeeze(0)
            prefix_len = find_assistant_prefix(input_ids, self.tokenizer)
            labels = input_ids.clone()
            labels[:prefix_len] = -100
            item["input_ids"] = input_ids
            item["labels"] = labels
            encoded.append(item)
            ids.append(example["id"])
            paths.append(example["image_path"])

        max_len = max(x["input_ids"].numel() for x in encoded)

        def pad_1d(key: str, value: int) -> torch.Tensor:
            values = [x[key] for x in encoded]
            return torch.stack(
                [F.pad(v, (0, max_len - v.numel()), value=value) for v in values], dim=0
            )

        pixel_values = torch.cat([x["pixel_values"] for x in encoded], dim=0)
        image_grid_thw = torch.cat([x["image_grid_thw"] for x in encoded], dim=0)
        return {
            "input_ids": pad_1d("input_ids", self.tokenizer.pad_token_id),
            "attention_mask": pad_1d("attention_mask", 0),
            "mm_token_type_ids": pad_1d("mm_token_type_ids", 0),
            "labels": pad_1d("labels", -100),
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "image_paths": paths,
            "ids": ids,
        }


class UNIFusionAdapter(nn.Module):
    """Map UNI CLS/patch tokens into residuals for Qwen visual tokens."""

    def __init__(
        self,
        uni_dim: int = 1024,
        qwen_dim: int = 4096,
        adapter_dim: int = 1024,
        num_heads: int = 16,
        max_uni_tokens: int = 197,
    ) -> None:
        super().__init__()
        if adapter_dim % num_heads:
            raise ValueError("adapter_dim must be divisible by num_heads")
        self.uni_dim = uni_dim
        self.qwen_dim = qwen_dim
        self.adapter_dim = adapter_dim
        self.max_uni_tokens = max_uni_tokens
        self.cls_mlp = nn.Sequential(
            nn.LayerNorm(uni_dim),
            nn.Linear(uni_dim, 2 * adapter_dim),
            nn.GELU(),
            nn.Linear(2 * adapter_dim, adapter_dim),
        )
        self.patch_mlp = nn.Sequential(
            nn.LayerNorm(uni_dim),
            nn.Linear(uni_dim, 2 * adapter_dim),
            nn.GELU(),
            nn.Linear(2 * adapter_dim, adapter_dim),
        )
        self.type_embedding = nn.Parameter(torch.zeros(2, adapter_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, max_uni_tokens, adapter_dim))
        self.query_projection = nn.Linear(qwen_dim, adapter_dim, bias=False)
        self.cross_attention = nn.MultiheadAttention(
            adapter_dim, num_heads, batch_first=True, dropout=0.0
        )
        self.output_mlp = nn.Sequential(
            nn.LayerNorm(adapter_dim),
            nn.Linear(adapter_dim, 2 * adapter_dim),
            nn.GELU(),
            nn.Linear(2 * adapter_dim, qwen_dim),
        )
        # Start close to the native Qwen vision model and let the data learn
        # how strongly the pathology branch should alter each visual token.
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.type_embedding, std=0.02)

    def forward(
        self,
        native_image_embeds: list[torch.Tensor],
        cls_tokens: torch.Tensor,
        patch_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if cls_tokens.ndim != 2 or patch_tokens.ndim != 3:
            raise ValueError("UNI features must have shapes [B,D] and [B,N,D]")
        if patch_tokens.shape[1] + 1 > self.max_uni_tokens:
            raise ValueError("UNI token sequence is longer than the adapter position table")
        cls = self.cls_mlp(cls_tokens).unsqueeze(1)
        patches = self.patch_mlp(patch_tokens)
        external = torch.cat([cls, patches], dim=1)
        type_ids = torch.cat(
            [
                torch.zeros((1,), dtype=torch.long, device=external.device),
                torch.ones((patches.shape[1],), dtype=torch.long, device=external.device),
            ]
        )
        external = external + self.type_embedding[type_ids].unsqueeze(0)
        external = external + self.position_embedding[:, : external.shape[1]]

        outputs: list[torch.Tensor] = []
        if len(native_image_embeds) != external.shape[0]:
            raise ValueError(
                f"Native image groups ({len(native_image_embeds)}) and UNI batch ({external.shape[0]}) differ"
            )
        gate = torch.sigmoid(self.gate_logit)
        for i, native in enumerate(native_image_embeds):
            query = self.query_projection(native).unsqueeze(0)
            attended, _ = self.cross_attention(query, external[i : i + 1], external[i : i + 1], need_weights=False)
            delta = self.output_mlp(attended.squeeze(0))
            outputs.append(native + gate * delta)
        return torch.cat(outputs, dim=0)


def load_uni(model_path: str, device: torch.device, dtype: torch.dtype):
    import timm

    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )
    state = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device=device, dtype=dtype).eval()
    model.requires_grad_(False)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return model, transform


def extract_uni_tokens(uni, transform, paths: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images: list[torch.Tensor] = []
    for path in paths:
        with Image.open(path) as image:
            images.append(transform(image.convert("RGB")))
    pixels = torch.stack(images, dim=0).to(device=device)
    # ``no_grad`` (rather than ``inference_mode``) is intentional: the UNI
    # result is frozen, but it is consumed by the trainable adapter and must
    # remain a regular tensor that autograd can save for the adapter backward.
    with torch.no_grad():
        features = uni.forward_features(pixels)
        if isinstance(features, dict):
            cls = features.get("x_norm_clstoken")
            patches = features.get("x_norm_patchtokens")
            if cls is None or patches is None:
                sequence = features.get("x_prenorm") or features.get("last_hidden_state")
                if sequence is None:
                    raise RuntimeError(f"Unsupported UNI forward_features keys: {list(features)}")
                cls, patches = sequence[:, 0], sequence[:, 1:]
        else:
            if features.ndim != 3 or features.shape[1] < 2:
                raise RuntimeError(f"Expected UNI token sequence, got {tuple(features.shape)}")
            cls, patches = features[:, 0], features[:, 1:]
    return cls, patches


def load_adapter_state(path: str | os.PathLike[str]) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Load an adapter state from a resumable checkpoint or safetensors export."""
    checkpoint_path = Path(path)
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(checkpoint_path), device="cpu"), {}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "adapter" not in checkpoint:
        raise KeyError(f"Expected an 'adapter' state in {checkpoint_path}")
    metadata = {
        key: str(checkpoint[key])
        for key in ("step", "epoch")
        if key in checkpoint
    }
    return checkpoint["adapter"], metadata


def fused_loss(
    qwen,
    adapter,
    uni,
    uni_transform,
    batch: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, float]]:
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    mm_token_type_ids = batch["mm_token_type_ids"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    pixel_values = batch["pixel_values"].to(device, non_blocking=True)
    image_grid_thw = batch["image_grid_thw"].to(device, non_blocking=True)

    # Qwen's native vision tower is frozen, but its output stays in the graph
    # so the loss can backpropagate through the language model into the adapter.
    inputs_embeds = qwen.model.get_input_embeddings()(input_ids)
    image_outputs = qwen.model.get_image_features(pixel_values, image_grid_thw, return_dict=True)
    native_groups = list(image_outputs.pooler_output)
    cls_tokens, patch_tokens = extract_uni_tokens(uni, uni_transform, batch["image_paths"], device)
    fused_image_embeds = adapter(native_groups, cls_tokens, patch_tokens)
    fused_image_embeds = fused_image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    image_mask, _ = qwen.model.get_placeholder_mask(
        input_ids, inputs_embeds=inputs_embeds, image_features=fused_image_embeds
    )
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, fused_image_embeds)
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
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    with torch.no_grad():
        valid = (shift_labels != -100).sum().item()
        ppl = math.exp(min(float(loss.detach()), 20.0))
    loss_value = float(loss.detach())
    return loss, {
        "loss": loss_value,
        "loss_sum": loss_value * float(valid),
        "ppl": ppl,
        "answer_tokens": float(valid),
    }


def save_checkpoint(adapter, optimizer, scheduler, step: int, epoch: int, out_dir: Path, args) -> None:
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    module = adapter.module if isinstance(adapter, DistributedDataParallel) else adapter
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "adapter": module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "args": vars(args),
        },
        out_dir / f"adapter_step_{step:07d}.pt",
    )
    (out_dir / "latest.json").write_text(json.dumps({"step": step, "epoch": epoch}, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--qwen-path", required=True, help="Local Qwen3.5-9B directory")
    p.add_argument("--uni-path", required=True, help="Local UNI directory")
    p.add_argument("--annotation-file", required=True, help="Prepared manifest directory or Quilt JSON")
    p.add_argument("--image-root", required=True, help="Directory containing the Quilt images")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0, help="0 means all steps in the requested epochs")
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=512)
    p.add_argument("--val-fraction", type=float, default=0.01)
    p.add_argument("--eval-steps", type=int, default=1000)
    p.add_argument("--eval-batches", type=int, default=0, help="0 means all selected validation batches")
    p.add_argument("--save-steps", type=int, default=1000)
    p.add_argument("--log-steps", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--adapter-dim", type=int, default=1024)
    p.add_argument("--resume", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.set_float32_matmul_precision("high")
    set_seed(args.seed, rank)
    dtype = torch.bfloat16
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))

    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    rank0_print(f"loading Qwen3.5 from {args.qwen_path}")
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
    rank0_print(f"Qwen loaded: {sum(p.numel() for p in qwen.parameters()) / 1e9:.2f}B parameters")

    rank0_print(f"loading UNI from {args.uni_path}")
    uni, uni_transform = load_uni(args.uni_path, device, dtype)
    # Keep trainable weights in FP32.  Frozen Qwen/UNI weights and activations
    # still use BF16 through autocast, but BF16 AdamW parameters quantize small
    # updates away and leave gates/LayerNorm scales stuck at their initial values.
    adapter = UNIFusionAdapter(
        uni_dim=1024,
        qwen_dim=int(qwen.config.text_config.hidden_size),
        adapter_dim=args.adapter_dim,
    ).to(device=device, dtype=torch.float32)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        adapter.load_state_dict(checkpoint["adapter"], strict=True)
        rank0_print(f"resumed adapter from {args.resume} at step {checkpoint.get('step', '?')}")
    if distributed:
        adapter = DistributedDataParallel(adapter, device_ids=[local_rank], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    train_set = QuiltPretrainDataset(
        args.annotation_file, args.image_root, "train", args.val_fraction, args.max_train_samples
    )
    val_set = QuiltPretrainDataset(
        args.annotation_file, args.image_root, "val", args.val_fraction, args.max_val_samples
    )
    if rank == 0:
        rank0_print(f"dataset train={len(train_set)} val={len(val_set)} missing={train_set.missing}")
    train_sampler = DistributedSampler(train_set, shuffle=True, seed=args.seed) if distributed else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if distributed else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=QuiltCollator(processor),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=max(1, min(args.num_workers, 2)),
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=QuiltCollator(processor),
        drop_last=False,
    )
    updates_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum_steps))
    total_steps = args.max_steps if args.max_steps > 0 else updates_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=args.lr * 0.1)
    rank0_print(f"train batches/epoch={len(train_loader)} optimizer_steps={total_steps}")

    start_step = 0
    last_step = 0
    epoch = 0
    optimizer.zero_grad(set_to_none=True)
    log_file = (output_dir / "train_log.jsonl").open("a", encoding="utf-8") if rank == 0 else None
    start_time = time.time()
    stop = False
    log_loss_sum = 0.0
    log_answer_tokens = 0.0
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        adapter.train()
        for batch_idx, batch in enumerate(train_loader):
            step = start_step + epoch * updates_per_epoch + batch_idx // args.grad_accum_steps
            with torch.autocast(device_type="cuda", dtype=dtype):
                loss, metrics = fused_loss(qwen, adapter, uni, uni_transform, batch, device, dtype)
            log_loss_sum += metrics["loss_sum"]
            log_answer_tokens += metrics["answer_tokens"]
            (loss / args.grad_accum_steps).backward()
            should_step = ((batch_idx + 1) % args.grad_accum_steps == 0) or (batch_idx + 1 == len(train_loader))
            if should_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                last_step = step
                if step % args.log_steps == 0 or step == 1:
                    reduced = torch.tensor(
                        [log_loss_sum, log_answer_tokens], device=device, dtype=torch.float64
                    )
                    if distributed:
                        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                    global_loss_sum, global_answer_tokens = (float(x) for x in reduced.tolist())
                    payload = {
                        "step": step,
                        "epoch": epoch,
                        "lr": scheduler.get_last_lr()[0],
                        "grad_norm": float(grad_norm),
                        "elapsed_sec": time.time() - start_time,
                        **metrics,
                    }
                    payload.update(
                        {
                            "loss": global_loss_sum / max(global_answer_tokens, 1.0),
                            "loss_sum": global_loss_sum,
                            "answer_tokens": global_answer_tokens,
                            "ppl": math.exp(min(global_loss_sum / max(global_answer_tokens, 1.0), 20.0)),
                        }
                    )
                    if rank == 0:
                        print(json.dumps(payload), flush=True)
                        log_file.write(json.dumps(payload) + "\n")
                        log_file.flush()
                    log_loss_sum = 0.0
                    log_answer_tokens = 0.0
                if step > 0 and step % args.save_steps == 0:
                    save_checkpoint(adapter, optimizer, scheduler, step, epoch, output_dir / "checkpoints", args)
                if args.eval_steps > 0 and step > 0 and step % args.eval_steps == 0:
                    adapter.eval()
                    val_loss_sum = 0.0
                    val_tokens = 0.0
                    val_examples = 0.0
                    with torch.no_grad():
                        for val_idx, val_batch in enumerate(val_loader):
                            if args.eval_batches > 0 and val_idx >= args.eval_batches:
                                break
                            with torch.autocast(device_type="cuda", dtype=dtype):
                                _, val_metrics = fused_loss(qwen, adapter, uni, uni_transform, val_batch, device, dtype)
                            val_loss_sum += val_metrics["loss_sum"]
                            val_tokens += val_metrics["answer_tokens"]
                            val_examples += len(val_batch["ids"])
                    if distributed:
                        reduced = torch.tensor(
                            [val_loss_sum, val_tokens, val_examples], device=device, dtype=torch.float64
                        )
                        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                        val_loss_sum, val_tokens, val_examples = (float(x) for x in reduced.tolist())
                        dist.barrier()
                    if rank == 0 and val_tokens > 0:
                        val_payload = {
                            "step": step,
                            "val_loss": val_loss_sum / val_tokens,
                            "val_tokens": int(val_tokens),
                            "val_examples": int(val_examples),
                            "val_batches_per_rank": val_idx + 1,
                        }
                        print(json.dumps(val_payload), flush=True)
                        log_file.write(json.dumps(val_payload) + "\n")
                        log_file.flush()
                    adapter.train()
                if step >= total_steps:
                    stop = True
                    break
        if stop:
            break
    final_step = last_step
    save_checkpoint(adapter, optimizer, scheduler, final_step, epoch, output_dir / "checkpoints", args)
    if log_file is not None:
        log_file.close()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    rank0_print(f"finished stage 1 at step={final_step}; output={output_dir}")


if __name__ == "__main__":
    main()
