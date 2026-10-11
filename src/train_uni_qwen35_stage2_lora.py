#!/usr/bin/env python3
"""Stage 2: Quilt instruction SFT with a frozen UNI/Qwen fusion adapter.

The native Qwen vision tower, UNI encoder, and the stage-1 fusion adapter are
frozen.  LoRA modules are attached only to Qwen3.5's language model.  The
image embeddings are prepared with the same visual-token replacement used in
stage 1, then the language model is trained on answer-token causal loss.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

import train_uni_qwen35_stage1 as stage1


def rank0_print(*args: Any, **kwargs: Any) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(*args, flush=True, **kwargs)


def make_lora_config(rank: int, alpha: int | None = None) -> LoraConfig:
    # Qwen3.5 contains both self-attention layers and linear-attention layers.
    # Include both families, plus the gated MLP projections, while applying
    # PEFT only to qwen.model.language_model (never to the vision tower).
    return LoraConfig(
        r=rank,
        lora_alpha=alpha or 2 * rank,
        lora_dropout=0.05,
        bias="none",
        # The object being wrapped is Qwen3.5's text backbone rather than the
        # outer ``ForConditionalGeneration`` class.  A generic PeftModel is
        # sufficient for the direct inputs_embeds forward used below and does
        # not require the generation-only ``prepare_inputs_for_generation``.
        task_type=None,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


def load_frozen_fusion(
    checkpoint_path: str,
    qwen_dim: int,
    device: torch.device,
) -> nn.Module:
    adapter = stage1.UNIFusionAdapter(
        uni_dim=1024,
        qwen_dim=qwen_dim,
        adapter_dim=1024,
    ).to(device=device, dtype=torch.float32)
    state, metadata = stage1.load_adapter_state(checkpoint_path)
    adapter.load_state_dict(state, strict=True)
    adapter.eval()
    adapter.requires_grad_(False)
    rank0_print(
        f"loaded stage-1 fusion: {checkpoint_path}; "
        f"source_step={metadata.get('step', '?')}"
    )
    return adapter


def fused_loss(
    qwen,
    language_model,
    input_embedding,
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

    # No gradient is needed through the frozen input embedding, Qwen vision
    # tower, UNI, or stage-1 fusion adapter.  The language-model forward below
    # still builds gradients for its LoRA parameters.
    with torch.no_grad():
        inputs_embeds = input_embedding(input_ids)
        image_outputs = qwen.model.get_image_features(pixel_values, image_grid_thw, return_dict=True)
        native_groups = list(image_outputs.pooler_output)
        cls_tokens, patch_tokens = stage1.extract_uni_tokens(
            uni, uni_transform, batch["image_paths"], device
        )
        fused_image_embeds = adapter(native_groups, cls_tokens, patch_tokens)
        fused_image_embeds = fused_image_embeds.to(dtype=inputs_embeds.dtype)
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

    outputs = language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=None,
        inputs_embeds=inputs_embeds,
        use_cache=False,
    )
    shift_labels = labels[:, 1:].contiguous()
    # Project only answer positions.  Materializing vocabulary logits for the
    # image and prompt positions wastes several GiB and caused batch-size 4 to
    # OOM during LoRA backward on H200.  Boolean indexing preserves the
    # language-model gradient while keeping the logits tensor small.
    shift_hidden = outputs.last_hidden_state[:, :-1, :]
    answer_mask = shift_labels != -100
    answer_hidden = shift_hidden[answer_mask]
    answer_labels = shift_labels[answer_mask]
    valid = int(answer_labels.numel())
    if valid == 0:
        raise RuntimeError("batch has no answer tokens after prompt masking")
    loss_sum = F.cross_entropy(
        qwen.lm_head(answer_hidden),
        answer_labels,
        reduction="sum",
    )
    loss = loss_sum / valid
    with torch.no_grad():
        loss_value = float(loss.detach())
    return loss, {
        "loss": loss_value,
        "loss_sum": loss_value * float(valid),
        "answer_tokens": float(valid),
    }


def save_lora_checkpoint(
    language_model,
    optimizer,
    scheduler,
    step: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    module = language_model.module if isinstance(language_model, DistributedDataParallel) else language_model
    path = output_dir / "checkpoints" / f"step_{step:07d}"
    path.mkdir(parents=True, exist_ok=True)
    module.save_pretrained(path / "lora", safe_serialization=True)
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
        },
        path / "optimizer.pt",
    )
    (path / "COMPLETE").write_text(str(step), encoding="utf-8")
    (output_dir / "latest.json").write_text(
        json.dumps({"step": step, "path": str(path)}, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--qwen-path", required=True, help="Local Qwen3.5-9B directory")
    p.add_argument("--uni-path", required=True, help="Local UNI directory")
    p.add_argument("--fusion-checkpoint", required=True, help="Stage-1 .pt or .safetensors adapter")
    p.add_argument("--annotation-file", required=True, help="Prepared Quilt instruction manifest directory")
    p.add_argument("--image-root", required=True, help="Directory containing the instruction images")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=512)
    p.add_argument("--eval-steps", type=int, default=1000)
    p.add_argument("--eval-batches", type=int, default=0)
    p.add_argument("--save-steps", type=int, default=1000)
    p.add_argument("--log-steps", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=3407)
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
    dtype = torch.bfloat16
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

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
    input_embedding = qwen.model.get_input_embeddings()
    input_embedding.requires_grad_(False)

    rank0_print(f"loading UNI from {args.uni_path}")
    uni, uni_transform = stage1.load_uni(args.uni_path, device, dtype)
    adapter = load_frozen_fusion(
        args.fusion_checkpoint,
        int(qwen.config.text_config.hidden_size),
        device,
    )

    lora_config = make_lora_config(args.lora_r, args.lora_alpha)
    language_model = get_peft_model(
        qwen.model.language_model,
        lora_config,
        autocast_adapter_dtype=True,
    )
    language_model = language_model.to(device)
    language_model.print_trainable_parameters()
    if rank == 0:
        trainable_names = [name for name, p in language_model.named_parameters() if p.requires_grad]
        (output_dir / "lora_trainable_names.json").write_text(
            json.dumps(trainable_names[:50], indent=2), encoding="utf-8"
        )
    if distributed:
        language_model = DistributedDataParallel(
            language_model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    train_set = stage1.QuiltPretrainDataset(
        args.annotation_file,
        args.image_root,
        "train",
        val_fraction=0.01,
        max_samples=args.max_train_samples,
    )
    val_set = stage1.QuiltPretrainDataset(
        args.annotation_file,
        args.image_root,
        "eval512",
        val_fraction=0.01,
        max_samples=args.max_val_samples,
    )
    train_sampler = DistributedSampler(train_set, shuffle=True, seed=args.seed) if distributed else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if distributed else None
    collator = stage1.QuiltCollator(processor)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collator,
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
        collate_fn=collator,
        drop_last=False,
    )
    updates_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum_steps))
    total_steps = args.max_steps if args.max_steps > 0 else updates_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(
        [p for p in language_model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        remaining = max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / remaining))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    start_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        checkpoint = torch.load(resume_path / "optimizer.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint.get("step", 0))
        module = language_model.module if isinstance(language_model, DistributedDataParallel) else language_model
        module.load_adapter(str(resume_path / "lora"), adapter_name="default", is_trainable=True)
        rank0_print(f"resumed LoRA from {resume_path} at step={start_step}")

    if rank == 0:
        rank0_print(
            f"dataset train={len(train_set)} eval={len(val_set)}; "
            f"batches/epoch={len(train_loader)} total_steps={total_steps} world_size={world_size}"
        )

    def evaluate(step: int) -> None:
        language_model.eval()
        local_sum = 0.0
        local_tokens = 0.0
        local_examples = 0.0
        with torch.no_grad():
            for val_idx, val_batch in enumerate(val_loader):
                if args.eval_batches > 0 and val_idx >= args.eval_batches:
                    break
                with torch.autocast(device_type="cuda", dtype=dtype):
                    _, metrics = fused_loss(
                        qwen, language_model, input_embedding, adapter,
                        uni, uni_transform, val_batch, device, dtype,
                    )
                local_sum += metrics["loss_sum"]
                local_tokens += metrics["answer_tokens"]
                local_examples += len(val_batch["ids"])
        reduced = torch.tensor([local_sum, local_tokens, local_examples], device=device, dtype=torch.float64)
        if distributed:
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        if rank == 0 and reduced[1] > 0:
            val_loss = float(reduced[0] / reduced[1])
            payload = {
                "step": step,
                "val_loss": val_loss,
                "val_tokens": int(reduced[1]),
                "val_examples": int(reduced[2]),
            }
            print(json.dumps(payload), flush=True)
            with (output_dir / "train_log.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
        language_model.train()

    log_file = (output_dir / "train_log.jsonl").open("a", encoding="utf-8") if rank == 0 else None
    start_time = time.time()
    log_loss_sum = 0.0
    log_tokens = 0.0
    step = start_step
    stop = False
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        language_model.train()
        for batch_idx, batch in enumerate(train_loader):
            if step >= total_steps:
                stop = True
                break
            with torch.autocast(device_type="cuda", dtype=dtype):
                loss, metrics = fused_loss(
                    qwen, language_model, input_embedding, adapter,
                    uni, uni_transform, batch, device, dtype,
                )
            log_loss_sum += metrics["loss_sum"]
            log_tokens += metrics["answer_tokens"]
            sync_context = (
                contextlib.nullcontext()
                if not isinstance(language_model, DistributedDataParallel)
                or ((batch_idx + 1) % args.grad_accum_steps == 0)
                or (batch_idx + 1 == len(train_loader))
                else language_model.no_sync()
            )
            with sync_context:
                (loss / args.grad_accum_steps).backward()
            should_step = ((batch_idx + 1) % args.grad_accum_steps == 0) or (batch_idx + 1 == len(train_loader))
            if not should_step:
                continue
            torch.nn.utils.clip_grad_norm_(
                [p for p in language_model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_steps == 0 or step == 1:
                reduced = torch.tensor([log_loss_sum, log_tokens], device=device, dtype=torch.float64)
                if distributed:
                    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                global_loss, global_tokens = (float(x) for x in reduced.tolist())
                payload = {
                    "step": step,
                    "epoch": epoch,
                    "lr": scheduler.get_last_lr()[0],
                    "loss": global_loss / max(global_tokens, 1.0),
                    "loss_sum": global_loss,
                    "answer_tokens": global_tokens,
                    "elapsed_sec": time.time() - start_time,
                }
                if rank == 0:
                    print(json.dumps(payload), flush=True)
                    log_file.write(json.dumps(payload) + "\n")
                    log_file.flush()
                log_loss_sum = 0.0
                log_tokens = 0.0
            if args.save_steps > 0 and step % args.save_steps == 0:
                save_lora_checkpoint(language_model, optimizer, scheduler, step, output_dir, args)
            if args.eval_steps > 0 and step % args.eval_steps == 0:
                evaluate(step)
            if step >= total_steps:
                stop = True
                break
        if stop:
            break

    save_lora_checkpoint(language_model, optimizer, scheduler, step, output_dir, args)
    if args.eval_steps <= 0 or step % args.eval_steps != 0:
        evaluate(step)
    if rank == 0:
        (output_dir / "fusion_reference.json").write_text(
            json.dumps(
                {
                    "checkpoint": args.fusion_checkpoint,
                    "trainable": "Qwen3.5 language-model LoRA only",
                    "frozen": ["UNI", "Qwen vision tower", "stage-1 fusion adapter"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (output_dir / "TRAINING_COMPLETE").write_text(str(step), encoding="utf-8")
    if log_file is not None:
        log_file.close()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    rank0_print(f"finished stage 2 at step={step}; output={output_dir}")


if __name__ == "__main__":
    main()
