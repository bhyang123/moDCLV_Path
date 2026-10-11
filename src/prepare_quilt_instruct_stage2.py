#!/usr/bin/env python3
"""Create immutable, image-grouped manifests for Quilt instruction SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def bucket(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) / 16**16


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", required=True, help="Path to the Quilt instruction JSON")
    p.add_argument("--image-root", required=True, help="Directory containing the instruction images")
    p.add_argument("--out", required=True, help="Output manifest directory")
    p.add_argument("--val-fraction", type=float, default=0.01)
    p.add_argument("--eval-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=3407)
    args = p.parse_args()

    annotations = Path(args.annotations)
    image_root = Path(args.image_root)
    out = Path(args.out)
    rows = json.loads(annotations.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"expected a JSON list, got {type(rows).__name__}")

    train: list[dict] = []
    validation: list[dict] = []
    missing = 0
    rejected = 0
    group_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        conversations = row.get("conversations", [])
        image = str(row.get("image", ""))
        key = str(row.get("id", image))
        if len(conversations) < 2 or not image:
            rejected += 1
            continue
        path = image_root / image
        if not path.is_file():
            missing += 1
            continue
        record = {
            "id": key,
            "image": image,
            "question": str(conversations[0].get("value", "")),
            "answer": str(conversations[1].get("value", "")),
        }
        if bucket(key, args.seed) < args.val_fraction:
            validation.append(record)
            group_rows[key].append(record)
        else:
            train.append(record)

    if missing or rejected:
        raise RuntimeError(f"missing_images={missing}, rejected_rows={rejected}")
    if not train or not validation:
        raise RuntimeError(f"empty split: train={len(train)}, validation={len(validation)}")

    # The diagnostic manifest contains one question per held-out image/source
    # group, so a duplicate image cannot dominate the fixed comparison.
    rng = random.Random(args.seed)
    eval_groups = list(group_rows.values())
    rng.shuffle(eval_groups)
    eval_rows = [group[0] for group in eval_groups[: args.eval_size]]

    out.mkdir(parents=True, exist_ok=True)
    for name, split_rows in (("train", train), ("validation", validation), ("eval512", eval_rows)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for record in split_rows:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "annotations": str(annotations),
        "annotations_sha256": hashlib.sha256(annotations.read_bytes()).hexdigest(),
        "image_root": str(image_root),
        "input_records": len(rows),
        "train_records": len(train),
        "validation_records": len(validation),
        "eval_records": len(eval_rows),
        "train_groups": len({r["id"] for r in train}),
        "validation_groups": len(group_rows),
        "split": "sha256(3407:id); 1% id/image groups reserved for validation",
        "missing_images": missing,
        "rejected_rows": rejected,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
