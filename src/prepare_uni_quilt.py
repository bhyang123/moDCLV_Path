#!/usr/bin/env python3
"""Prepare immutable, indexed Quilt manifests with source-group separation."""
import argparse
import collections
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np


def write_rows(path, rows):
    offsets = []
    with path.open('wb') as f:
        for row in rows:
            offsets.append(f.tell())
            f.write((json.dumps(row, ensure_ascii=False) + '\n').encode())
    np.save(str(path) + '.offsets.npy', np.array(offsets, dtype=np.int64))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--annotations', required=True, help='Path to the Quilt pretraining JSON')
    p.add_argument('--image-root', required=True, help='Directory containing the Quilt images')
    p.add_argument('--output', required=True, help='Output manifest directory')
    p.add_argument('--val-fraction', type=float, default=0.01)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    raw = Path(a.annotations).read_bytes()
    rows = json.loads(raw)
    # The downloaded Quilt images are flat; one directory scan avoids
    # millions of metadata requests from distributed training workers.
    available = {e.name for e in os.scandir(a.image_root) if e.is_file()}
    train, val, rejected = [], [], []
    reasons = collections.Counter()
    for i, r in enumerate(rows):
        image = str(r.get('image', ''))
        conv = r.get('conversations', [])
        reason = None
        if image not in available:
            reason = 'missing_image'
        elif len(conv) != 2 or [x.get('from') for x in conv] != ['human', 'gpt']:
            reason = 'unexpected_conversation'
        elif conv[0]['value'].count('<image>') != 1 or not conv[1]['value'].strip():
            reason = 'invalid_image_marker_or_empty_answer'
        if reason:
            reasons[reason] += 1
            rejected.append({'row': i, 'image': image, 'reason': reason})
            continue
        # In Quilt, id identifies a source video or a shared prefix of a
        # source post/figure. Grouping on id is conservative for short ids.
        group = str(r['id'])
        frac = int(hashlib.sha256(('3407:' + group).encode()).hexdigest()[:16], 16) / 2**64
        record = {'row': i, 'id': group, 'image': image,
                  'question': conv[0]['value'], 'answer': conv[1]['value']}
        (val if frac < a.val_fraction else train).append(record)
    train_images = {r['image'] for r in train}
    val_images = {r['image'] for r in val}
    train_groups = {r['id'] for r in train}
    val_groups = {r['id'] for r in val}
    assert not train_images & val_images, 'Identical images cross splits'
    assert not train_groups & val_groups, 'Source groups cross splits'
    rng = random.Random(3407)
    val_eval = rng.sample(val, min(512, len(val)))
    smoke = rng.sample(train, min(64, len(train)))
    for name, selected in [('train', train), ('validation', val), ('eval512', val_eval), ('smoke', smoke)]:
        write_rows(out / f'{name}.jsonl', selected)
    write_rows(out / 'rejected.jsonl', rejected)
    summary = {
        'annotations': a.annotations, 'annotations_sha256': hashlib.sha256(raw).hexdigest(),
        'image_root': a.image_root, 'input_records': len(rows), 'train_records': len(train),
        'validation_records': len(val), 'eval_records': len(val_eval), 'rejected': dict(reasons),
        'unique_train_images': len(train_images), 'unique_val_images': len(val_images),
        'train_source_groups': len(train_groups), 'val_source_groups': len(val_groups),
        'split': 'sha256(3407:source_id); 1% source groups reserved; identical image overlap asserted zero',
        'image_decode_validation': 'decode on access; malformed image raises an explicit error',
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
