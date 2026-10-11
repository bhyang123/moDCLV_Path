# Data and split protocol

The public package contains no QUILT images or annotation files. Obtain QUILT-1M from the [official repository](https://github.com/wisdomikezogwo/quilt1m), complete any required access agreement, and keep the data outside this Git repository.

## Stage 1

`src/prepare_uni_quilt.py` validates image availability, requires one human turn and one GPT turn, requires exactly one `<image>` marker, and assigns complete source IDs to train or validation using the SHA256 bucket rule with seed 3407. It asserts zero identical-image and zero source-group overlap.

## Stage 2

`src/prepare_quilt_instruct_stage2.py` validates image availability and keeps the first human/assistant pair. It assigns records using a SHA256 bucket rule with seed 3407 and writes `train.jsonl`, `validation.jsonl`, `eval512.jsonl`, and `summary.json`. The fixed evaluation file is sampled from the held-out records with the same seed.

The current release documents the source counts and filtering decisions but does not commit the generated manifests. If a license permits sharing an annotation-only manifest, publish its SHA256 and the exact generator command rather than copying source images into this repository.
