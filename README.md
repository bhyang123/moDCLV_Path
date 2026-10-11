# moDCLV_Path: pathology branch of the moDCLV drug-discovery system

This directory is the public-facing release package for **moDCLV_Path**, the Pathology branch of the moDCLV system. The name means **multiomics Design, Chat, Loop, Value**: a long-term research and product direction in which multiomics data, foundation models, tool-using agents, experimental feedback, and measurable translational value form one loop.

This release contains the reproducible core code, sanitized training scripts, data-preparation code, evaluation summaries, and licensing notes for the current pathology image-text model. Restricted datasets, base-model files, private cluster configuration, and training caches are intentionally not included.

The current model combines frozen UNI ViT-L/16 features with Qwen3.5-9B native visual tokens. Stage 1 trains a fusion adapter. Stage 2 freezes UNI, the Qwen vision tower, and the fusion adapter, then trains LoRA modules in Qwen's language model on Quilt instruction data. The final system therefore requires the Qwen base model, an approved UNI installation, the Stage 1 fusion adapter, and the Stage 2 LoRA adapter.

## Long-term direction

moDCLV_Path is the pathology branch of a broader multiomics effort. The intended direction is to connect pathology images with genomics, transcriptomics, proteomics, clinical phenotypes, and other molecular measurements; use an agent to discover and prioritize intervention targets from large cohorts; test hypotheses computationally; coordinate wet-laboratory validation; and feed experimental results back into the next design cycle. The objective is to support innovation in drug discovery and create a traceable path from cohort evidence to validated biological and commercial value.

Those multiomics, autonomous-agent, virtual-validation, wet-lab, and industrial workflow capabilities are future roadmap items. The current release implements only the pathology image-text branch and should not be described as an autonomous drug-discovery system. See [docs/roadmap.md](docs/roadmap.md) for the staged direction.

## Repository layout

```text
src/       training, data preparation, evaluation, and adapter-export code
scripts/   path-agnostic local or cluster launch scripts
docs/      data, reproducibility, and licensing notes
weights/   release manifest and adapter configuration; no binary weights
results/   sanitized summaries from the reported runs
```

## Base models and data

Download the base models from their original repositories and comply with their terms:

- [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
- [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI)
- [QUILT-1M](https://github.com/wisdomikezogwo/quilt1m)

Do not commit the QUILT images or annotations, UNI weights, Qwen weights, access tokens, or patient-related data to this repository. UNI is gated and its terms address non-commercial research use and redistribution. QUILT-1M is distributed under restricted access terms.

## Environment

The reported run used two NVIDIA H200 GPUs, PyTorch 2.14.0+cu130, Transformers 5.8.0, PEFT 0.18.1, timm 1.0.15, flash-linear-attention 0.5.2, and einops 0.8.1. Install a CUDA-compatible PyTorch build first, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The Qwen and UNI directories must be available locally. Set the paths used by the scripts:

```bash
export QWEN_PATH=/path/to/Qwen3.5-9B
export UNI_PATH=/path/to/UNI
export STAGE1_ANNOTATION=/path/to/uni_quilt_stage1
export STAGE1_IMAGE_ROOT=/path/to/quilt_pretrain_images
export STAGE2_ANNOTATION=/path/to/quilt_instruct_stage2
export STAGE2_IMAGE_ROOT=/path/to/quilt_instruct_images
export FUSION_CHECKPOINT=/path/to/stage1_fusion.safetensors
```

## Data preparation

The preparation scripts create JSONL manifests without copying the source data:

```bash
python src/prepare_uni_quilt.py \
  --annotations /path/to/quilt_pretrain.json \
  --image-root "$STAGE1_IMAGE_ROOT" \
  --output "$STAGE1_ANNOTATION"

python src/prepare_quilt_instruct_stage2.py \
  --annotations /path/to/quilt_instruct_107k.json \
  --image-root "$STAGE2_IMAGE_ROOT" \
  --out "$STAGE2_ANNOTATION"
```

See [docs/data.md](docs/data.md) for the split rules and the restrictions on redistributing source data.

## Training

Run the smoke tests before a full job:

```bash
./scripts/smoke_stage1.sh
./scripts/smoke_stage2.sh
```

For the reported two-GPU configuration:

```bash
./scripts/train_stage1_2xh200.sh
./scripts/train_stage2_lora_2xh200.sh
```

The scripts accept `RUN_DIR`, `PYTHON`, `NPROC_PER_NODE`, and the hyperparameter environment variables documented in the script files. Use `FUSION_CHECKPOINT` to point Stage 2 to either the original resumable `.pt` checkpoint or an adapter-only `.safetensors` export.

To export a Stage 1 adapter without optimizer state:

```bash
python src/export_fusion_adapter.py \
  --input /path/to/adapter_step_0030000.pt \
  --output /path/to/stage1_fusion.safetensors \
  --metadata-output /path/to/stage1_fusion.json
```

Check the UNI terms before distributing that adapter.

## Evaluation

Use `scripts/evaluate_stage1.sh` for the fixed 512-record Stage 1 comparison. The recorded summaries are in [results/stage1_fixed_eval.json](results/stage1_fixed_eval.json) and [results/stage2_eval.json](results/stage2_eval.json). These are answer-token causal NLL values, not diagnostic accuracy.

## Release status

This is an experimental research release of moDCLV_Path. Binary weights are not included in this directory. Before publishing weights, record the exact base-model revisions, upload locations, SHA256 values, and the separate license or usage terms for code, adapters, base models, and data. See [MODEL_CARD.md](MODEL_CARD.md), [weights/README.md](weights/README.md), and [docs/licensing.md](docs/licensing.md).
