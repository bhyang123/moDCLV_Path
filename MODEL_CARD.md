# Model card: moDCLV_Path

## Summary

moDCLV_Path is the pathology branch of the moDCLV system, whose name means **multiomics Design, Chat, Loop, Value**. This branch is an experimental pathology VLM that augments the native Qwen3.5-9B visual representation with frozen UNI ViT-L/16 features. It was trained for image-grounded answer generation on QUILT-derived data. It is a research artifact and has not been clinically validated.

The broader moDCLV direction is to connect pathology with other omics, enable agent-assisted intervention-target discovery from large cohorts, perform computational and wet-laboratory validation, and close the loop with evidence and value measurement. These capabilities are not implemented by this checkpoint.

## Architecture

- **UNI branch:** UNI ViT-L/16 at 224 x 224 input resolution; one 1024-dimensional CLS token and 196 1024-dimensional spatial tokens.
- **Qwen branch:** Qwen3.5-9B native image encoder and language model; the native visual output dimension is 4096.
- **Stage 1 fusion:** separate LayerNorm/MLP projections for the CLS and spatial tokens, type and position embeddings, 16-head cross-attention with Qwen visual tokens as queries, and an output MLP that produces a gated residual in the Qwen visual space. UNI and Qwen are frozen.
- **Stage 2 language adaptation:** LoRA is attached only to the Qwen language model. The Qwen vision tower, UNI, and Stage 1 fusion adapter remain frozen.

## Intended use

Research on pathology image-text modeling, representation fusion, and reproducible VLM experiments. This checkpoint is one model component for the longer-term moDCLV multiomics and drug-discovery roadmap; it is not an autonomous agent and does not perform target discovery or laboratory validation by itself.

## Out-of-scope use

This release must not be presented as a diagnostic device or used as the sole basis for clinical decisions. It has not been validated on independent clinical cohorts, prospective data, or clinician-adjudicated diagnostic endpoints.

## Training data

Stage 1 used prepared QUILT pretraining manifests with 714,277 training records and 9,051 validation records. Stage 2 used the QUILT instruction set after local image filtering and manifest preparation. The Stage 2 preparation keeps only `conversations[0]` and `conversations[1]` from each record, so it is a first-question/first-answer SFT setup rather than full multi-turn instruction tuning. The recorded Stage 2 split contains 106,001 training records and 1,130 validation records, with a fixed 512-example evaluation manifest.

The source images and annotation files are not redistributed here. Users must obtain them through the official QUILT-1M access process and follow its terms.

## Training procedure

The reported jobs used distributed training. Stage 1 selected the 30,000-step fusion checkpoint for the Stage 2 run. Stage 2 ran one epoch with global batch size 16, learning rate 2e-5, LoRA rank 16, alpha 32, dropout 0.05, and seed 3407. The final Stage 2 checkpoint was step 6626.

The exact commands and path variables are in `scripts/`, and the implementation is in `src/`. Record the exact Qwen and UNI revisions when publishing a weight artifact; the local training run used downloaded copies and did not preserve a public revision field.

## Evaluation

### Stage 1 fixed evaluation

Answer-token weighted causal NLL on the same 512-record manifest:

| Candidate | NLL | PPL |
|---|---:|---:|
| Native Qwen visual features | 2.462954 | 11.7394 |
| Stage 1 step 5,000 | 1.786240 | 5.9670 |
| Stage 1 step 30,000 | 1.770435 | 5.8734 |
| Stage 1 final checkpoint | 1.770844 | 5.8758 |
| Stage 1 final with shuffled UNI features | 1.780949 | 5.9355 |

### Stage 2 validation

The final Stage 2 checkpoint at step 6626 reached answer-token weighted validation NLL **0.834089** over 512 examples and 93,022 answer tokens. This number is a language-modeling loss and is not a clinical accuracy measure.

## Limitations

- The Stage 2 validation and fixed evaluation are derived from the same QUILT source family as training. Source-prefix audits found substantial source overlap between training and the 512-image evaluation set, so these results should not be described as independent generalization.
- The current evaluation reports loss and perplexity only; it does not establish factuality, localization quality, diagnostic accuracy, calibration, or clinical utility.
- The selected Stage 1 checkpoint came from an earlier completed run whose diagnostic notes identified a BF16 adapter-optimization issue. Treat the reported fusion result as an experimental baseline.
- The model can produce fluent but unsupported pathology descriptions and may inherit biases or errors from the source data and base model.

## Licenses and access terms

The code license has not been selected in this local moDCLV_Path release package; see `LICENSE.pending.md`. Qwen3.5, UNI, QUILT-1M, and any resulting adapter weights have separate terms. In particular, users must review the [Qwen3.5 model terms](https://huggingface.co/Qwen/Qwen3.5-9B), the gated [UNI model terms](https://huggingface.co/MahmoodLab/UNI), and the [QUILT-1M repository and data access terms](https://github.com/wisdomikezogwo/quilt1m) before downloading, using, or redistributing any component.

## Citation

See [CITATION.md](CITATION.md) for the upstream citations.
