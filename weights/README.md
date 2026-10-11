# Weights

No binary model weights are included in this GitHub package. This avoids placing large files and gated upstream material in the source repository.

The complete inference graph requires:

1. Qwen3.5-9B base weights and processor;
2. an approved UNI installation;
3. the Stage 1 fusion adapter, selected at step 30,000 in the reported run; and
4. the Stage 2 LoRA adapter, selected at step 6,626 in the reported run.

The Stage 1 adapter can be converted from a resumable checkpoint with `src/export_fusion_adapter.py`. The Stage 2 PEFT configuration is included in `stage2_lora/adapter_config.json`; the corresponding `adapter_model.safetensors` must be hosted separately after the applicable terms are checked.

Before publishing a weight link, add:

- the external download URL;
- the exact base-model and UNI revisions;
- SHA256 values;
- the weight-specific license or usage terms;
- a note that the custom loader must attach the LoRA to Qwen's language model and restore the Stage 1 fusion path.
