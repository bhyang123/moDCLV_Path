# Reproducibility

moDCLV_Path is currently the pathology image-text branch of the broader moDCLV direction (multiomics Design, Chat, Loop, Value). Reproduction of this release covers the pathology VLM only; it does not reproduce a multiomics agent, virtual target validation, or wet-laboratory workflow.

Record these values with every released weight artifact:

- Qwen3.5 model ID and exact revision or commit;
- UNI model ID and exact revision or download hash;
- Stage 1 fusion checkpoint and Stage 2 LoRA checkpoint;
- annotation and image-manifest SHA256 values;
- Python, PyTorch, Transformers, PEFT, timm, and flash-linear-attention versions;
- distributed device count, global batch size, gradient accumulation, seed, learning rate, precision, and checkpoint-selection rule;
- processor, tokenizer, chat template, image resize/normalization, and generation settings.

The release scripts deliberately require paths through arguments or environment variables. They do not assume the author's directory layout. The smoke scripts use only a few samples and two optimizer steps; they verify imports, data loading, and the forward/backward path but are not a quality test.

The Stage 2 LoRA adapter wraps `qwen.model.language_model`, not the outer multimodal `Qwen3_5ForConditionalGeneration` object. A custom loader that reconstructs the native visual path and Stage 1 fusion path is therefore required for faithful inference.
