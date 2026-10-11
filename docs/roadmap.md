# moDCLV roadmap

moDCLV means **multiomics Design, Chat, Loop, Value**. `moDCLV_Path` is the pathology branch. The long-term objective is an evidence-traceable drug-discovery loop that connects cohort-scale biology to experimentally tested interventions and measurable translational value.

## Current release

The current release is a pathology image-text VLM based on UNI and Qwen3.5. It supports representation fusion and answer generation experiments on QUILT-derived pathology data. It does not yet integrate multiomics, operate an autonomous agent, discover intervention targets, run virtual experiments, or control wet-laboratory work.

## Planned capability layers

1. **Pathology foundation:** improve image understanding, spatial reasoning, quality control, and evaluation on independent pathology cohorts.
2. **Multiomics integration:** align pathology with genomics, transcriptomics, proteomics, clinical phenotypes, perturbation data, and other molecular measurements.
3. **Agentic design:** build a tool-using agent that can search large cohorts, generate and rank intervention hypotheses, identify missing evidence, and propose experiments.
4. **Virtual validation:** test hypotheses with structured cohort analyses, molecular network models, perturbation prediction, and other computational checks.
5. **Wet-laboratory loop:** connect selected hypotheses to reproducible laboratory assays, capture results, and feed validated or rejected hypotheses back into model training and planning.
6. **Value and translation:** measure robustness, reproducibility, biological validity, development relevance, and downstream product value rather than optimizing language-model loss alone.

Each layer should retain data provenance, model and tool versions, uncertainty estimates, negative results, and human or experimental review. The roadmap is a research direction; it is not a claim that the current checkpoint provides these capabilities.
