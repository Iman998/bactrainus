# Configuration Reference

The YAML files under [`configs/`](../configs/) are provenance specifications for the revised manuscript setup. They are intentionally explicit and omit unreported historical values rather than inventing defaults. They are not flat `SftRecipe` files and must not be passed directly to `bactrainus train`.

## Directory layout

```text
configs/
  data/
    hotpotqa_distractor.yaml
  models/
    legacy_hf_artifacts.yaml
    llama_3_1_8b_instruct.yaml
    llama_3_1_70b_instruct.yaml
  training/
    paragraph_selector.yaml
    question_decomposer.yaml
    reader_8b.yaml
    reader_8b_rationale_8b.yaml
    reader_8b_rationale_70b.yaml
    reader_70b.yaml
    sentence_selector.yaml
    single_stage_selector.yaml
  experiments/
    context_ablation.yaml
    foundation_model_screening.yaml
    generation.yaml
    integration_scenarios.yaml
```

`foundation_model_screening.yaml` enumerates all 26 historical display names and the 15-model paired calibration pool. Those names denote the latest checkpoints or endpoints available to the authors by the January 2025 cutoff; they are not aliases for live provider endpoints in 2026.

## Training-field semantics

| Field | Meaning |
|---|---|
| `experiment_id` | Stable human-readable experiment identifier |
| `task` | Module trained by the configuration |
| `model` | Reference to a checked-in base-model identity |
| `train_examples` | Number of records reported for the run |
| `epochs` | Complete passes through that run's training set |
| `per_device_batch_size` | Local micro-batch per device |
| `gradient_accumulation_steps` | Micro-batches accumulated before an optimizer step |
| `max_sequence_length` | Tokenized input/target sequence limit |
| `learning_rate` | Peak configured learning rate |
| `scheduler` | Learning-rate schedule |
| `warmup_ratio` | Fraction of steps used for warm-up |
| `lora.rank` | Low-rank dimension \(r\) |
| `lora.alpha` | LoRA scaling parameter \(\alpha\) |
| `lora.dropout` | Dropout applied in LoRA adapters |
| `lora.target_modules` | Exact module-name families adapted |

For a frozen matrix \(W_0\), LoRA applies

$$
W=W_0+\frac{\alpha}{r}BA.
$$

The `effective_batch_size` field is recorded as a derived check:

$$
b_{\mathrm{eff}}=(\text{per-device batch})(\text{gradient accumulation}).
$$

It does not include distributed replication.

## Target-module notation

The manuscript's `QKVO, MLP, head` notation maps to:

- `q_proj`, `k_proj`, `v_proj`, `o_proj`;
- `gate_proj`, `up_proj`, `down_proj`;
- `lm_head` when the language-model head is adapted.

The 70B direct reader excludes `lm_head`, matching the manuscript table.

## Overrides

Never edit a release configuration in place to describe a new run. Copy it to a run directory and record overrides explicitly. A run manifest should include:

- source configuration path and checksum;
- code commit;
- dataset and model revisions;
- all overridden keys;
- seed and launcher arguments;
- hardware, precision, and distributed strategy;
- output schema version.

A result is associated with the resolved configuration, not merely with a filename.

## Resolving an executable training recipe

`bactrainus train` accepts the strict flat schema implemented by `SftRecipe`. Resolution is a deliberate step: copy the relevant values from a manuscript specification, select the matching released dataset view, pin the exact model and dataset commits you can access, and choose a new output directory. The manuscript records the Llama 3.1 family and January 2025 cutoff but does not report immutable base-model Hub commits, so this repository does not fabricate them. At execution time, the trainer materializes the dataset's role/content messages through that pinned tokenizer's native chat template.

The following block is a **template, not an executable checked-in recipe**. Replace every angle-bracketed value and review the resulting diff before use:

```yaml
name: <run-name>
model_id: meta-llama/Meta-Llama-3.1-8B-Instruct
model_revision: <immutable-hugging-face-commit>
dataset_id: bactrianus/bactrainus-hotpotqa
dataset_config: reader-sft
dataset_revision: <immutable-hugging-face-commit>
split: train
output_dir: <new-output-directory>
epochs: 2
per_device_batch_size: 8
gradient_accumulation_steps: 32
learning_rate: 1.0e-4
max_sequence_length: 512
warmup_ratio: 0.03
scheduler: cosine
seed: 42
bf16: true
gradient_checkpointing: true
lora:
  rank: 64
  alpha: 128
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
    - lm_head
  bias: none
```

Validate a resolved file without starting training:

```bash
bactrainus train path/to/resolved-recipe.yaml
```

After verifying the model and dataset revisions, licenses, output path, hardware capacity, and run manifest, execution must be requested explicitly:

```bash
bactrainus train path/to/resolved-recipe.yaml --execute
```
