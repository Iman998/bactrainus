# Reproducibility

This document defines what the public repository can reproduce, which artifact identities must remain separate, and which reported quantities depend on historical external services.

## Experiment identity

| Dimension | Revised manuscript |
|---|---|
| Dataset | English HotpotQA distractor setting |
| Training size | 90,447 instances |
| Development size | 7,405 instances |
| Candidate set | 10 paragraphs per instance |
| Trainable model family | Llama 3.1 Instruct |
| Controlled sizes | 8B and 70B |
| Adaptation | LoRA |
| Screening cutoff | Latest checkpoints/endpoints available to the authors in January 2025 |

The historical public Hugging Face repositories were created in August 2024 and contain Llama 3 artifacts. They are related project artifacts, not a bit-equivalent release of the revised Llama 3.1 experiments. See [Model lineage](model-lineage.md).

## Configuration is authoritative

Use the files under [`configs/`](../configs/) as the machine-readable provenance record of revised settings. They are not directly executable trainer inputs. Resolve an explicit flat recipe, pin external revisions, and record that resolved file with the run manifest; do not copy values from historical notebooks or project filenames.

For a training job with per-device batch \(b\) and gradient accumulation \(g\), the reported local effective batch is

$$
b_{\mathrm{eff}}=b\,g.
$$

Distributed replication is not included in that manuscript definition. If a launcher multiplies the batch across data-parallel workers, record the resulting global batch separately.

All revised LoRA jobs use cosine learning-rate decay and dropout 0.05. The default warm-up ratio is 0.03; the 8B continuation on 15,661 hard instances with 70B-generated rationales uses 0.10.

## Inference settings

The reported generation groups are:

| Group | Temperature | Top-p |
|---|---:|---:|
| Prompted and proprietary API screening | 0.01 | 0.99 |
| Fine-tuned selector and reader modules | 0.0001 | 0.90 |
| Integrated one-step baseline | 0.0001 | 0.99 |

These values make generation near deterministic but do not guarantee identical output across libraries, kernels, quantization modes, model revisions, or hosted endpoints. Pin the model revision, tokenizer revision, package environment, prompt version, and decoding implementation.

## Hardware context

The reported experiments ran on a server with four NVIDIA A100 80-GB GPUs. Most 8B jobs used one device. Model-parallel large-model jobs used up to three devices. Hardware information is contextual; it does not replace logging of precision, sharding strategy, library versions, and peak memory for a reproduced run.

## Closed-source screening

Closed-source models were evaluated on a fixed 700-question prefix \(H_{700}\). For metric \(M\in\{\mathrm{EM},\mathrm{F1}\}\), a pooled open-model correction was defined as

$$
\widehat\kappa_M=
\frac{|\mathcal O|^{-1}\sum_{o\in\mathcal O}M_o(H_{\mathrm{dev}})}
{|\mathcal O|^{-1}\sum_{o\in\mathcal O}M_o(H_{700})},
\qquad
\widetilde M_c=\widehat\kappa_M M_c(H_{700}).
$$

The ratio direction is full development set divided by the 700-question subset, and EM and F1 are calibrated independently.

With \(N=7{,}405\), \(n=700\), and the finite-population correction, the worst-case 95% binomial half-width under an exchangeability reference design is approximately 3.52 percentage points:

$$
1.96\sqrt{\frac{N-n}{N-1}\frac{1}{4n}}\approx0.0352.
$$

Because the 700 records form a fixed prefix rather than a probability sample, this is a resolution reference, not unconditional design-based coverage. The calibration additionally assumes that an average open-model full/subset ratio transports to each proprietary model. Therefore, the displayed proprietary values are coarse historical screening estimates and must not support fine-grained ranking or significance claims.

## Metrics

Evaluation keeps three levels independent:

- Answer EM and token-overlap F1 after official-style normalization.
- Supporting-fact EM and F1 over exact `(title, sentence_index)` pairs.
- Joint EM/F1 combining answer and evidence components according to the HotpotQA protocol.

Use the package metric functions directly rather than parsing formatted tables. Preserve missing-generation accounting and evaluate paired comparisons over the intersection of valid records.

## Repository exclusions

The code repository intentionally excludes model weights, training data, development/test data, raw API payloads, predictions, and result files. This prevents accidental mixing of code and data licenses and keeps the Git history auditable.

The absence of result files means that a clean test run validates implementation behavior; it does not certify reproduction of every paper value. A full reproduction additionally requires the exact external artifacts, revisions, data, prompts, and compute environment.

## Reproduction checklist

1. Record the code commit and release tag.
2. Pin dataset, model, and tokenizer revisions.
3. Select the exact provenance specification without modifying it in place.
4. Resolve it into a new run-specific recipe with immutable external revisions and record every override.
5. Log environment versions, device topology, precision, and distributed strategy.
6. Preserve source IDs and deterministic input order.
7. Save prompts and parsed outputs with a schema version.
8. Keep raw generations separate from normalized metric inputs.
9. Report exclusions and missing generations.
10. Treat current hosted endpoints as new experiments, not silent replacements for the January 2025 snapshot.
