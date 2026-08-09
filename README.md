# Bactrainus: Optimizing Large Language Models for Multi-hop Complex Question Answering Tasks

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/Code%20License-Apache--2.0-4D7A97.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/arXiv-2501.06286-B31B1B.svg)](https://arxiv.org/abs/2501.06286)
[![Models and data](https://img.shields.io/badge/Hugging%20Face-bactrianus-FFD21E.svg)](https://huggingface.co/bactrianus)

Bactrainus is a modular selector--reader framework for multi-hop question answering over the HotpotQA distractor setting. It turns one opaque generation step into explicit, inspectable decisions: select relevant paragraphs, identify supporting sentences, and answer from compact evidence. Optional question decomposition refines sentence selection, while optional rationale supervision changes the reader target without treating generated rationales as gold evidence.

This repository is the cleaned research implementation accompanying the revised manuscript. It contains code, machine-readable experiment configurations, documentation, and small test fixtures. Model weights, release datasets, and experiment outputs are deliberately hosted separately.

> **Artifact identity matters.** The revised manuscript's controlled architecture experiments use Llama 3.1 Instruct checkpoints. The public models created in the `bactrianus` Hugging Face organization in August 2024 are historical Llama 3 artifacts. They are documented as legacy artifacts and are not renamed, rebased, or presented as the Llama 3.1 checkpoints evaluated in the revised manuscript.

## Architecture

![Bactrainus selector--reader architecture](assets/architecture.svg)

For a HotpotQA instance with question \(q\), ten supplied candidate paragraphs \(D\), gold answer \(a\), supporting facts \(S\), and gold paragraph set \(P\), the modular path is

$$
\widehat P=f_P(q,D),\qquad
U=f_Q(q,\widehat P),\qquad
\widehat S=f_S(q,\widehat P,U),\qquad
\widehat a=f_R(q,C(\widehat P,\widehat S)).
$$

The decomposition output \(U\) is optional. The reader context \(C\) can contain selected supporting sentences or full selected paragraphs. This is fixed-candidate evidence selection: Bactrainus does not search all of Wikipedia at inference time.

See [Architecture](docs/architecture.md) for the formal interfaces and failure boundaries.

## Repository scope

| Resource | Location | Included here |
|---|---|---:|
| Clean implementation and tests | This repository | Yes |
| Revised experiment configurations | [`configs/`](configs/) | Yes |
| Canonical training data | [`bactrianus/bactrianus-hotpotqa`](https://huggingface.co/datasets/bactrianus/bactrianus-hotpotqa) | No |
| Historical Llama 3 models | [`bactrianus`](https://huggingface.co/bactrianus) | No |
| Evaluation predictions and result files | Not distributed in the code repository | No |
| Full methodology and reported results | [arXiv:2501.06286](https://arxiv.org/abs/2501.06286) | No |

The canonical dataset release has five train-only configurations, each keyed by the same 90,447 unique HotpotQA source IDs:

- `structured`
- `reader-sft`
- `paragraph-selector-sft`
- `sentence-selector-sft`
- `joint-selector-reader-sft`

No development/test examples, predictions, evaluation outputs, cross-lingual records, or unverified synthetic rationale/decomposition records are included in that release. See [Data](docs/data.md).

## Installation

```bash
git clone https://github.com/Iman998/bactrainus.git
cd bactrainus
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

Verify the installation:

```bash
bactrainus --help
python -m pytest
```

## Command-line workflow

The command-line interface separates data construction, training, inference, evaluation, and integrated execution:

```bash
bactrainus data validate --help
bactrainus data build --help
bactrainus train --help
bactrainus infer --help
bactrainus evaluate --help
bactrainus pipeline --help
bactrainus version
```

Validate the official training file and materialize an ID-preserving view locally:

```bash
bactrainus data validate data/hotpot_train_v1.1.json --expected-count 90447
bactrainus data build data/hotpot_train_v1.1.json outputs/structured.jsonl \
  --view structured --expected-count 90447
```

The checked-in YAML files are **provenance specifications**, not directly executable trainer recipes. They preserve the schedules reported in the revised manuscript, including per-device batch size, gradient accumulation, maximum sequence length, learning rate, LoRA rank and scale, target modules, warm-up ratio, and generation settings. `bactrainus train` instead accepts a separately resolved flat `SftRecipe` with immutable model and dataset revisions plus an explicit output directory. The runner converts the released role/content messages with the selected tokenizer's native chat template. It validates the recipe without launching a job; the expensive run starts only when `--execute` is supplied. See [Configuration](docs/configuration.md).

## Python interfaces

The package keeps data conversion, parsing, generation, pipeline orchestration, and evaluation independent:

```python
from bactrainus.data.hotpot import load_hotpot_examples
from bactrainus.evaluation import evaluate_answers, evaluate_evidence, evaluate_joint
from bactrainus.pipeline import BactrainusPipeline
```

Important public boundaries include:

- `load_hotpot_examples` and strict HotpotQA schema parsers;
- deterministic structured and chat-view builders;
- `TextGenerator`, a backend-neutral batched generation protocol;
- `BactrainusPipeline`, which composes paragraph selection, optional decomposition, sentence selection, and reading;
- answer, evidence, joint, and calibration metrics implemented independently of model backends.

See the docstrings and [Reproducibility](docs/reproducibility.md) before connecting a model service. Never place API credentials in configuration files or command history.

Run the bundled synthetic example without downloading model weights or benchmark data:

```bash
python examples/build_reader_view.py
```

## Configuration map

| Area | Configuration |
|---|---|
| Dataset boundary | [`configs/data/hotpotqa_distractor.yaml`](configs/data/hotpotqa_distractor.yaml) |
| Revised base models | [`configs/models/`](configs/models/) |
| Reader adaptation | [`configs/training/reader_*.yaml`](configs/training/) |
| Selector and decomposition adaptation | [`configs/training/*selector*.yaml`](configs/training/) and [`question_decomposer.yaml`](configs/training/question_decomposer.yaml) |
| Generation settings | [`configs/experiments/generation.yaml`](configs/experiments/generation.yaml) |
| Evidence conditions | [`configs/experiments/context_ablation.yaml`](configs/experiments/context_ablation.yaml) |
| Integrated scenarios | [`configs/experiments/integration_scenarios.yaml`](configs/experiments/integration_scenarios.yaml) |
| Legacy Hugging Face identity map | [`configs/models/legacy_hf_artifacts.yaml`](configs/models/legacy_hf_artifacts.yaml) |

## Reproducibility boundaries

- The official English HotpotQA distractor split contains 90,447 training and 7,405 development instances, with ten candidate paragraphs per instance.
- Reported architecture experiments use the full development split and Llama 3.1 checkpoints.
- The 26-model screening is a January 2025 snapshot. Proprietary services may change after that date.
- Closed-source screening values were based on a fixed 700-question subset and a full-set/subset calibration ratio estimated from paired open-model runs. They are coarse screening estimates, not significance tests.
- Reported training contrasts are single-run observations unless explicitly stated otherwise in the paper.
- Historical Hugging Face Llama 3 weights are not bit-equivalent substitutes for revised Llama 3.1 experiments.

These constraints are intentional and are documented in greater detail in [Reproducibility](docs/reproducibility.md) and [Model lineage](docs/model-lineage.md).

## Development standards

The implementation favors small, typed units with one responsibility, explicit protocols at external boundaries, deterministic data transformations, strict parsing, and side-effect-free metric functions. New contributions must preserve source IDs and ordering, avoid hidden global state, and include focused tests.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the review checklist and [SECURITY.md](SECURITY.md) for responsible disclosure and credential handling.

## Licensing and attribution

The repository's original code, documentation, and architecture artwork are licensed under Apache License 2.0. This does not relicense external data or model weights.

- HotpotQA data is distributed under CC BY-SA 4.0.
- Historical Llama 3 weights remain subject to the Meta Llama 3 Community License.
- Llama 3.1 checkpoints remain subject to the applicable Meta Llama license.
- The architecture SVG is original project artwork and does not contain Flaticon assets.

See [NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistribution.

## Citation

```bibtex
@article{barati2025bactrainus,
  title   = {Bactrainus: Optimizing Large Language Models for Multi-hop Complex Question Answering Tasks},
  author  = {Barati, Iman and Ghafouri, Arash and Minaei-Bidgoli, Behrouz},
  journal = {arXiv preprint arXiv:2501.06286},
  year    = {2025},
  doi     = {10.48550/arXiv.2501.06286},
  url     = {https://arxiv.org/abs/2501.06286}
}
```

For the complete methodology, experimental protocol, and results, read the paper: https://arxiv.org/abs/2501.06286
