# Model Lineage

Bactrainus has two distinct model-artifact generations. They must not be conflated.

## Revised manuscript checkpoints

The revised manuscript's trainable selector, decomposer, and reader modules use Llama 3.1 Instruct checkpoints at 8B or 70B scale. Their experiment settings are recorded under [`configs/models/`](../configs/models/) and [`configs/training/`](../configs/training/).

Those configurations document the revised experiments. They do not change the identity or contents of the public 2024 Hugging Face repositories.

## Historical Hugging Face artifacts

The following public repositories are legacy Llama 3 artifacts created in August 2024. The revisions below are the public revisions observed during the release audit.

| Repository | Intended role | Public artifact status | Audited revision |
|---|---|---|---|
| `bactrianus/HotpotQA-Reader-Llama-3-8B-Instruct` | Direct reader | Full merged checkpoint | `98c63afd55b5b4bd46890165e85e69c7d7d10a3d` |
| `bactrianus/HotpotQA-Reader-Llama-3-70B-Instruct` | Direct reader | Full merged checkpoint | `8f76e68c65955843fffbf5c9b0e0ae1e446fbf1a` |
| `bactrianus/HotpotQA-Reader-CoT-Llama-3-8B-Instruct` | Rationale-generating reader | Full merged checkpoint | `852277e5b9534ff51a66adbad1ad43b7a3ef4457` |
| `bactrianus/HotpotQA-OneStep-Retriever-Llama-3-8B-Instruct` | Joint selector/reader | Full merged checkpoint | `7842b5845df7be3618e077a618866a3e0826972e` |
| `bactrianus/HotpotQA-Paragraph-Retriever-Llama-3-8B-Instruct` | Paragraph selector | Full merged checkpoint | `1bc650152552e5a0ee7ee4afdfae01abc3bdc76e` |
| `bactrianus/HotpotQA-Sentence-Retriever-Llama-3-8B-Instruct` | Sentence selector | Full merged checkpoint | `0ba9d0d84d7913f3d614061ebd71dca8690e9f93` |
| `bactrianus/HotpotQA-Question-Decomposition-Llama-3-8B-Instruct` | Question decomposer | Metadata-only placeholder; no weights | `a55a003d5f1b748598b428facd3ffec306674546` |
| `bactrianus/HotpotQA-Sentence-Retriever-QD-Llama-3-8B-Instruct` | Decomposition-conditioned sentence selector | Metadata-only placeholder; no weights | `389dcb33320132800ca8198e764d46755c13fbcf` |

Six full repositories expose `LlamaForCausalLM` checkpoints with tokenizer and generation configuration files. The two placeholder repositories contain only repository metadata and a README; users cannot load weights from them.

## Identity rules

1. Do not change a model card's base model from Llama 3 to Llama 3.1 without replacing and versioning the actual weights.
2. Do not cite revised Llama 3.1 paper scores as if they were direct evaluations of a historical Llama 3 Hub checkpoint.
3. Pin the full Hugging Face revision when using a historical artifact.
4. Label the two metadata-only repositories as unavailable until their intended weights are verified and uploaded.
5. If a repository is renamed for license compliance, preserve the old-to-new mapping and immutable revision in release notes.
6. Do not infer missing legacy optimizer settings from the revised manuscript configuration.

## Data relationship

All historical models are associated with HotpotQA-derived task data. The complete canonical suite at [`bactrianus/bactrainus-hotpotqa`](https://huggingface.co/datasets/bactrianus/bactrainus-hotpotqa) defines eight clean, deterministic, ID-aligned training views. Those views are not asserted to be byte-for-byte identical to every historical 2024 training file.

Model-card metadata therefore names the official `hotpotqa/hotpot_qa` source dataset and links the canonical Bactrainus release textually with this provenance qualification.

## License requirements

The historical weights remain subject to the Meta Llama 3 Community License. Redistribution requires the Meta license agreement, the required attribution notice, and prominent “Built with Meta Llama 3” wording. HotpotQA-derived training data remains subject to CC BY-SA 4.0.

The model cards under [`release/hf_model_cards`](../../hf_model_cards/) document these constraints without modifying or relabeling weights.
