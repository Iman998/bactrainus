# Third-Party Notices

This file records external resources referenced or consumed by Bactrainus. These resources are not relicensed by the repository's Apache-2.0 license.

## HotpotQA

Bactrainus uses the English HotpotQA dataset:

- Project: [HotpotQA](https://hotpotqa.github.io/)
- Authors: Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning
- Dataset license: [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/)
- Reference implementation license: Apache License 2.0

The canonical Bactrainus dataset release is an adaptation of HotpotQA. It preserves source identifiers and documents its transformations. Redistribution of that data must retain attribution and comply with CC BY-SA 4.0.

```bibtex
@inproceedings{yang2018hotpotqa,
  title     = {HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering},
  author    = {Yang, Zhilin and Qi, Peng and Zhang, Saizheng and Bengio, Yoshua and Cohen, William W. and Salakhutdinov, Ruslan and Manning, Christopher D.},
  booktitle = {Proceedings of EMNLP},
  year      = {2018}
}
```

## Meta Llama 3 legacy artifacts

The historical Bactrainus model repositories in the `bactrianus` Hugging Face organization, published in August 2024, are derived from Meta Llama 3 Instruct checkpoints and remain subject to the [Meta Llama 3 Community License](https://github.com/meta-llama/llama3/blob/main/LICENSE) and Acceptable Use Policy. The model weights are hosted on Hugging Face and are not bundled in this code repository.

Required attribution notice for redistributed Llama 3 materials:

> Meta Llama 3 is licensed under the Meta Llama 3 Community License, Copyright Meta Platforms, Inc. All Rights Reserved.

Built with Meta Llama 3.

The historical Hugging Face repository names and revisions are retained as artifact identifiers. This repository does not relabel those weights as Llama 3.1.

## Meta Llama 3.1 revised experiments

The controlled architecture experiments in the revised manuscript use Llama 3.1 Instruct checkpoints. Those checkpoints remain subject to the applicable [Meta Llama 3.1 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE) and use policy. They are not bundled in this repository, and the revised experiment identity is kept separate from the historical Llama 3 Hugging Face artifacts.

## Repository artwork

`assets/architecture.svg` is original Bactrainus project artwork. It contains no Flaticon assets or other third-party icon files. `assets/bactrainus-code.png` is project-provided repository artwork and is not derived from the architecture figure.
