# Experiment Specifications

This directory records the revised manuscript's dataset boundary, model identities, training hyperparameters, decoding profiles, and integration scenarios in machine-readable YAML.

These files are **provenance specifications**. They retain information that does not belong in a trainer input, such as reported example counts, evidence conditions, release availability, artifact lineage, and manuscript-only notes. Consequently, files under `training/` are not accepted directly by `bactrainus train`.

To run training:

1. choose the relevant specification under `training/`;
2. select the compatible public dataset view;
3. resolve the base-model and dataset identities to immutable Hub commits;
4. copy only the strict `SftRecipe` fields into a new run file;
5. record the source specification checksum and every override;
6. validate the resolved file with `bactrainus train <recipe>`;
7. add `--execute` only after reviewing compute, output, license, and provenance settings.

The manuscript did not report immutable Llama 3.1 base-model commits. None is invented here. A placeholder-only recipe example and the full field mapping are documented in [`docs/configuration.md`](../docs/configuration.md).

The `models/legacy_hf_artifacts.yaml` inventory describes historical public Llama 3 repositories. It must not be used as a substitute for the revised Llama 3.1 model specifications.
