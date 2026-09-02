# Artifact staging

This directory defines the files used to assemble the public artifact for the
RASE 2026 paper. It deliberately does not copy complete experiment directories.
In particular, Hugging Face Trainer checkpoints, optimizer state, W&B files,
generated retrieval embeddings, and recomputable `cache-*.arrow` shards are
excluded. The exact seed-42 training-pair index used by the reported InfoNCE
runs is included. See `PAPER_TO_FILES.md` for the mapping from historical run
names to the labels used in the paper.

## 1. Audit the core artifact on the remote machine

Run the staging script from the repository root after activating the `binai`
environment created from `environment.yml`:

```bash
python \
    artifact/stage_artifact.py \
    --source-root . \
    --check-only
```

The command fails if any paper-critical file is missing and prints the expected
archive size. Resolve missing paths in the manifest rather than copying a
similarly named run manually.

## 2. Stage the core artifact

Use a new destination directory:

```bash
python \
    artifact/stage_artifact.py \
    --source-root . \
    --destination ../BinAI_artifact_release
```

The destination must be empty. Every staged file is recorded in `SHA256SUMS`.
Metric reports are sanitized during staging: machine-specific dataset paths are
replaced by portable source descriptions, and their ground-truth-key metadata
is normalized to `(binary_name, function_name)`. An accompanying audit note
records that no such key spans multiple projects in the project test split, so
including `project` produces identical relevance labels and metric values.

## 3. RQ4 efficiency metadata provenance

`rq4_efficiency.json` is a static, human-readable record containing the five
CLAP and five 128-token `distil_embd_CLAP` runs used in Table 1. It includes the
individual measurements, summary statistics, and timing protocol. Staging
performs no cleaning or other data transformation.

## 4. RQ2 gradient-analysis provenance

`rq2_gradient_history.csv` contains the aligned per-step gradient histories for
the three objectives, and `rq2_gradient_summary.json` records the aggregate
values. The historical W&B field named `cos_sim_distill` actually stored the
raw gradient dot product. The artifact therefore retains that value under the
accurate name `raw_dot_product` and reconstructs cosine similarity as
`raw_dot_product / (infonce_grad_norm * distill_grad_norm)`. The reconstruction
script validates the shared step grid and the resulting cosine range.

To rebuild these files from fresh W&B CSV exports, run
`prepare_gradient_history.py --help` for the required paths.

## 5. W&B provenance

Complete W&B run directories are intentionally excluded because they contain
service-specific metadata and are not required for reproduction. The principal
training configurations are documented in the repository README and
`PAPER_TO_FILES.md`. For the gradient analysis, the exact per-step series used
for Figure 2 and Table 4 are provided in `rq2_gradient_history.csv`, together
with the reconstruction and plotting scripts.
