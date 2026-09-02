# Paper-to-artifact mapping

This file translates historical experiment names into the labels used in the
paper. The corrected reports for Tables 2, 3, and 5 have been checked against
the camera-ready tables; their rounded values match the paper.

| Saved name | Paper label |
|---|---|
| `base` | Student Baseline (no InfoNCE), i.e. the MLM checkpoint |
| `ft_in_batch` | Student Baseline, trained with task-only InfoNCE |
| `ft_in_batch_128_scratch` | Student Baseline (no MLM) |
| `distil_cosine` | `distil_embd_CLAP`, embedding alignment with a projector |
| `cosine_in_batch` | `distil_sim_CLAP`, similarity-score matching without InfoNCE |
| default `joint_in_batch` | `joint_sim_CLAP` or `joint_sim_Nova` |
| `embedding_cosine` | `joint_embd_CLAP` |
| `pairwiserank_retrieval` | `joint_rank_CLAP` |
| `m0.0_d1.0_n1.0` | MLM weight 0, distillation weight 1, InfoNCE weight 1 |
| `dt0.05` | Distillation temperature 0.05 |
| `analyze_grads` | Separate run with early gradient measurements enabled |

## Table mapping

- Table 2 uses `base`, `ft_in_batch_128_scratch`, `ft_in_batch`,
  `distil_cosine`, `cosine_in_batch`, and the seed-42 joint embedding,
  ranking, and similarity runs.
- Table 3 uses seeds 42, 7, and 99 for `ft_in_batch`, CLAP `joint_in_batch`,
  and Nova `joint_in_batch`. CLAP and Nova teacher weights are obtained from
  their public releases and are therefore not duplicated here.
- Table 5 uses `distil_cosine_64`, `distil_cosine` (128 tokens),
  `distil_cosine_256`, `distil_cosine_512`, and `distil_cosine_1024`. Each
  requires both `student.pth` and `projector.pth`.
- The RQ4 compact student is the 128-token `distil_cosine` checkpoint, called
  `distil_embd_CLAP` in the paper.

## RQ4 measurements

`rq4_efficiency.json` contains exactly the five CLAP and five
`distil_embd_CLAP` measurements used for Table 1, together with the protocol,
individual runs, and summary statistics. The timed compact student is the
128-token `distil_embd_CLAP` variant saved historically as `distil_cosine`.

## RQ2 gradient analysis

The final checkpoints from the three gradient-analysis runs and the corrected
per-step gradient history used for Figure 2 are included. The history retains
the historical raw dot product as well as the reconstructed cosine similarity.
Complete W&B run directories should not be published; the paper, README, and
checkpoint names document the reported settings.

The core artifact excludes Trainer checkpoints, optimizer and scheduler state,
W&B directories, Hugging Face `cache-*.arrow` files, generated retrieval
embeddings, and large derived teacher-representation datasets. The exact
seed-42 InfoNCE pair index is included under
`data/training_pairs/cross_project_split`; the released
`similarity_random_ft_precompute.py` script can regenerate it from the split
definition.
