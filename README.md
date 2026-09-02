# Can We Trust Strong Teachers?

Code and evaluation artifacts for the paper:

> **Can We Trust Strong Teachers? Auditing Knowledge Distillation for Binary Code Similarity Detection**<br>
> Minh Khang Van, Yunru Wang, and Johannes Kinder<br>
> 2026 Workshop on Reliable and Trustworthy Automated Software Engineering (RASE '26)

- Paper: [https://doi.org/10.1145/3820756.3844917](https://doi.org/10.1145/3820756.3844917)
- Replication package: [https://doi.org/10.5281/zenodo.22249637](https://doi.org/10.5281/zenodo.22249637)

This repository studies whether supervision from large binary-code models improves a compact encoder for binary code similarity detection (BCSD) once that encoder is already trained directly for retrieval. The experiments use CLAP and Nova as frozen teachers and compare three forms of knowledge transfer:

- embedding alignment;
- pairwise similarity-score matching; and
- relative-ranking distillation.

The student is a 6-layer BERT encoder with a 512-dimensional hidden state and a default 128-token context window. It is trained with masked language modeling (MLM), InfoNCE, distillation, or combinations of these objectives.

## Main findings

The controlled experiments show that direct InfoNCE training accounts for most of the compact student's retrieval performance. Distillation transfers useful BCSD information in isolation, but adds only a small benefit once task training is present, and its effect depends on the supervision objective. The artifact also supports analyses of gradient interaction, input coverage, and corpus-encoding efficiency.

Please consult the paper for the complete experimental design, results, limitations, and interpretation.

## Artifact scope

The repository contains code for:

- extracting and normalizing x64 disassembly;
- constructing cross-project and cross-binary splits;
- training the WordPiece tokenizer and MLM student initialization;
- precomputing CLAP and Nova teacher representations;
- task-only, distillation-only, and joint student training;
- exporting teacher and student embeddings; and
- exact full-corpus MRR, Recall@k, and nDCG@k evaluation.

Large datasets, model checkpoints, generated embeddings, W&B files, and metric reports are intentionally excluded from Git. The materials needed to reproduce the reported results are archived in the [Zenodo replication package](https://doi.org/10.5281/zenodo.22249637). Store locally generated files under `outputs/` or another external artifact directory.

## Requirements

The experiments require Python 3.10 or newer. Training and embedding generation require a CUDA-capable GPU; exact metric calculation can run efficiently on CPU.

Core dependencies are:

- PyTorch;
- Hugging Face Transformers and Datasets;
- Tokenizers;
- NumPy;
- tqdm;
- Weights & Biases; and
- GPUtil for GPU measurements.

A pinned copy of the Linux/CUDA environment used for the reported experiments
is provided in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate binai
```

The environment records Python 3.11.13 and pins the Linux x86-64 CUDA 12.8
PyTorch wheels directly from the official PyTorch index. The complete pip
package snapshot is also provided in `requirements-lock.txt` for provenance.
Because GPU wheels and drivers are platform-specific, users on a different
architecture or CUDA platform must select the corresponding PyTorch build while
retaining the recorded library versions.

Set W&B credentials through the environment rather than storing them in this repository. IDA Pro is additionally required only when reconstructing the disassembly dataset from the original binaries.

## Data and teacher models

The experiments use the x64 portion of the public [Cisco Talos binary-function similarity dataset](https://github.com/Cisco-Talos/binary_function_similarity). The project split keeps training, validation, and test projects disjoint.

The teacher models are:

- [CLAP](https://huggingface.co/hustcw/clap-asm), a retrieval-oriented binary representation model; and
- [Nova](https://huggingface.co/lt-asset/nova-1.3b), a generative assembly model used as a contrasting teacher.

The scripts load teacher checkpoints with `trust_remote_code=True`. Review the corresponding model repositories before execution. The Nova paths in `precompute.py` and `inference_embeddings.py` may need to be changed to the checkpoint location on the target machine.

## Expected artifact layout

The default workflow assumes the following layout:

```text
outputs/
├── assembly_x64/
├── assembly_x64_1024_clap/
├── assembly_x64_1024_nova/
├── cosine_random_ft/
│   └── cross_project_split/
├── cross_project_split.json
├── cross_binary_split.json
├── tokenizer/
├── bert_mlm_project/
│   └── best_model/
├── bert_project/
│   ├── clap/
│   └── nova/
└── inference/
    ├── datasets/
    │   └── project/
    │       ├── clap/
    │       └── nova/
    └── metrics/
        └── project/
```

Paths can be changed with the command-line arguments shown by each script's `--help` option.

## Reproduction workflow

### 1. Prepare the dataset

To reconstruct the Hugging Face dataset from downloaded binaries using IDA:

```bash
python collect_asm.py \
    --data_dir /path/to/talos/binaries \
    --output_dir outputs/assembly_x64 \
    --ida_path /path/to/idat64 \
    --ida_script process_asm.py
```

Create the cross-project and cross-binary split definitions:

```bash
python create_split.py \
    --data_dir outputs/assembly_x64 \
    --output_dir outputs
```

Each split file stores dataset `unique_id` values. The paper reports results for the `project` split.

### 2. Train the tokenizer and MLM initialization

Train the 33,555-token WordPiece tokenizer:

```bash
python train_tokenizer.py \
    --data_dir outputs \
    --output_dir outputs
```

Pretrain the compact student with MLM:

```bash
python pretrain_bert.py \
    --data_dir outputs \
    --output_dir outputs \
    --split project
```

The resulting initialization is expected at `outputs/bert_mlm_project/best_model`.

### 3. Precompute teacher representations

Teacher representations are added to separate copies of the assembly dataset:

```bash
python precompute.py --data_dir outputs --model clap --max_len 1024 --batch_size 64
python precompute.py --data_dir outputs --model nova --max_len 1024 --batch_size 64
```

These commands produce `assembly_x64_1024_clap` and `assembly_x64_1024_nova`, respectively.

Generate the deterministic anchor/positive/negative index used by the
task-only and joint InfoNCE variants:

```bash
python similarity_random_ft_precompute.py \
    --data_dir outputs/assembly_x64_1024_clap \
    --splits_dir outputs \
    --output_dir outputs/cosine_random_ft \
    --split project \
    --top_k 10 \
    --seed 42
```

The resulting `cosine_random_ft/cross_project_split` dataset contains IDs
rather than teacher representations, so the same index is used for CLAP and
Nova experiments. The exact seed-42 index used for the paper is included in the
Zenodo artifact under `data/training_pairs/cross_project_split`; the command
above regenerates it from the released split definition.

### 4. Train student variants

Task-only InfoNCE baseline:

```bash
python train.py \
    --data_dir outputs \
    --output_dir outputs \
    --split project \
    --teacher_type clap \
    --method ft_in_batch \
    --max_len 128 \
    --batch_size 128 \
    --num_train_epochs 6 \
    --seed 42
```

Joint similarity-score distillation with InfoNCE:

```bash
python train.py \
    --data_dir outputs \
    --output_dir outputs \
    --split project \
    --teacher_type clap \
    --method joint_in_batch \
    --distill_loss_type mse \
    --max_len 128 \
    --batch_size 128 \
    --lambda_mlm 0.0 \
    --lambda_distill 1.0 \
    --lambda_nce 1.0 \
    --num_train_epochs 6 \
    --seed 42
```

The principal method mapping is:

| Paper variant | Important training arguments |
|---|---|
| MLM-only checkpoint | `pretrain_bert.py` |
| Task-only InfoNCE | `--method ft_in_batch` |
| Pure embedding alignment | `--method distil_cosine` (uses `nn.CosineEmbeddingLoss` on projected, normalized embeddings) |
| Similarity distillation | `--method cosine_in_batch --distill_loss_type mse` |
| Joint similarity matching | `--method joint_in_batch --distill_loss_type mse` |
| Joint relative ranking | `--method joint_in_batch --distill_loss_type pairwiserank_retrieval` |
| Joint embedding alignment | `--method joint_in_batch --distill_loss_type embedding_cosine --use_projector_in_ft` |

Use `--teacher_type nova` for Nova-supervised variants. Other relevant controls include `--from_scratch`, `--filter_truncated`, `--analyze_gradients`, `--distill_temperature`, and `--finetune_checkpoint`. Run `python train.py --help` for the complete interface.

### 5. Generate retrieval embeddings

For a trained student checkpoint:

```bash
CHECKPOINT=joint_in_batch_128_bs128_no_proj_m0.0_d1.0_n1.0_seed42

python inference_embeddings.py \
    --data_dir outputs \
    --output_dir outputs \
    --split project \
    --model clap \
    --method "${CHECKPOINT}" \
    --checkpoint_dir "outputs/bert_project/clap/${CHECKPOINT}" \
    --batch_size 64
```

For a zero-shot teacher:

```bash
python inference_embeddings.py \
    --data_dir outputs \
    --output_dir outputs \
    --split project \
    --model clap \
    --method clap \
    --is_teacher \
    --batch_size 64
```

`--test_dataset_path` can point to an already filtered Hugging Face test dataset. This avoids loading and filtering the complete dataset on an evaluation-only machine.

Embedding datasets are written to:

```text
<output_dir>/inference/datasets/<split>/<teacher>/<method>-embeddings
```

Each dataset contains the columns `unique_id` and `embedding`. Runtime and GPU statistics are stored in the adjacent metadata JSON file.

### 6. Calculate exact retrieval metrics

```bash
python calculate_ranking_metrics.py \
    --data_dir outputs \
    --output_dir outputs \
    --split project \
    --model_name clap \
    --method "${CHECKPOINT}" \
    --embeddings_dataset_path \
        "outputs/inference/datasets/project/clap/${CHECKPOINT}-embeddings" \
    --device cpu \
    --query_batch_size 128 \
    --candidate_block_size 32768 \
    --k_values 1 512 1024
```

The implementation computes exact full-corpus rankings in query and candidate blocks; `candidate_block_size` controls memory use, not the evaluation population or metric definition. Reduce either block size if memory is constrained.


Reports are written to:

```text
<output_dir>/inference/metrics/<split>/<teacher>/<method>_corrected_metrics_report.json
```

## Retrieval protocol

The reported project-split evaluation uses all 514,071 x64 test functions as the retrieval corpus rather than sampling a small candidate pool. For every eligible query:

- the query itself is excluded from the candidate pool;
- all remaining test functions are retained as candidates;
- another function is relevant when it represents the same source-level function identity;
- queries may have multiple relevant compiled instances;
- embeddings are explicitly L2-normalized and ranked by cosine similarity;
- MRR uses the rank of the first relevant non-self candidate;
- Recall@k measures the fraction of all relevant instances retrieved by rank `k`; and
- nDCG@k uses binary relevance and discounts relevant instances by rank.

The exact implementation and tie policy are documented in `src/utils/ranking_metrics.py` and recorded in every JSON report.

## Repository structure

```text
collect_asm.py                      Build the assembly-function dataset with IDA
process_asm.py                      IDA-side extraction and jump-target normalization
create_split.py                     Create project and binary split definitions
train_tokenizer.py                  Train the student WordPiece tokenizer
pretrain_bert.py                    MLM pretraining for the compact student
precompute.py                       Add CLAP or Nova teacher representations
train.py                            Task, distillation, and joint training
inference_embeddings.py             Export teacher or student retrieval embeddings
similarity_random_ft_precompute.py  Generate the deterministic anchor/positive/negative index
calculate_ranking_metrics.py        Exact blockwise full-corpus evaluation
src/models/                         Student and distillation model definitions
src/utils/                          Dataset, GPU, and ranking-metric utilities
```

## Reproducibility notes

- Record the split, teacher, method, seed, loss weights, maximum sequence length, batch size, and hardware for every run.
- The paper's primary student results use three seeds where reported; ablation experiments may be single runs.
- Generated artifacts are large and should remain outside Git.
- Re-normalizing already L2-normalized embeddings is numerically harmless and makes the cosine protocol explicit.
- CLAP and Nova use model-specific tokenization and preprocessing. Consult the paper's threats to validity before comparing their zero-shot representations.

## Citation

If you use this work, please cite the paper:

```bibtex
@inproceedings{van2026trust,
    author    = {Van, Minh Khang and Wang, Yunru and Kinder, Johannes},
    title     = {Can We Trust Strong Teachers? Auditing Knowledge Distillation
                    for Binary Code Similarity Detection},
    booktitle = {Proceedings of the 1st International Workshop on Reliable and
                    Trustworthy Automated Software Engineering (RASE '26)},
    publisher = {Association for Computing Machinery},
    year      = {2026},
    month     = oct,
    isbn      = {979-8-4007-2787-0/2026/10},
    doi       = {10.1145/3820756.3844917},
    url       = {https://doi.org/10.1145/3820756.3844917}
}
```

The archived replication package is available at
[https://doi.org/10.5281/zenodo.22249637](https://doi.org/10.5281/zenodo.22249637).
Please use the citation exported by Zenodo when citing the package separately.

## License and third-party materials

Source code authored for this project is released under the
[Apache License 2.0](LICENSE). The associated ACM paper is published under the
[Creative Commons Attribution 4.0 International (CC BY 4.0) license](https://creativecommons.org/licenses/by/4.0/).

The replication package also contains experimental outputs and derived
benchmark data. Third-party datasets, models, and derived materials remain
subject to their respective upstream licenses and terms of use. CLAP and Nova
teacher checkpoints are not redistributed.
