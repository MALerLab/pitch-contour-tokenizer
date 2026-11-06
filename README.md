# Pitch Contour Tokenization using VQ-VAE and Its Application on Korean Traditional Music Analysis

**Seonguk Ju, Seola Cho, Suin Chung, Danbinaerin Han, Dasaem Jeong**

The official implementation of the ISMIR 2026 paper *"Pitch Contour
Tokenization using VQ-VAE and Its Application on Korean Traditional
Music Analysis"*.

[**Paper**](https://arxiv.org/abs/2608.10979) &nbsp;|&nbsp;
[**Demo**](https://seongukju.github.io/pitch-contour-tokenizer-page/) &nbsp;|&nbsp;
[**Data & Weights**](https://github.com/MALerLab/pitch-contour-tokenizer/releases)

[![arXiv](https://img.shields.io/badge/arXiv-2608.10979-b31b1b)](https://arxiv.org/abs/2608.10979)
[![Code License: MIT](https://img.shields.io/badge/code%20license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)](pyproject.toml)
[![Release](https://img.shields.io/badge/checkpoints-v2.0.0-green)](https://github.com/MALerLab/pitch-contour-tokenizer/releases)

The model learns a discrete vocabulary of local pitch-contour patterns
directly from unlabeled audio:

* **Contour tokenizer** — A 1D convolutional VQ-VAE over fixed-length
  (1.28 s, 100 Hz) pitch-contour segments. Each median-subtracted
  segment is encoded, quantized to one of 256 codebook entries — its
  discrete token — and decoded back to a contour.
* **Transformation-minimized loss** — The reconstruction is evaluated
  under the best alignment among candidate temporal and pitch-domain
  transformations (tempo, zoom, y-offset) and only the minimum error is
  back-propagated, so tokens stay stable across segmentation positions
  and small variations in timing and pitch range.

| Config | Model | Loss | Notes |
|---|---|---|---|
| `contour_ae` | Conv1DAE | MSE | AE baseline (+ post-hoc KMeans codes) |
| `contour_vqvae` | Conv1DVQVAE | MSE | VQ-VAE baseline |
| `xoffset_vqvae` | OffsetConv1DVQVAE | Offset-MSE | temporal alignment: tempo candidates `[50, 52, 55, 58, 60]` |
| `foffset_vqvae` | OffsetConv1DVQVAE | Offset-MSE | all transformations: tempo + zoom `[0.9 … 1.1]` + y-offset `[-0.1, 0, 0.1]` octave |

---

## Install

**With [uv](https://docs.astral.sh/uv/) (recommended)** — installs the exact
locked versions from `uv.lock`:

```bash
# one-time uv install, if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync   # creates .venv/ and installs all dependencies
```

Then either prefix commands with `uv run` (no activation needed), e.g.
`uv run python src/train.py ...`, or activate the environment with
`source .venv/bin/activate` and use the commands below as written.

**With pip:**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10–3.13. `uv.lock` pins the training environment of the
released checkpoints — Python 3.10, PyTorch 2.8.0 (CUDA 12.8), scikit-learn
1.6.1, NumPy 2.2.6, umap-learn 0.5.11 with numba 0.61.2 / pynndescent 0.6.0
(the autoencoder's `token_umap.joblib` is a numba-bound pickle and only loads
under these versions).

---

## Download data & weights

Checkpoints are attached to the **v2.0.0** release and the datasets to the
**v1.0.0** release (they have not changed) on
[GitHub Releases](https://github.com/MALerLab/pitch-contour-tokenizer/releases):

| Release | Asset | Contents | Size |
|---|---|---|---|
| v2.0.0 | `checkpoints.tar.gz` | 4 models, seed 42 (`best_model.pt` + `config.yaml`; the autoencoder also ships its token space `token_*.joblib`) + `PROVENANCE.md` | ~33 MB |
| v2.0.0 | `checkpoints_seeds.tar.gz` | the same 4 models for all 10 training seeds (`weights/seeds/<model>/s<seed>/`; each autoencoder seed ships its own `token_*.joblib`) | ~330 MB |
| v2.0.0 | `SHA256SUMS.txt` | checksums for the two archives above | — |
| v1.0.0 | `inhouse_data.tar.gz` | pansori training corpus (CREPE contour CSVs) + `rename_map.csv` | ~2.0 GB |
| v1.0.0 | `nia_pansori_meta.tar.gz` | NIA Gugak contours, annotated pansori contours, metadata CSVs | ~0.5 GB |

Extract from the repository root:

```bash
sha256sum -c SHA256SUMS.txt              # optional integrity check (v2.0.0 archives)
tar xzf checkpoints.tar.gz               # → weights/<model>/
tar xzf checkpoints_seeds.tar.gz         # → weights/seeds/<model>/s<seed>/   (optional)
mkdir -p data
tar xzf inhouse_data.tar.gz -C data/     # → data/inhouse_data/
tar xzf nia_pansori_meta.tar.gz -C data/ # → data/{nia_gugak_data,pansori_annotated_data,meta}/
```

Checkpoint ↔ config ↔ paper mapping:

| Checkpoint dir | Config | Paper method |
|---|---|---|
| `weights/autoencoder` | `contour_ae` | Autoencoder baseline |
| `weights/vqvae` | `contour_vqvae` | VQ-VAE |
| `weights/temporal_align_vqvae` | `xoffset_vqvae` | + temporal alignment |
| `weights/all_transformation_vqvae` | `foffset_vqvae` | + all transformations |

`weights/PROVENANCE.md` (also tracked as `results/PROVENANCE.md`) lists, for every
checkpoint, the training seed, source run, code commit and its metrics.
With the released data in place, the presets `data=crepe` (default split: the artist-level
manifest) and `exp=nia_pair` point at it directly, e.g.
`python src/train.py --config-name foffset_vqvae data=crepe exp=nia_pair exp.run=true`.

### Results (mean ± std over 10 training seeds)

Token consistency (KLD ↓, code-flip accuracy ↑ over frame offsets 4–60 and
4–124 of the test split) and downstream NIA ornament classification
(macro-F1 / mAP ↑ on the NIA test set and its vocal subset). Code-flip
accuracy compares the token of each shifted segment with the token at the
same position for shifts below 64 frames (half a window) and with the
following token for shifts of 64 frames or more. Per-seed values:
`results/seed_results.csv` (tokenizers) and `results/edtcn_seed_results.csv`
(ED-TCN); LaTeX: `results/table_token_consistency_10seed.tex`.

| Method | KLD 4–60 | KLD 4–124 | Acc 4–60 | Acc 4–124 | F1 NIA | F1 Vocal | mAP NIA | mAP Vocal |
|---|---|---|---|---|---|---|---|---|
| Autoencoder | 1.596 ± 0.022 | 1.918 ± 0.026 | 0.141 ± 0.006 | 0.167 ± 0.009 | 0.288 ± 0.008 | 0.307 ± 0.011 | 0.268 ± 0.003 | 0.290 ± 0.007 |
| VQ-VAE | 0.628 ± 0.120 | 0.818 ± 0.142 | 0.503 ± 0.080 | 0.545 ± 0.077 | 0.265 ± 0.009 | 0.305 ± 0.011 | 0.249 ± 0.009 | 0.298 ± 0.008 |
| + temporal alignment | **0.543 ± 0.053** | **0.742 ± 0.067** | **0.520 ± 0.036** | **0.562 ± 0.035** | 0.298 ± 0.010 | 0.325 ± 0.016 | 0.278 ± 0.007 | 0.322 ± 0.009 |
| + all transformations | 0.582 ± 0.046 | 0.795 ± 0.059 | 0.493 ± 0.031 | 0.536 ± 0.030 | **0.305 ± 0.015** | **0.331 ± 0.012** | **0.283 ± 0.009** | **0.325 ± 0.012** |
| ED-TCN (supervised reference) | – | – | – | – | 0.580 ± 0.018 | 0.534 ± 0.014 | 0.622 ± 0.016 | 0.525 ± 0.012 |

Bold marks the best tokenizer. ED-TCN is a supervised classifier trained
directly on the NIA pitch contours with the ornament labels (no tokenizer, 10
seeds); it is an upper reference for the classification columns only, and its
training code is not part of this repository.

---

## Reproducing the paper

### 1 · Prepare pitch-contour CSVs

Use the released datasets above, or provide your own per-clip pitch-contour
CSVs (e.g. [CREPE](https://github.com/marl/crepe) output) with at least
the columns:

- `frequency` — F0 in Hz per frame
- `confidence` — voicing confidence in [0, 1]

Frames are assumed to be 10 ms (100 Hz). CSVs are discovered recursively
(`rglob`) under `data.data_pth`, so any directory nesting works.
Frames with `confidence < 0.5` are masked/interpolated (see
`src/configs/dataset/full_threshold.yaml`).

### 2 · Train

```bash
python src/train.py --config-name foffset_vqvae data.data_pth=/path/to/csvs
```

- CLI dotted overrides take precedence over the YAML files; see
  `src/configs/` for the full set of knobs.
- Split: the default `split=manifest` is the artist-level partition of the
  released corpus (`splits/inhouse_crepe_split_artist.csv`, columns
  `path,artist,split`; 1,349 / 331 / 231 clips, 3 non-singing clips marked
  `exclude`). Every file under `data.data_pth` must be listed, so for your
  own data use `split=random` (fixed 8:1:1 shuffle via
  `split.params.split_seed`, independent of `random_seed`). `split=manifest_v1`
  is the random file-level split of the v1.0.0 checkpoints.
- WandB is disabled by default. Enable with `wandb.project=<name>`
  (requires login).
- Checkpoints + resolved `config.yaml` land in `weights/<run_name>/`.
- VQ codebooks are initialized with k-means over encoder latents
  (`model.params.init=kmeans`).

### 3 · Post-training experiments

Add `exp.run=true` to run the paper's analyses after training, or run them
for any saved checkpoint without training (no wandb account needed):

```bash
python src/run_exp.py --run_dir weights/temporal_align_vqvae            # → weights/temporal_align_vqvae/exp_results.json
python src/run_exp.py --run_dir weights/<your_run> --split random --batch_size 256
```

The analyses are:

- **KLD & Code-Flip** (token consistency) — run on the held-out test
  split of *your* data; no extra assets needed. For the autoencoder,
  the post-hoc token space (scaler/UMAP/k-means) is fit on the train
  split and applied to the test split; saved `token_*.joblib` files in the
  run directory (as shipped with the released autoencoder) are reused.
- **NIA sigimsae classification** — requires the NIA Gugak dataset
  (available via [AI-Hub](https://aihub.or.kr), Korea) plus a metadata
  CSV with columns `filename,onset,offset,label,split`, where `label`
  is one of `southern_flick`, `southern_vibrato`, `western_vibrato`,
  `soft_vibrato`, `upward_accent`, `descending_slide`.
  Fill in the `nia:`/`nia_vocal:` blocks in `src/configs/exp/default.yaml`
  (a commented template is provided there; `exp=nia_pair` is the released
  layout). When unset, these stages are skipped.

### 4 · Inference

```bash
python src/infer.py --run_dir weights/all_transformation_vqvae --input /path/to/contour.csv --out out/
```

Works with your own training runs (`weights/<run_name>`) or the released
checkpoints. Writes `reconstructions.csv`
(`window, frame, target_midi, recon_midi`) and target-vs-reconstruction plots.

### Reproducibility notes

- **Released checkpoints → reported numbers.** `src/run_exp.py` on a released
  VQ-family checkpoint with the released data and the default manifest
  reproduces its row of `results/seed_results.csv` exactly: tokens are
  integer code ids, so KLD, code-flip and the NIA metrics do not depend on
  batch size or file order. For the autoencoder the shipped
  `token_*.joblib` (StandardScaler → UMAP → k-means, fit on train-split
  latents) is reused when all of its files are present; delete them to
  refit. With the shipped files, seed 42 reproduces its row exactly and the
  other autoencoder seeds agree to within 1e-4 on KLD / code-flip (F1 / mAP
  exact). A refit changes the AE row (UMAP is not bit-stable across runs and
  numeric stacks).
- **Retraining is bit-exact for a fixed seed on one machine.** The
  codebook's k-means initialisation runs single-threaded
  (`threadpoolctl`), the training-window sampler and the data loader are
  seeded, and `cudnn.deterministic` is on. Repeating a run with the same
  seed on the same GPU model reproduced every checkpoint and every metric
  bit-for-bit. Across GPU models or library versions, expect tiny
  numerical differences that grow over 50k iterations.
- **What changed from v1.0.0.** The v2 checkpoints use an artist-level
  split (`split=manifest`), tempo candidates
  `[50, 52, 55, 58, 60]` (v1: `[50, 54, 60]`) and zoom candidates
  `[0.9, 0.95, 1.0, 1.05, 1.1]` (v1: `[0.85, 1.0, 1.15]`), and the dataset
  returns a boolean mask of interpolated frames in place of the raw
  confidence weights. v1 checkpoints stay downloadable from the v1.0.0
  release and load with the v1.0.0 tag of this repository.
- Released checkpoints were trained with Python 3.10, PyTorch 2.8.0
  (CUDA 12.8), scikit-learn 1.6.1, umap-learn 0.5.11 (numba 0.61.2,
  pynndescent 0.6.0) on an NVIDIA RTX 4090 (`OMP_NUM_THREADS=8`); the same
  stack is pinned in `pyproject.toml`/`uv.lock`.

---

## Directory layout

```
src/
    train.py                  # config-driven training entry point
    trainers.py               # AETrainer / VQTrainer loops
    experiments.py            # post-training analyses (KLD, code-flip, NIA)
    run_exp.py                # run the analyses for a saved checkpoint (no training, no wandb)
    infer.py                  # checkpoint inference on a contour CSV
    models/
        model_zoo.py          # Conv1DAE, Conv1DVQVAE, OffsetConv1DVQVAE
        modules.py            # encoder / decoder / vector-quantizer blocks
    losses/loss_zoo.py        # MSELoss, OffsetMSELoss, VectorQuantizeLoss
    datasets/dataset_zoo.py   # contour-CSV datasets (FullPitch*, NIAFixedCls)
    metrics/metric_zoo.py     # Perplexity
    schedulers/scheduler_zoo.py  # warmup + cosine LR schedule
    utils/                    # data split, k-means init, inference & analysis helpers
    configs/                  # Hydra configs (one YAML per model preset)

splits/                       # train/valid/test manifests for the released corpus
    inhouse_crepe_split_artist.csv   # v2 (default): artist-level, path,artist,split
    inhouse_crepe_split.csv          # v1.0.0: random 8:1:1
scripts/make_manifest.py      # how the manifests are built (content-hash match to the research corpus)
results/                      # per-seed metrics, 10-seed summary, LaTeX table, PROVENANCE.md

weights/                      # pretrained checkpoints (from Releases; gitignored)
    autoencoder/              # best_model.pt + config.yaml (+ token_*.joblib) per model, seed 42
    vqvae/
    temporal_align_vqvae/
    all_transformation_vqvae/
    seeds/<model>/s<seed>/    # all 10 seeds (checkpoints_seeds.tar.gz; autoencoder seeds include token_*.joblib)
    PROVENANCE.md

data/                         # datasets (from Releases; gitignored)
    inhouse_data/             # pansori training corpus (CREPE contours)
    nia_gugak_data/           # NIA Gugak contours
    pansori_annotated_data/   # mode-annotated pansori contours
    meta/                     # NIA & pansori metadata CSVs

pyproject.toml + uv.lock      # locked environment (uv)
requirements.txt              # pip fallback
```

---

## Citation

```bibtex
@inproceedings{ju2026pitchcontour,
  title         = {Pitch Contour Tokenization using {VQ-VAE} and Its Application on Korean Traditional Music Analysis},
  author        = {Ju, Seonguk and Cho, Seola and Chung, Suin and Han, Danbinaerin and Jeong, Dasaem},
  booktitle     = {Proceedings of the 27th International Society for Music Information Retrieval Conference (ISMIR)},
  year          = {2026},
  eprint        = {2608.10979},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD}
}
```

## License

[MIT](LICENSE).
