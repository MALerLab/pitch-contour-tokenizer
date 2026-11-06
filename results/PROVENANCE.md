# Provenance of the released checkpoints (v2.0.0)

All checkpoints were trained in the (private) research repository on the research corpus that is byte-identical to the released `inhouse_crepe_separated` (1,911 clips used; the 3 non-singing clips listed with `split=exclude` in `splits/inhouse_crepe_split_artist.csv` were held out). File names differ (Korean vs. romanised), which is why the split ships as a manifest matched by content hash (`scripts/make_manifest.py`).

## Common training setup

| item | value |
|---|---|
| dataset | `FullPitchThresholdDataset`, window 128 frames (10 ms), `confidence >= 0.5`, `num_interpolate 2`; no region filtering |
| split | artist-level (`splits/inhouse_crepe_split_artist.csv`): 1,349 / 331 / 231 clips train / valid / test |
| optimisation | AdamW lr 1e-3, cosine schedule (3,000 warm-up), 50,000 iterations, batch 1024, `best_model.pt` = lowest validation reconstruction loss |
| codebook | 256 codes, k-means initialisation on train-split latents (single-threaded), beta 2.0 |
| offsets | temporal alignment: `tempo_sr [50, 52, 55, 58, 60]`; all transformations: + `zoom_ratio [0.9, 0.95, 1.0, 1.05, 1.1]`, `y_offset [-0.1, 0.0, 0.1]` (octave) |
| seeds | 42 (default directories) and 0, 1, 2, 3, 4, 43, 44, 45, 46 (`weights/seeds/`); the seed sets parameter init, k-means init, training-window offsets and batch order; the split is fixed |
| hardware / determinism | one NVIDIA RTX 4090 per run, `OMP_NUM_THREADS=8`, `cudnn.deterministic=True`; repeated runs with the same seed were verified bit-identical (checkpoints and all metrics) |
| evaluation | KLD / code-flip accuracy over frame offsets 4–60 and 4–124 on the test split (code-flip: shifts of 64 frames or more are compared with the following token); NIA ornament classification (macro-F1, mAP) on the NIA gugak test set and its vocal subset; autoencoder token space: StandardScaler → UMAP(5) → k-means(256) fit on train-split latents (`token_*.joblib`) |
| research code | commits `cb0cc3e` (seeds 0–2 and 42, plus the autoencoder seed-3 run) and `99a94af` (the remaining seed 3 runs and seeds 4, 43–46) of the research repository; the two differ only in code paths not used by these models |
| dates | 2026-09-02 – 2026-09-06 |

## Checkpoints

| directory | paper method | config | seed | source run | commit | nia_f1 | vocal_f1 | nia_mAP | vocal_mAP | kld_4-60 | kld_4-124 | flip_4-60 | flip_4-124 | perplexity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `weights/autoencoder/` | Autoencoder | `contour_ae` | 42 | `1DAE_r1_0902_160920` | `cb0cc3e` | 0.2805 | 0.3083 | 0.2682 | 0.2911 | 1.5834 | 1.9062 | 0.1388 | 0.1576 | – |
| `weights/seeds/autoencoder/s0/` | Autoencoder | `contour_ae` | 0 | `1DAE_s0_0903_125408` | `cb0cc3e` | 0.2901 | 0.3113 | 0.2686 | 0.2858 | 1.6103 | 1.9416 | 0.1345 | 0.1628 | – |
| `weights/seeds/autoencoder/s1/` | Autoencoder | `contour_ae` | 1 | `1DAE_s1_0903_154733` | `cb0cc3e` | 0.3006 | 0.2791 | 0.2681 | 0.2742 | 1.6189 | 1.9370 | 0.1365 | 0.1602 | – |
| `weights/seeds/autoencoder/s2/` | Autoencoder | `contour_ae` | 2 | `1DAE_s2_0903_184248` | `cb0cc3e` | 0.2864 | 0.3138 | 0.2686 | 0.2934 | 1.6197 | 1.9393 | 0.1371 | 0.1554 | – |
| `weights/seeds/autoencoder/s3/` | Autoencoder | `contour_ae` | 3 | `1DAE_s3_0903_214645` | `cb0cc3e` | 0.2745 | 0.3044 | 0.2650 | 0.2880 | 1.6021 | 1.9241 | 0.1389 | 0.1685 | – |
| `weights/seeds/autoencoder/s4/` | Autoencoder | `contour_ae` | 4 | `1DAE_s4_0905_100127` | `99a94af` | 0.2955 | 0.3182 | 0.2728 | 0.2967 | 1.5510 | 1.8572 | 0.1519 | 0.1845 | – |
| `weights/seeds/autoencoder/s43/` | Autoencoder | `contour_ae` | 43 | `1DAE_s43_0905_125426` | `99a94af` | 0.2818 | 0.3115 | 0.2642 | 0.2982 | 1.5857 | 1.9144 | 0.1393 | 0.1695 | – |
| `weights/seeds/autoencoder/s44/` | Autoencoder | `contour_ae` | 44 | `1DAE_s44_0905_154746` | `99a94af` | 0.2871 | 0.3076 | 0.2682 | 0.2910 | 1.6119 | 1.9356 | 0.1406 | 0.1630 | – |
| `weights/seeds/autoencoder/s45/` | Autoencoder | `contour_ae` | 45 | `1DAE_s45_0905_184057` | `99a94af` | 0.2974 | 0.3114 | 0.2708 | 0.2917 | 1.5774 | 1.8947 | 0.1497 | 0.1775 | – |
| `weights/seeds/autoencoder/s46/` | Autoencoder | `contour_ae` | 46 | `1DAE_s46_0905_213433` | `99a94af` | 0.2905 | 0.3053 | 0.2702 | 0.2863 | 1.6011 | 1.9305 | 0.1426 | 0.1667 | – |
| `weights/vqvae/` | VQ-VAE | `contour_vqvae` | 42 | `1DVQ_r1_0902_165048` | `cb0cc3e` | 0.2611 | 0.3021 | 0.2356 | 0.2959 | 0.4417 | 0.5953 | 0.6463 | 0.6808 | 16.7266 |
| `weights/seeds/vqvae/s0/` | VQ-VAE | `contour_vqvae` | 0 | `1DVQ_s0_0903_133518` | `cb0cc3e` | 0.2704 | 0.3161 | 0.2500 | 0.2985 | 0.7118 | 0.9186 | 0.4320 | 0.4795 | 40.2759 |
| `weights/seeds/vqvae/s1/` | VQ-VAE | `contour_vqvae` | 1 | `1DVQ_s1_0903_162849` | `cb0cc3e` | 0.2751 | 0.3170 | 0.2544 | 0.3036 | 0.6463 | 0.8372 | 0.4933 | 0.5374 | 34.9989 |
| `weights/seeds/vqvae/s2/` | VQ-VAE | `contour_vqvae` | 2 | `1DVQ_s2_0903_192424` | `cb0cc3e` | 0.2436 | 0.2868 | 0.2290 | 0.2843 | 0.3840 | 0.5295 | 0.6526 | 0.6900 | 11.7380 |
| `weights/seeds/vqvae/s3/` | VQ-VAE | `contour_vqvae` | 3 | `1DVQ_s3_0903_223033` | `99a94af` | 0.2688 | 0.3150 | 0.2587 | 0.3029 | 0.7016 | 0.9064 | 0.4684 | 0.5139 | 41.4532 |
| `weights/seeds/vqvae/s4/` | VQ-VAE | `contour_vqvae` | 4 | `1DVQ_s4_0905_104248` | `99a94af` | 0.2649 | 0.3159 | 0.2536 | 0.3037 | 0.7051 | 0.8995 | 0.4459 | 0.4837 | 37.9307 |
| `weights/seeds/vqvae/s43/` | VQ-VAE | `contour_vqvae` | 43 | `1DVQ_s43_0905_133539` | `99a94af` | 0.2638 | 0.3026 | 0.2517 | 0.2905 | 0.6968 | 0.9057 | 0.4545 | 0.5008 | 41.3828 |
| `weights/seeds/vqvae/s44/` | VQ-VAE | `contour_vqvae` | 44 | `1DVQ_s44_0905_162831` | `99a94af` | 0.2711 | 0.2911 | 0.2521 | 0.3115 | 0.6337 | 0.8311 | 0.5015 | 0.5389 | 30.8858 |
| `weights/seeds/vqvae/s45/` | VQ-VAE | `contour_vqvae` | 45 | `1DVQ_s45_0905_192201` | `99a94af` | 0.2667 | 0.3004 | 0.2461 | 0.2986 | 0.6209 | 0.8110 | 0.4793 | 0.5271 | 25.7910 |
| `weights/seeds/vqvae/s46/` | VQ-VAE | `contour_vqvae` | 46 | `1DVQ_s46_0905_221528` | `99a94af` | 0.2696 | 0.3022 | 0.2543 | 0.2907 | 0.7369 | 0.9473 | 0.4546 | 0.4951 | 47.4090 |
| `weights/temporal_align_vqvae/` | + temporal alignment | `xoffset_vqvae` | 42 | `1DXVQ_t60x5_0902_230426` | `cb0cc3e` | 0.3001 | 0.3111 | 0.2759 | 0.3091 | 0.5291 | 0.7246 | 0.5278 | 0.5672 | 41.4852 |
| `weights/seeds/temporal_align_vqvae/s0/` | + temporal alignment | `xoffset_vqvae` | 0 | `1DXVQ_t60x5_s0_0903_141321` | `cb0cc3e` | 0.3025 | 0.3450 | 0.2836 | 0.3251 | 0.5802 | 0.7908 | 0.4875 | 0.5293 | 53.5196 |
| `weights/seeds/temporal_align_vqvae/s1/` | + temporal alignment | `xoffset_vqvae` | 1 | `1DXVQ_t60x5_s1_0903_170734` | `cb0cc3e` | 0.3078 | 0.3508 | 0.2864 | 0.3402 | 0.6178 | 0.8316 | 0.4719 | 0.5190 | 61.6942 |
| `weights/seeds/temporal_align_vqvae/s2/` | + temporal alignment | `xoffset_vqvae` | 2 | `1DXVQ_t60x5_s2_0903_200812` | `cb0cc3e` | 0.2821 | 0.3079 | 0.2726 | 0.3214 | 0.4975 | 0.6865 | 0.5512 | 0.5946 | 36.6408 |
| `weights/seeds/temporal_align_vqvae/s3/` | + temporal alignment | `xoffset_vqvae` | 3 | `1DXVQ_t60x5_s3_0903_230856` | `99a94af` | 0.3040 | 0.3408 | 0.2719 | 0.3257 | 0.4959 | 0.6761 | 0.5388 | 0.5891 | 31.9371 |
| `weights/seeds/temporal_align_vqvae/s4/` | + temporal alignment | `xoffset_vqvae` | 4 | `1DXVQ_t60x5_s4_0905_112051` | `99a94af` | 0.3071 | 0.3232 | 0.2901 | 0.3265 | 0.5874 | 0.8110 | 0.4846 | 0.5246 | 54.0543 |
| `weights/seeds/temporal_align_vqvae/s43/` | + temporal alignment | `xoffset_vqvae` | 43 | `1DXVQ_t60x5_s43_0905_141341` | `99a94af` | 0.2986 | 0.3236 | 0.2705 | 0.3259 | 0.4542 | 0.6281 | 0.5870 | 0.6241 | 31.7706 |
| `weights/seeds/temporal_align_vqvae/s44/` | + temporal alignment | `xoffset_vqvae` | 44 | `1DXVQ_t60x5_s44_0905_170644` | `99a94af` | 0.2769 | 0.3072 | 0.2747 | 0.3174 | 0.5504 | 0.7490 | 0.5311 | 0.5787 | 45.6526 |
| `weights/seeds/temporal_align_vqvae/s45/` | + temporal alignment | `xoffset_vqvae` | 45 | `1DXVQ_t60x5_s45_0905_195957` | `99a94af` | 0.2974 | 0.3252 | 0.2809 | 0.3249 | 0.6007 | 0.8074 | 0.4905 | 0.5295 | 51.2200 |
| `weights/seeds/temporal_align_vqvae/s46/` | + temporal alignment | `xoffset_vqvae` | 46 | `1DXVQ_t60x5_s46_0905_225332` | `99a94af` | 0.3019 | 0.3145 | 0.2732 | 0.3072 | 0.5194 | 0.7175 | 0.5277 | 0.5653 | 39.1575 |
| `weights/all_transformation_vqvae/` | + all transformations | `foffset_vqvae` | 42 | `1DFVQ_t60x5_z10x5_y0.1x3_0903_033933` | `cb0cc3e` | 0.3145 | 0.3220 | 0.2897 | 0.3267 | 0.6037 | 0.8309 | 0.4833 | 0.5325 | 58.4710 |
| `weights/seeds/all_transformation_vqvae/s0/` | + all transformations | `foffset_vqvae` | 0 | `1DFVQ_t60x5_z10x5_y0.1x3_s0_0903_145558` | `cb0cc3e` | 0.2789 | 0.3180 | 0.2666 | 0.3161 | 0.5323 | 0.7392 | 0.5313 | 0.5666 | 38.4297 |
| `weights/seeds/all_transformation_vqvae/s1/` | + all transformations | `foffset_vqvae` | 1 | `1DFVQ_t60x5_z10x5_y0.1x3_s1_0903_175054` | `cb0cc3e` | 0.3332 | 0.3529 | 0.2971 | 0.3355 | 0.6121 | 0.8304 | 0.4728 | 0.5192 | 61.7883 |
| `weights/seeds/all_transformation_vqvae/s2/` | + all transformations | `foffset_vqvae` | 2 | `1DFVQ_t60x5_z10x5_y0.1x3_s2_0903_205512` | `cb0cc3e` | 0.3013 | 0.3383 | 0.2813 | 0.3264 | 0.5488 | 0.7651 | 0.5126 | 0.5398 | 44.9030 |
| `weights/seeds/all_transformation_vqvae/s3/` | + all transformations | `foffset_vqvae` | 3 | `1DFVQ_t60x5_z10x5_y0.1x3_s3_0903_235123` | `99a94af` | 0.2975 | 0.3237 | 0.2753 | 0.3113 | 0.4989 | 0.6763 | 0.5439 | 0.6003 | 38.8722 |
| `weights/seeds/all_transformation_vqvae/s4/` | + all transformations | `foffset_vqvae` | 4 | `1DFVQ_t60x5_z10x5_y0.1x3_s4_0905_120311` | `99a94af` | 0.2990 | 0.3208 | 0.2798 | 0.3200 | 0.5532 | 0.7504 | 0.5144 | 0.5454 | 46.8990 |
| `weights/seeds/all_transformation_vqvae/s43/` | + all transformations | `foffset_vqvae` | 43 | `1DFVQ_t60x5_z10x5_y0.1x3_s43_0905_145606` | `99a94af` | 0.3069 | 0.3202 | 0.2853 | 0.3281 | 0.5944 | 0.8113 | 0.4881 | 0.5290 | 57.1408 |
| `weights/seeds/all_transformation_vqvae/s44/` | + all transformations | `foffset_vqvae` | 44 | `1DFVQ_t60x5_z10x5_y0.1x3_s44_0905_174923` | `99a94af` | 0.3007 | 0.3442 | 0.2871 | 0.3484 | 0.6172 | 0.8346 | 0.4642 | 0.5069 | 59.0056 |
| `weights/seeds/all_transformation_vqvae/s45/` | + all transformations | `foffset_vqvae` | 45 | `1DFVQ_t60x5_z10x5_y0.1x3_s45_0905_204245` | `99a94af` | 0.2953 | 0.3329 | 0.2785 | 0.3092 | 0.6275 | 0.8618 | 0.4513 | 0.4990 | 60.8720 |
| `weights/seeds/all_transformation_vqvae/s46/` | + all transformations | `foffset_vqvae` | 46 | `1DFVQ_t60x5_z10x5_y0.1x3_s46_0905_233604` | `99a94af` | 0.3203 | 0.3330 | 0.2879 | 0.3325 | 0.6318 | 0.8487 | 0.4729 | 0.5236 | 55.9393 |

## Seed statistics (mean ± sample std over the 10 seeds)

| config | n | seeds | nia_f1 | vocal_f1 | nia_mAP | vocal_mAP | kld_4-60 | kld_4-124 | flip_4-60 | flip_4-124 | perplexity |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1DAE | 10 | 0,1,2,3,4,42,43,44,45,46 | 0.2885 ± 0.0081 | 0.3071 ± 0.0107 | 0.2685 ± 0.0025 | 0.2896 ± 0.0068 | 1.5961 ± 0.0217 | 1.9181 ± 0.0264 | 0.1410 ± 0.0056 | 0.1666 ± 0.0090 | – |
| 1DFVQ_t60x5_z10x5_y0.1x3 | 10 | 0,1,2,3,4,42,43,44,45,46 | 0.3048 ± 0.0150 | 0.3306 ± 0.0117 | 0.2829 ± 0.0085 | 0.3254 ± 0.0119 | 0.5820 ± 0.0455 | 0.7949 ± 0.0594 | 0.4935 ± 0.0305 | 0.5362 ± 0.0296 | 52.2321 ± 9.0662 |
| 1DVQ | 10 | 0,1,2,3,4,42,43,44,45,46 | 0.2655 ± 0.0087 | 0.3049 ± 0.0108 | 0.2485 ± 0.0093 | 0.2980 ± 0.0080 | 0.6279 ± 0.1200 | 0.8181 ± 0.1424 | 0.5028 ± 0.0801 | 0.5447 ± 0.0770 | 32.8592 ± 11.5649 |
| 1DXVQ_t60x5 | 10 | 0,1,2,3,4,42,43,44,45,46 | 0.2979 ± 0.0103 | 0.3249 ± 0.0158 | 0.2780 ± 0.0068 | 0.3223 ± 0.0094 | 0.5433 ± 0.0530 | 0.7423 ± 0.0673 | 0.5198 ± 0.0358 | 0.5621 ± 0.0355 | 44.7132 ± 10.1848 |
| ED-TCN (supervised reference) | 10 | 0,1,2,3,4,42,43,44,45,46 | 0.5799 ± 0.0175 | 0.5337 ± 0.0142 | 0.6224 ± 0.0157 | 0.5249 ± 0.0122 | – | – | – | – | – |

`src/run_exp.py --run_dir weights/<model>` recomputes the per-seed values above from a checkpoint (the VQ-family metrics are exactly reproducible; the autoencoder metrics depend on the shipped `token_*.joblib`).

## Verification (2026-11-05)

`src/run_exp.py --run_dir <copy of weights/<model>> --batch_size 256` on all four released seed-42 checkpoints, with the released corpus (`data=crepe`, default artist manifest) and the pinned environment (`uv sync`) reproduced all eight metrics of the seed-42 rows above (max abs. difference 6e-8; token-derived metrics are exact). The autoencoder was evaluated with the shipped `token_*.joblib` (reused by `run_exp_ae` when present; umap-learn 0.5.11, numba 0.61.2, pynndescent 0.6.0 — the UMAP pickle is numba-bound and requires these versions). All 44 `best_model.pt` files and the `token_scaler`/`token_kmeans` joblibs of all 10 autoencoder seeds are byte-identical to the research runs listed above; in `token_umap.joblib` only the site-packages source paths that numba embeds in the pickle were overwritten with a same-length neutral path (the UMAP transform output is unchanged). Re-running the autoencoder seed 0 reproduces its F1/mAP exactly and its KLD / code-flip values to within 1e-4.
