#!/bin/bash
# Build the v2 checkpoint archives + checksums into release_assets/ (run from the repo root after verification).
#   checkpoints.tar.gz        weights/{autoencoder,vqvae,temporal_align_vqvae,all_transformation_vqvae}/ + weights/PROVENANCE.md
#   checkpoints_seeds.tar.gz  weights/seeds/<model>/s<seed>/ (10 seeds x 4 models; autoencoder seeds incl. token_*.joblib)
set -e
cd "$(dirname "$0")/.."
mkdir -p release_assets
rm -f release_assets/checkpoints.tar.gz release_assets/checkpoints_seeds.tar.gz release_assets/SHA256SUMS.txt
tar czf release_assets/checkpoints.tar.gz --exclude='wandb' --exclude='.exp_state*' \
    weights/PROVENANCE.md weights/autoencoder weights/vqvae weights/temporal_align_vqvae weights/all_transformation_vqvae
tar czf release_assets/checkpoints_seeds.tar.gz weights/seeds
(cd release_assets && sha256sum checkpoints.tar.gz checkpoints_seeds.tar.gz > SHA256SUMS.txt)
ls -la release_assets/checkpoints.tar.gz release_assets/checkpoints_seeds.tar.gz; cat release_assets/SHA256SUMS.txt
echo; echo "contents (checkpoints.tar.gz):"; tar tzf release_assets/checkpoints.tar.gz | sed 's|/[^/]*$||' | sort -u
