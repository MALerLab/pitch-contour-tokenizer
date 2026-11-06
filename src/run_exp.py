"""Run the post-training experiments (KLD, code-flip accuracy, NIA classification) for a saved
checkpoint without training and without a wandb account.

    python src/run_exp.py --run_dir weights/temporal_align_vqvae            # released checkpoint
    python src/run_exp.py --run_dir weights/<your_run> --split random        # your own run / data
    python src/run_exp.py --run_dir weights/vqvae --skip nia/map nia/vocal/map --force

Results are written to <run_dir>/exp_results.json (metric name -> value); completed experiments are
tracked in <run_dir>/.exp_state_n<k>.json, so re-running only computes what is missing unless --force.
For the autoencoder the token space (scaler -> UMAP -> k-means) saved in run_dir (token_*.joblib, e.g. the
released files) is reused; when any of these files is missing it is refit on the train split and saved.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments import run_exp, run_exp_ae
from models import model_zoo
from utils import data_utils, exp_utils
from utils.train_utils import set_seed

AE_MODELS = ['Conv1DAE']
VQVAE_MODELS = ['Conv1DVQVAE', 'OffsetConv1DVQVAE']


class _Summary:
    """Minimal stand-in for wandb's run.summary: keeps numeric values, drops images/tables."""
    def __init__(self):
        self.values = {}

    def update(self, d):
        for k, v in d.items():
            if isinstance(v, (bool, int, float, np.integer, np.floating)):
                self.values[k] = float(v)


class _Recorder:
    def __init__(self):
        self.summary = _Summary()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_dir', required=True, help='directory with config.yaml and the checkpoint')
    p.add_argument('--ckpt', default='best_model.pt')
    p.add_argument('--data_pth', default=None, help='override data.data_pth (default: from config.yaml)')
    p.add_argument('--split', default=None, help='split config name (manifest | manifest_v1 | random); default: from config.yaml')
    p.add_argument('--batch_size', type=int, default=None, help='inference batch size (default: train.batch_size in config)')
    p.add_argument('--skip', nargs='+', default=[], help='experiment keys to skip (kld code_flip nia/map nia/vocal/map)')
    p.add_argument('--force', action='store_true', help='recompute experiments already marked done')
    p.add_argument('--n_codes', type=int, default=None, help='autoencoder only: token-space size (default: exp.n_codes_list[0] or 256)')
    p.add_argument('--out', default=None, help='results JSON path (default: <run_dir>/exp_results.json)')
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    cfg = OmegaConf.load(run_dir / 'config.yaml')
    if args.data_pth:
        cfg.data.data_pth = args.data_pth
    if args.split:
        cfg.split = OmegaConf.load(Path(__file__).resolve().parent / 'configs' / 'split' / f'{args.split}.yaml')
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
        exp_utils.VQTokens.chunk = args.batch_size
    if not cfg.get('exp'):
        raise SystemExit('config.yaml has no exp block (NIA paths); see src/configs/exp/nia_pair.yaml')
    OmegaConf.update(cfg, 'exp.skip', list(args.skip), merge=False)
    OmegaConf.update(cfg, 'exp.force', bool(args.force), merge=False)
    n_codes = args.n_codes or (list(cfg.exp.get('n_codes_list', [256]))[0])
    OmegaConf.update(cfg, 'exp.n_codes_list', [int(n_codes)], merge=False)   # single value -> no extra runs spawned

    set_seed(cfg.random_seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = getattr(model_zoo, cfg.model.name)(cfg.model.params)
    model.load_state_dict(torch.load(run_dir / args.ckpt, map_location='cpu'))
    model = model.to(device).eval()
    if cfg.model.name in VQVAE_MODELS and 'VQ' not in model.__class__.__name__:
        raise SystemExit(f'unexpected model class {model.__class__.__name__}')

    maker = getattr(data_utils, cfg.train.get('dataset_maker', 'BaseDatasetMaker'))(cfg)
    trainset, validset, testset = maker.get_datasets()
    print(f'dataset windows: train {len(trainset)}, valid {len(validset)}, test {len(testset)}')

    rec = _Recorder()
    if cfg.model.name in AE_MODELS:
        run_exp_ae(cfg, model, trainset, testset, run_dir, run=rec, device=device)
    elif cfg.model.name in VQVAE_MODELS:
        run_exp(cfg, model, testset, run_dir, run=rec)
    else:
        raise SystemExit(f'unknown model {cfg.model.name}')

    out = Path(args.out) if args.out else run_dir / 'exp_results.json'
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev.update(rec.summary.values)
    out.write_text(json.dumps(prev, indent=2, sort_keys=True))
    print(f'\n{len(rec.summary.values)} metrics written -> {out}')
    for k in sorted(rec.summary.values):
        print(f'  {k}: {rec.summary.values[k]:.4f}')


if __name__ == '__main__':
    main()
