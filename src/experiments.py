import torch
from torch.utils.data import DataLoader

import json
import wandb
import joblib
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from utils.exp_utils import (
    VQTokens, AETokens,
    get_segment_meta,
    get_total_mean_klds, get_total_avg_acc,
    compute_label_token_cnt, fit_temperature_with_log, evaluate_with_log,
)
from utils.data_utils import ClsDatasetMaker


# ---------- resume / skip helpers ----------

class ExpState:
    # skip이 최우선, 완료 항목은 force 없이는 재실행 안 함. done()은 즉시 파일에 반영
    def __init__(self, path: Path, skip, force: bool):
        self.path, self.skip, self.force = path, set(skip), force
        self.state = json.loads(path.read_text()) if path.exists() else {}

    def should(self, name: str) -> bool:
        if name in self.skip:
            return False
        return self.force or not self.state.get(name, False)

    def done(self, name: str) -> None:
        self.state[name] = True
        self.path.write_text(json.dumps(self.state, indent=2, ensure_ascii=False))


# ---------- NIA classification variants ----------

KR2EN = {'남도 꺾는청': 'nam-kk', '남도요성': 'nam-yo', '서도요성': 'seo-yo',
         '잔요성': 'jan-yo', '전성': 'jeon', '퇴성': 'toe'}


def _nia_cls_variants(cfg: DictConfig, save_dir: Path) -> list:
    # (nia, nia_vocal) two tracks, both on NIAFixedClsDataset
    variants = []
    for exp_cfg, track_state, track_log, cnt_prefix in [
            (cfg.exp.nia, '', '', ''),
            (cfg.exp.nia_vocal, '/vocal', '/VOCAL', 'VOCAL')]:
        c = OmegaConf.to_container(exp_cfg, resolve=True)
        c['dataset']['name'] = 'NIAFixedClsDataset'
        variants.append((
            OmegaConf.create(c),          # nia_cfg
            f'nia{track_state}/map',      # state_key
            f'EXP/NIA{track_log}',        # log prefix
            cnt_prefix,                   # label_token_cnt file suffix (_vocal)
        ))
    return variants


# ---------- 공통 실행 블록 (VQ/AE는 tokenizer만 다름) ----------

def _run_kld_flip(st, model, tokenizer, testset, meta_fn, n_codes, window, offset, save_dir, run):
    do_kld, do_flip = st.should('kld'), st.should('code_flip')
    if not (do_kld or do_flip):
        return

    data_ids, start_ids, end_ids, _ = meta_fn()
    for end_offset in [window//2 - offset, window - offset]:
        if do_kld:
            get_total_mean_klds(
                model=model, tokenizer=tokenizer, dataset=testset,
                data_ids=data_ids, start_ids=start_ids, end_ids=end_ids,
                n_codes=n_codes, offset=offset, end_offset=end_offset,
                prefix='EXP/KLD', pth=save_dir, run=run,
            )
        if do_flip:
            get_total_avg_acc(
                model=model, tokenizer=tokenizer, dataset=testset,
                data_ids=data_ids, start_ids=start_ids, end_ids=end_ids,
                offset=offset, end_offset=end_offset,
                prefix='EXP/CODE_FLIP', pth=save_dir, run=run,
            )
    if do_kld:
        st.done('kld')
    if do_flip:
        st.done('code_flip')


def _run_nia_cls(st, model, tokenizer, n_codes, save_dir, run, variants):
    # Classification: NIA and NIA-VOCAL tracks
    # token cnt와 temperature fit은 state 완료 여부와 무관하게 매번 수행(기존 동작), evaluate만 state로 가드
    for nia_cfg, state_key, prefix, cnt_prefix in variants:
        cls_dm = ClsDatasetMaker(nia_cfg)
        ntrainset, nvalidset, ntestset = cls_dm.get_datasets()

        label_map = ntrainset.label_map
        idx2label = {v: k for k, v in label_map.items()}
        eng_idx2label = {k: KR2EN.get(v, v) for k, v in idx2label.items()}   # released NIA metadata already uses English labels
        num_labels = len(eng_idx2label)

        label_token_cnt, P_token_given_label, P_label = compute_label_token_cnt(
            trainset=ntrainset, model=model, tokenizer=tokenizer,
            num_labels=num_labels, n_codes=n_codes,
            pth=save_dir, prefix=cnt_prefix,
        )

        best_T = fit_temperature_with_log(
            validset=nvalidset, model=model, tokenizer=tokenizer,
            P_token_given_label=P_token_given_label, P_label=P_label,
            metric='f1', run=run, prefix=prefix,
        )
        if st.should(state_key):
            evaluate_with_log(
                testset=ntestset, model=model, tokenizer=tokenizer,
                P_token_given_label=P_token_given_label, P_label=P_label,
                temperature=best_T, num_labels=num_labels, idx2label=eng_idx2label,
                prefix=f'{prefix}/MAP', run=run,
            )
            st.done(state_key)
        # evaluate_with_log(
        #     testset=ntestset, model=model, tokenizer=tokenizer,
        #     P_token_given_label=P_token_given_label, P_label=P_label_uniform,
        #     temperature=0.0, num_labels=num_labels, idx2label=eng_idx2label,
        #     prefix=f'{prefix}/MLE', run=run,
        # )


# ---------- VQVAE ----------

def run_exp(cfg: DictConfig, model, testset, save_dir: Path, run=None):
    if not cfg.get('exp', False):
        return

    model.load_state_dict(torch.load(save_dir / 'best_model.pt'))
    model.eval()

    window = cfg.dataset.params.window
    n_codes = cfg.model.params.num_codebooks
    offset = cfg.exp.offset
    tokenizer = VQTokens()
    st = ExpState(save_dir / f'.exp_state_n{n_codes}.json',
                  cfg.exp.get('skip', []), cfg.exp.get('force', False))

    # KLD & Code Flip
    _run_kld_flip(st, model, tokenizer, testset, lambda: get_segment_meta(testset),
                  n_codes, window, offset, save_dir, run)

    _run_nia_cls(st, model, tokenizer, n_codes, save_dir, run, _nia_cls_variants(cfg, save_dir))


# ---------- AE (multi n_codes via separate wandb runs) ----------

def run_exp_ae(cfg: DictConfig, model, trainset, testset, save_dir: Path, run=None, device='cuda'):
    if not cfg.get('exp', False):
        return

    model.load_state_dict(torch.load(save_dir / 'best_model.pt'))
    model.eval()

    batch_size = cfg.train.batch_size
    window = cfg.dataset.params.window
    offset = cfg.exp.offset

    # n_codes_list resolution order:
    #   1) cfg.exp.n_codes_list (explicit list)
    #   2) cfg.exp.n_codes (legacy single int, e.g. src/etc/run_exp.py)
    #   3) default [128, 256, 512]
    if 'n_codes_list' in cfg.exp:
        n_codes_list = list(cfg.exp.n_codes_list)
    elif 'n_codes' in cfg.exp:
        n_codes_list = [int(cfg.exp.n_codes)]
    else:
        n_codes_list = [128, 256, 512]
        # n_codes_list = [256]

    # spawn_new_runs: only spin up fresh wandb runs when we genuinely need multiple.
    # If caller supplied a run AND only one n_codes is requested, reuse the external run
    # (e.g. resume_exp.py --n_codes 256, or etc/run_exp.py per-iteration usage).
    spawn_new_runs = (run is None) or (len(n_codes_list) > 1)

    skip = cfg.exp.get('skip', [])
    force = cfg.exp.get('force', False)

    # Latent extraction (n_codes-independent — done once): scaler/umap/kmeans fit은 drive trainset 기준
    # is_valid=False windowing은 random start라 latent 표본이 실행마다 달라짐 → 결정적 windowing으로 고정
    # Reuse a saved token space (e.g. the released token_*.joblib) when all of its files exist; delete them to refit.
    token_files = [save_dir / 'token_scaler.joblib', save_dir / 'token_umap.joblib'] + \
                  [save_dir / f'token_kmeans_n{n}.joblib' for n in n_codes_list]
    reuse_tokens = all(p.exists() for p in token_files)
    if reuse_tokens:
        print(f'[EXP] reusing saved token space: {[p.name for p in token_files]}')
        scaler = joblib.load(token_files[0])
        umap_embed = joblib.load(token_files[1])
    else:
        if not trainset.is_valid:
            trainset.is_valid = True
            trainset.resample_receptive_field()

        dl = DataLoader(trainset, batch_size=batch_size, shuffle=False, drop_last=False)
        latent = []
        with torch.inference_mode():
            for x, weight in tqdm(dl, desc='Collect Latent (train)'):
                _, _, z = model(x.to(device))
                latent.append(z.squeeze().detach().cpu())
        latent = torch.cat(latent, dim=0).numpy()

        scaler = StandardScaler()
        latent_z = scaler.fit_transform(latent)

        umap_embed = UMAP(n_components=5, random_state=cfg.random_seed, low_memory=True, n_jobs=-1)
        latent_red = umap_embed.fit_transform(latent_z)

        # 토큰 공간 재현용 fitted pipeline 저장 (predict 경로: scaler → umap → kmeans)
        joblib.dump(scaler, save_dir / 'token_scaler.joblib')
        joblib.dump(umap_embed, save_dir / 'token_umap.joblib')

    # Segment meta / NIA classification 조합 (n_codes-independent — done once)
    meta = get_segment_meta(testset)
    cls_variants = _nia_cls_variants(cfg, save_dir)

    # Capture wandb context (only needed if we're about to finish the external run and spawn new ones)
    if spawn_new_runs:
        if run is not None:
            wandb_project = run.project
            wandb_group = run.group
            wandb_base_name = run.name
            wandb_notes = run.notes
            run.finish()
        else:
            wandb_project = cfg.wandb.get('project', 'EXP') if cfg.get('wandb', False) else 'EXP'
            wandb_group = None
            wandb_base_name = save_dir.name
            wandb_notes = None

    for n_codes in n_codes_list:
        st = ExpState(save_dir / f'.exp_state_ae_n{n_codes}.json', skip, force)

        if spawn_new_runs:
            run_name = f"{wandb_base_name}_n{n_codes}"
            new_run = wandb.init(
                project=wandb_project, name=run_name, group=wandb_group, notes=wandb_notes,
                dir=str(save_dir), mode='online',
            )
        else:
            new_run = run

        if reuse_tokens:
            kmeans = joblib.load(save_dir / f'token_kmeans_n{n_codes}.joblib')
        else:
            kmeans = KMeans(n_clusters=n_codes, random_state=cfg.random_seed)
            # multi-threaded KMeans fit is nondeterministic even with fixed random_state
            with threadpool_limits(limits=1):
                kmeans.fit(latent_red)
            joblib.dump(kmeans, save_dir / f'token_kmeans_n{n_codes}.joblib')

        tokenizer = AETokens(scaler, umap_embed, kmeans)

        # KLD & Code Flip
        _run_kld_flip(st, model, tokenizer, testset, lambda: meta,
                      n_codes, window, offset, save_dir, new_run)

        _run_nia_cls(st, model, tokenizer, n_codes, save_dir, new_run, cls_variants)

        if spawn_new_runs:
            new_run.finish()
