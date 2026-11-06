# experiments.py 전용 헬퍼 — infer_utils에서 분리.
# VQ/AE 중복(_ae 쌍)은 tokenizer 주입으로 통합:
#   VQTokens — extract_min_e argmax가 곧 코드 id (배치/segment 단위 즉시 인코딩)
#   AETokens — latent → scaler→umap→kmeans. UMAP transform은 배치 구성에 따라 결과가 달라지므로
#              기존 구현의 수집 단위를 그대로 유지 (NIA cnt는 배치별, proba/segment는 전체 수집 후 일괄 변환)
from utils.infer_utils import extract_min_e

import torch
from torch.utils.data import DataLoader

import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, average_precision_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt


class VQTokens:
    batch_size = 256   # NIA DataLoader
    file_tag = ''      # 저장 파일명 suffix (code_flip{file_tag}{tag}.npy 등)
    chunk = 1024       # pad segment는 분 단위 길이가 가능하므로 chunk 순회 (예제별 독립 forward라 결과는 단일 forward와 동일)
                       # grid 계열은 to_grid (b,1,F,rf) → 2D conv activation이 커서 8192면 OOM (max_pool2d 한 층 4.5GiB)

    def encode_batch(self, model, x):
        # 배치 → 코드 id (np). x는 caller가 device로 옮겨서 전달
        return extract_min_e(model, x).argmax(dim=-1).detach().cpu().numpy()

    # _predict_codes용: 배치 출력이 곧 코드 id, finalize는 no-op
    def batch_out(self, model, x):
        return self.encode_batch(model, x)

    def finalize(self, outs):
        return outs

    def encode_segments(self, model, segments, total, device):
        # segment별 (offset_cnt, n_windows) 코드 id를 순서대로 yield
        model.eval()
        for res in tqdm(segments, total=total):
            x = res.reshape(-1, model.rf)
            ids = []
            with torch.inference_mode():
                for i in range(0, x.shape[0], self.chunk):
                    ids.append(extract_min_e(model, x[i:i+self.chunk].to(device)).argmax(dim=-1).detach().cpu())
            yield torch.cat(ids).reshape(res.shape[0], -1)


class AETokens:
    batch_size = 1024

    def __init__(self, scaler, umap_embed, kmeans):
        self.scaler, self.umap_embed, self.kmeans = scaler, umap_embed, kmeans
        self.file_tag = f'_ae_{kmeans.n_clusters}'

    def predict(self, z):
        return self.kmeans.predict(self.umap_embed.transform(self.scaler.transform(z)))

    def _z(self, model, x):
        _, _, z = model(x)
        return z.squeeze().detach().cpu().numpy()

    def encode_batch(self, model, x):
        return self.predict(self._z(model, x))

    # _predict_codes용: 배치는 latent만 모으고 finalize에서 일괄 변환
    def batch_out(self, model, x):
        return self._z(model, x)

    def finalize(self, outs):
        return self.predict(outs)

    def encode_segments(self, model, segments, total, device):
        all_z, shapes = [], []
        model.eval()
        with torch.inference_mode():
            for res in tqdm(segments, total=total, desc='Extract latent'):
                b, s, t = res.shape
                all_z.append(self._z(model, res.reshape(-1, t).to(device)))
                shapes.append((b, s))
        code_ids = self.predict(np.concatenate(all_z, axis=0))
        cursor = 0
        for b, s in shapes:
            yield torch.tensor(code_ids[cursor:cursor + b*s]).reshape(b, s)
            cursor += b*s


def _log(log_dict, run=None):
    # summary.update: 값만 기록 (run.log와 달리 step을 만들지/증가시키지 않음)
    if run is not None:
        run.summary.update(log_dict)
    elif wandb.run is not None:
        wandb.run.summary.update(log_dict)
    else:
        print("[WANDB] No active run, skipping log.")


# Exp Code Flip
def get_segment_meta(dataset, dur=None):
    if dur is None:
        dur = dataset.window * 3

    data_ids, start_ids, end_ids, durations = [], [], [], []
    for idx, (x, _, is_filtered_full) in tqdm(enumerate(dataset.loaded_data)):
        if x.shape[-1] < dur:
            continue
        mask = (~is_filtered_full).int()
        diff = torch.diff(mask, prepend=mask[:1])
        starts = (diff == 1).nonzero(as_tuple=True)[0]
        ends = (diff == -1).nonzero(as_tuple=True)[0]

        if mask[0] == 1:
            starts = torch.cat((torch.tensor([0], device=mask.device), starts))
        if mask[-1] == 1:
            ends = torch.cat((ends, torch.tensor([mask.shape[0]], device=mask.device)))

        valid_ids = torch.where((ends - starts) >= dur)[0]
        starts, ends = starts[valid_ids], ends[valid_ids]
        data_ids.extend([idx] * valid_ids.shape[0])
        start_ids.extend(starts)
        end_ids.extend(ends)
        durations.extend(ends - starts)

    sorted_indices = sorted(range(len(durations)), key=lambda i: durations[i], reverse=True)
    data_ids = [data_ids[i] for i in sorted_indices]
    start_ids = [start_ids[i] for i in sorted_indices]
    end_ids = [end_ids[i] for i in sorted_indices]
    durations = [durations[i] for i in sorted_indices]

    return data_ids, start_ids, end_ids, durations


def get_segments(dataset, data_ids, start_ids, end_ids, idx, offset, end_offset=None, contours=None):
    offset_cnt = dataset.window // offset if end_offset is None else end_offset // offset + 1
    start_idx, end_idx = start_ids[idx], end_ids[idx]
    valid_len = end_idx - start_idx - dataset.window + offset if end_offset is None else end_idx - start_idx - end_offset
    segment_len = valid_len // dataset.window * dataset.window
    required_len = segment_len + (offset_cnt - 1) * offset
    x = contours[data_ids[idx]] if contours is not None else dataset.loaded_data[data_ids[idx]][0]
    x = x[start_idx:start_idx + required_len]
    segments = torch.stack([x[i * offset:i * offset + segment_len].reshape(-1, dataset.window) for i in range(offset_cnt)])
    return segments


def get_acc(inference_output, offset, window=128):
    flip_idx = window // 2 // offset   # shifts >= window/2 (64 frames) are compared with the following token
    gt = inference_output[0]
    if flip_idx > inference_output.shape[0]:
        acc = (gt == inference_output[1:]).float().mean(dim=-1)
    else:
        acc = torch.cat([
            (gt == inference_output[1:flip_idx]).float().mean(dim=-1),
            (gt[1:] == inference_output[flip_idx:, :-1]).float().mean(dim=-1)
        ])
    return acc


def get_total_avg_acc(model, tokenizer, dataset, data_ids, start_ids, end_ids, offset, end_offset,
                      device='cuda', prefix='test', run=None, pth=None, contours=None, tag=''):
    segments = (get_segments(dataset, data_ids, start_ids, end_ids, idx, offset, end_offset, contours=contours)
                for idx in range(len(data_ids)))
    total_avg_acc = np.array([get_acc(ids, offset).mean()
                              for ids in tokenizer.encode_segments(model, segments, len(data_ids), device)])

    if pth is not None:
        np.save(f"{pth}/code_flip{tokenizer.file_tag}{tag}.npy", total_avg_acc)

    _log({f'{prefix}/code_flip_acc/{offset}-{end_offset}': np.mean(total_avg_acc)}, run)
    return total_avg_acc


# EXP KLD
def _codes_to_df(code_ids, n_codes, offset):
    # 행=offset, 열=코드 id별 빈도. 열은 총 빈도 내림차순 정렬 (기존 get_df(used_df=False)와 동일)
    keys = [offset * i for i in range(code_ids.shape[0])]
    df = pd.DataFrame(0, columns=range(n_codes), index=keys)
    df.index.name = 'offset'

    for i, k in enumerate(keys):
        df.loc[k] = torch.bincount(code_ids[i], minlength=n_codes).numpy()

    return df[df.sum(axis=0).sort_values(ascending=False).index]


def kld_discrete(p, q, epsilon=1e-12):
    q, p = q + epsilon, p + epsilon
    q, p = q / q.sum(), p / p.sum()
    return (p * torch.log(p / q)).sum()


def compute_baseline_distribution_from_df(df):
    baseline = torch.tensor(df.sum(axis=0).values, dtype=torch.float32)
    return baseline / baseline.sum()


def kld_from_baseline_df(df):
    baseline = compute_baseline_distribution_from_df(df)
    kld_dict = {}
    for offset_idx in sorted(df.index):
        p = torch.tensor(df.loc[offset_idx].values, dtype=torch.float32)
        p = p / p.sum()
        kld_dict[offset_idx] = kld_discrete(p, baseline).item()
    return kld_dict, baseline


def get_mean_kld(kld_dict):
    kld_values = [kld_dict[o] for o in sorted(kld_dict.keys())]
    return np.mean(kld_values)


def get_total_mean_klds(model, tokenizer, dataset, data_ids, start_ids, end_ids, n_codes, offset, end_offset,
                        device='cuda', prefix='test', run=None, pth=None, contours=None, tag=''):
    segments = (get_segments(dataset, data_ids, start_ids, end_ids, i, offset, end_offset, contours=contours)
                for i in range(len(data_ids)))
    mean_klds = {}
    for i, ids in enumerate(tokenizer.encode_segments(model, segments, len(data_ids), device)):
        kld_dict, _ = kld_from_baseline_df(_codes_to_df(ids, n_codes, offset))
        mean_klds[i] = get_mean_kld(kld_dict)

    if pth is not None:
        np.save(f"{pth}/kld{tokenizer.file_tag}{tag}.npy", mean_klds)

    _log({f'{prefix}/kld/{offset}-{end_offset}': np.mean(sorted(mean_klds.values()))}, run)

    return mean_klds


# EXP NIA
def compute_label_token_cnt(trainset, model, tokenizer, num_labels, n_codes, pth=None, prefix='', device='cuda'):
    label_token_cnt = np.zeros((num_labels, n_codes))
    dl = DataLoader(trainset, batch_size=tokenizer.batch_size, shuffle=False, drop_last=False)

    model.eval()
    with torch.inference_mode():
        for x, _, y in tqdm(dl):
            code_idx = tokenizer.encode_batch(model, x.to(device))
            np.add.at(label_token_cnt, (y.numpy(), code_idx), 1)

    P_token_given_label = label_token_cnt / label_token_cnt.sum(axis=-1, keepdims=True)
    P_label = label_token_cnt.sum(axis=-1) / label_token_cnt.sum()

    if pth is not None:
        suffix = ('_pad' if 'PAD' in prefix else '') + ('_vocal' if 'VOCAL' in prefix else '')
        torch.save(label_token_cnt, f'{pth}/label_token_cnt{tokenizer.file_tag}{suffix}.pt')

    return label_token_cnt, P_token_given_label, P_label


def _predict_codes(dataset, model, tokenizer, device):
    dl = DataLoader(dataset, batch_size=tokenizer.batch_size, shuffle=False, drop_last=False)
    y_true, outs = [], []

    model.eval()
    with torch.inference_mode():
        for x, _, y in tqdm(dl):
            y_true.append(y.numpy())
            outs.append(tokenizer.batch_out(model, x.to(device)))

    return np.concatenate(y_true), tokenizer.finalize(np.concatenate(outs))


def _map_log_prob(code_idx, P_token_given_label, P_label, temperature):
    log_prior = np.log(P_label + 1e-12)
    log_likelihood = np.log(P_token_given_label + 1e-12)
    return temperature * log_prior.reshape(-1, 1) + log_likelihood[:, code_idx]
    # return temperature * log_prior.reshape(-1,1) + (1 - temperature) * log_likelihood[:,code_idx]


def get_proba(dataset, model, tokenizer, P_token_given_label, P_label, temperature=1.0, device='cuda'):
    y_true, code_idx = _predict_codes(dataset, model, tokenizer, device)

    log_prob = _map_log_prob(code_idx, P_token_given_label, P_label, temperature)
    log_prob -= log_prob.max(axis=0, keepdims=True)
    prob = np.exp(log_prob)
    prob /= prob.sum(axis=0)

    return prob, y_true


def predict_by_map_with_temperature(dataset, model, tokenizer, P_token_given_label, P_label, temperature=1.0, device='cuda'):
    y_true, code_idx = _predict_codes(dataset, model, tokenizer, device)

    log_prob = _map_log_prob(code_idx, P_token_given_label, P_label, temperature)
    return log_prob.argmax(axis=0), y_true


def fit_temperature_with_log(validset, model, tokenizer, P_token_given_label, P_label,
                            #   temperatures=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
                              temperatures=np.round(np.linspace(0.00, 1.00, 50), 5),
                              metric='f1', device='cuda', run=None, prefix=None):
    best_T, best_score = 1.0, 0.0
    log_table = wandb.Table(columns=['temperature', 'score']) if (run is not None or wandb.run is not None) else None

    for T in temperatures:
        y_pred, y_true = predict_by_map_with_temperature(validset, model, tokenizer, P_token_given_label, P_label, T, device)

        # f1 = f1_score(y_true, y_pred, average='weighted')
        if metric == 'f1':
            score = f1_score(y_true, y_pred, average='macro')
        elif metric == 'acc':
            score = np.mean(np.array(y_true) == np.array(y_pred))
        print(f"T={T:.2f}: score={score:.5f}")

        if log_table is not None:
            log_table.add_data(T, score)

        if score > best_score:
            best_score, best_T = score, T

    p = f'{prefix}/' if prefix else ''
    log_dict = {f'{p}best_temperature': best_T, f'{p}best_val_score': best_score}
    if log_table is not None:
        log_dict[f'{p}temperature_search'] = log_table
    _log(log_dict, run)

    print(f"\nBest T={best_T}, best_score={best_score:.5f}")
    return best_T


def evaluate_with_log(testset, model, tokenizer, P_token_given_label, P_label, temperature,
                      num_labels, idx2label, prefix='test', device='cuda', run=None):
    y_prob, y_true = get_proba(testset, model, tokenizer, P_token_given_label, P_label, temperature, device)
    y_score = y_prob.T        # (N, 6)
    y_pred = y_score.argmax(axis=1)

    y_true_onehot = label_binarize(y_true, classes=list(range(num_labels)))
    ap_per_class = average_precision_score(y_true_onehot, y_score, average=None)
    mAP = average_precision_score(y_true_onehot, y_score, average='macro')

    report = classification_report(y_true, y_pred,
                                   target_names=list(idx2label.values()),
                                   output_dict=True, zero_division=0)
    print(classification_report(y_true, y_pred, target_names=list(idx2label.values()), zero_division=0))

    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(cm*100, display_labels=idx2label.values())
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    disp.im_.colorbar.remove()
    plt.tight_layout()

    log_dict = {
        f'{prefix}/mAP': mAP,
        f'{prefix}/f1_macro_avg': report['macro avg']['f1-score'],
        f'{prefix}/confusion_matrix': wandb.Image(fig),
    }
    for i, ap in enumerate(ap_per_class):
        print(f"{idx2label[i]}: AP={ap:.4f}")
    print(f"\nmAP: {mAP:.4f}")

    _log(log_dict, run)
    plt.close(fig)

    return y_true, y_pred, y_score
