from models import model_zoo

import torch
import torch.nn as nn

import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt


def get_model(path: str) -> nn.Module:
    cfg = OmegaConf.load(f'{path}/config.yaml')
    ckpt = torch.load(f'{path}/best_model.pt')
    model = getattr(model_zoo, cfg.model.name)(cfg.model.params)
    model.load_state_dict(ckpt)

    return model



def get_min_offset_contour(pred_contour: torch.Tensor,
                            target_contour: torch.Tensor,
                            receptive_field: int,
                            weight: torch.Tensor | None=None,
                            return_ids: bool=False):
    b,c,r,t = pred_contour.shape
    pr = pred_contour.reshape(b,c,r,t//receptive_field,receptive_field)
    tr = target_contour.unsqueeze(-2).reshape(b,c,1,t//receptive_field,receptive_field)
    squared_errors = torch.pow(pr-tr, 2)

    if weight is not None:
        weight = weight.unsqueeze(-2).unsqueeze(-2).reshape(b,c,1,t//receptive_field,receptive_field)
        sqe = torch.sum(squared_errors * ~weight, dim=-1) / torch.sum(~weight, dim=-1)
    else:
        sqe = torch.mean(squared_errors, dim=-1)

    min_ids = torch.argmin(sqe, dim=-2).unsqueeze(-2).unsqueeze(-1).expand(-1,-1,-1,-1,receptive_field)
    pred_contour = pr.gather(dim=-3, index=min_ids).squeeze().reshape(b,c,-1)
    selected_ids = min_ids[...,0].squeeze()

    if return_ids:
        return pred_contour, selected_ids

    return pred_contour


def get_selected_offset(idx, rf, offset_params):
    y_offset    = getattr(offset_params, 'y_offset', [0])
    zoom_ratio  = getattr(offset_params, 'zoom_ratio', [1])
    tempo_sr    = getattr(offset_params, 'tempo_sr', [1])

    n_y, n_z = len(y_offset), len(zoom_ratio)
    n_windows_per_tempo = [(int(rf*2*sr/100 + 1) - rf + 1) if rf*2*sr/100 % 1 != 0 else (int(rf*2*sr/100) - rf + 1) for sr in tempo_sr]

    y_i = idx % n_y
    z_i = (idx // n_y) % n_z
    fin = (idx // n_y) // n_z

    cumsum = 0
    for t_i, nw in enumerate(n_windows_per_tempo):
        if fin < cumsum + nw:
            w_i = fin - cumsum
            break
        cumsum += nw

    return t_i, z_i, y_i, w_i, fin


def extract_min_e(model, x):
    return model(x)[-1]


def visualize_kld_from_baseline(kld_dict, baseline):
    offsets = sorted(kld_dict.keys())
    kld_values = [kld_dict[o] for o in offsets]

    fig, axes = plt.subplots(2, 1, figsize=(20, 10))

    axes[0].bar(range(len(offsets)), kld_values, alpha=0.7, edgecolor='black')
    axes[0].axhline(y=np.mean(kld_values), color='r', linestyle='--',
                    label=f'Mean KLD: {np.mean(kld_values):.4f}')
    axes[0].set_xlabel('Offset Index')
    axes[0].set_ylabel('KL(Offset || Baseline)')
    axes[0].set_title('KL Divergence from Baseline Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    codebook_indices = np.arange(len(baseline))
    axes[1].bar(codebook_indices, baseline.numpy(), alpha=0.7, width=1.0)
    axes[1].set_xlabel('Used Codebook')
    axes[1].set_ylabel('Probability')
    axes[1].set_title('Baseline Distribution (Average of All Offsets)')

    plt.tight_layout()
    plt.show()

    print(f"Mean KLD: {np.mean(kld_values):.4f}")
    print(f"Std KLD:  {np.std(kld_values):.4f}")
    print(f"Min KLD:  {np.min(kld_values):.4f} (offset {offsets[np.argmin(kld_values)]})")
    print(f"Max KLD:  {np.max(kld_values):.4f} (offset {offsets[np.argmax(kld_values)]})")


def plot_confusion_matrix(y_true, y_pred, idx2label):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(cm, display_labels=idx2label.values())
    disp.plot(cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.show()
