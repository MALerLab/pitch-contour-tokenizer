import warnings
from typing import Optional, List, Tuple

import torch
import torchaudio
import torch.nn as nn
from torch import Tensor
from torch.nn import _reduction as _Reduction


class MSELoss(nn.Module):
    def __init__(self, 
                 reduction: str="mean",
                 use_weight: bool=True) -> None:
        super().__init__()
        self.reduction = reduction
        self.use_weight = use_weight

    def forward(self, 
                pred: Tensor, 
                target: Tensor, 
                weight: Optional[Tensor]=None,
                **kwargs) -> Tensor:
        if isinstance(pred, tuple):
            pred, target = pred[0], target[0]

        if not (target.size() == pred.size()):
            warnings.warn(
                f"Using a target size ({target.size()}) that is different to the input size ({pred.size()}). "
                "This will likely lead to incorrect results due to broadcasting. "
                "Please ensure they have the same size.",
                stacklevel=2,
            )

        pred, target = torch.broadcast_tensors(pred, target)

        if (weight is not None) and (self.use_weight):
            if weight.size() != pred.size():
                raise ValueError('Weights and input must have the same size.')
            
            weight = ~weight

            squared_errors = torch.pow(pred-target, 2)
            if self.reduction == 'none':
                loss = squared_errors * weight
            elif self.reduction == 'sum':
                loss = torch.sum(squared_errors * weight)
            elif self.reduction == 'mean':
                weighted_squared_errors = torch.sum(squared_errors * weight, dim=-1) / torch.sum(weight, dim=-1)
                loss = torch.mean(weighted_squared_errors)
            else:
                raise ValueError(
                    f"Invalid reduction mode: {self.reduction}. Expected one of 'none', 'sum', 'mean'."
                )
        
        else:
            loss = torch._C._nn.mse_loss(pred, target, _Reduction.get_enum(self.reduction))
        
        return loss



class OffsetMSELoss(MSELoss):
    def __init__(self, 
                 receptive_field: int,
                 reduction: str="mean",
                 use_weight: bool=True) -> None:
        super().__init__(reduction, use_weight)
        self.rf = receptive_field


    def forward(self, 
                pred: Tensor, 
                target: Tensor, 
                weight: Optional[Tensor]=None,
                **kwargs) -> Tensor:
        if isinstance(pred, tuple):
            pred, target = pred[0], target[0]
        b,c,r,t = pred.shape
        pred, target = pred.reshape(b,c,r,t//self.rf,self.rf), target.unsqueeze(-2).reshape(b,c,1,t//self.rf,self.rf)
        squared_errors = torch.pow(pred-target, 2) # b,c_in,rolled_size,t_comp,rf

        if (weight is not None) and (self.use_weight):
            weight = weight.unsqueeze(-2).unsqueeze(-2).reshape(b,c,1,t//self.rf,self.rf)
            # weighted_squared_errors = torch.sum(squared_errors * weight, dim=-1) / torch.sum(weight, dim=-1)
            weighted_squared_errors = torch.sum(squared_errors * ~weight, dim=-1) / torch.sum(~weight, dim=-1)
            min_squared_errors = torch.min(weighted_squared_errors, dim=-2).values
        else:
            avg_squared_errors = torch.mean(squared_errors, dim=-1)
            min_squared_errors = torch.min(avg_squared_errors, dim=-2).values
        loss = torch.mean(min_squared_errors)

        return loss
    # def forward(self, 
    #             pred: Tensor, 
    #             target: Tensor, 
    #             weight: Optional[Tensor]=None,
    #             **kwargs) -> Tensor:
    #     if isinstance(pred, tuple):
    #         pred, target = pred[0], target[0].unsqueeze(-2)

    #     squared_errors = torch.pow(pred-target, 2) # b,c_in,rolled_size,t

    #     if (weight is not None) and (self.use_weight):
    #         weight = weight.unsqueeze(-2)
    #         weighted_squared_errors = torch.sum(squared_errors * weight, dim=-1) / torch.sum(weight, dim=-1)
    #         min_weighted_squared_errors = torch.min(weighted_squared_errors, dim=-1).values
    #         loss = torch.mean(min_weighted_squared_errors)

    #     else:
    #         avg_squared_errors = torch.mean(squared_errors, dim=-1)
    #         min_avg_squared_errors = torch.min(avg_squared_errors, dim=-1).values
    #         loss = torch.mean(min_avg_squared_errors)

    #     return loss




# class FFTMSELoss(nn.Module):
#     def __init__(self, 
#                  mode: str,
#                  reduction: str="mean",
#                  w_contour: float=0.0,
#                  use_weight: bool=True) -> None:
#         super().__init__()
#         self.mode = mode
#         self.reduction = reduction

#         self.w_contour = w_contour
#         if self.w_contour:
#             self.contour_criterion = MSELoss(reduction=reduction, use_weight=use_weight)

#     def forward(self, 
#                 input: Tensor, 
#                 target: Tensor, 
#                 weight: Tensor,
#                 **kwargs) -> Tensor:
#         x, f = input[0], input[1]
#         x_hat, f_hat = target[0], target[1]

#         x, x_hat = torch.broadcast_tensors(x, x_hat)
#         f, f_hat = torch.broadcast_tensors(f, f_hat)

#         if self.mode == 'cartesian':
#             real, imag = f.chunk(chunks=2, dim=-2) # b,1,t
#             real_hat, imag_hat = f_hat.chunk(chunks=2, dim=-2)
#             real_mse = torch._C._nn.mse_loss(real, real_hat, _Reduction.get_enum(self.reduction))
#             imag_mse = torch._C._nn.mse_loss(imag, imag_hat, _Reduction.get_enum(self.reduction))
#             fft_loss = (real_mse + imag_mse)/2

#         elif self.mode == 'polar':
#             mag, phi = f.chunk(chunks=2, dim=-2) # b,1,t
#             mag_hat, phi_hat = f_hat.chunk(chunks=2, dim=-2)

#             cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)
#             cos_phi_hat, sin_phi_hat = torch.cos(phi_hat), torch.sin(phi_hat)

#             cos_phi_mse = torch._C._nn.mse_loss(cos_phi_hat, cos_phi, _Reduction.get_enum(self.reduction))
#             sin_phi_mse = torch._C._nn.mse_loss(sin_phi_hat, sin_phi, _Reduction.get_enum(self.reduction))

#             phi_mse = (cos_phi_mse+sin_phi_mse)/2
#             mag_mse = torch._C._nn.mse_loss(mag_hat, mag, _Reduction.get_enum(self.reduction))
#             fft_loss = (mag_mse + phi_mse)/2
        
#         loss = fft_loss

#         if self.w_contour:
#             contour_loss = self.contour_criterion(x, x_hat, weight=weight)
#             loss = self.w_contour*contour_loss + (1-self.w_contour)*fft_loss
        
#         return loss




# class OffsetMSELoss(nn.Module):
#     def __init__(self, 
#                  window: int=64,
#                  reduction: str="mean",
#                  use_weight: bool=False) -> None:
#         super().__init__()
#         self.window = window
#         self.reduction = reduction
#         self.use_weight = use_weight

#     def forward(self, 
#                 input: Tensor, 
#                 target: Tensor, 
#                 weight: Optional[Tensor]=None,
#                 **kwargs) -> Tensor:
#         if isinstance(input, tuple):
#             input, target = input[0], target[0]

#         unfolded_pred_contour = input.unfold(dimension=-1, size=self.window, step=1)
#         squared_errors = torch.pow((target.unsqueeze(-2) - unfolded_pred_contour), 2)

#         if self.use_weight and weight is not None:
#             if weight.ndim != squared_errors.ndim:
#                 weight = weight.unsqueeze(-2)

#             loss = torch.mean(squared_errors * weight, dim=-1).min(dim=-1)[0].mean()

#         else:
#             loss = torch.mean(squared_errors, dim=-1).min(dim=-1)[0].mean()
        
#         return loss




# ---- OffsetGridCELoss offset별 점수 커널 (track 경로) ----
# 입력 팩: OffsetGrid2DVQVAE.decode 반환값 (grid_variants, mask_variants, meta)
#   grid_variants[t]: (batch, 1, n_freq, n_variant, track_len_t) — tempo 트랙 t의 freq-bin logits,
#                     n_variant = n_y·n_z (y-major·zoom-minor)
#   mask_variants[t]: (batch, 1, track_len_t) — mask(unvoiced) logit, 트랙 내 변형 간 공유
#   meta: {step, rf, n_windows, n_zoom, n_y} — 트랙 t의 창 수 = (track_len_t − rf)//step + 1
# grid_target: (batch, 1, n_freq, rf) — to_grid one-hot, unvoiced 프레임은 all-zero
# 출력 점수: (batch, 1, n_offset) — flat offset 순서는 y > zoom > tempo·window


class VectorQuantizeLoss(nn.Module):
    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta


    def forward(self, 
                z: torch.Tensor, 
                z_q: torch.Tensor) -> torch.Tensor:
        codebook_loss = torch.nn.functional.mse_loss(z_q, z.detach())
        commitment_loss = torch.nn.functional.mse_loss(z_q.detach(), z)
        loss = codebook_loss + self.beta * commitment_loss

        return loss, codebook_loss, commitment_loss
