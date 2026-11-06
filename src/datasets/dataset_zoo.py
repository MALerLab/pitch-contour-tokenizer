import math
import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self,
                 data_pth: str | list,
                 window: int | float,
                 sr: int | float=100,
                 weight: str | List[str] | bool=False,
                 is_valid: bool=False,) -> None:
        super().__init__()
        if isinstance(data_pth, str):
            self.data_dir = sorted(Path(data_pth).rglob('*.csv'))
        elif isinstance(data_pth, list):
            self.data_dir = data_pth

        self.weight = weight
        if isinstance(weight, str):
            self.weight_cols = [weight]
        elif hasattr(weight, "__iter__"):
            self.weight_cols = list(weight)
        else:
            self.weight_cols = []

        self.is_valid = is_valid

        self.sr = sr
        self.window = window

        self.loaded_data = self._load_data()


    def _frequency_to_midi(self, 
                          frequency: float) -> float:
        # Convert frequency to MIDI note
        return 69 + 12 * math.log2(frequency / 440)


    def _get_data(self,
                  csv_file: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        contour_df = pd.read_csv(csv_file, dtype={'frequency':np.float32})
        frequency = contour_df['frequency'].values
        if self.weight_cols:
            weight = contour_df[self.weight_cols].values
        else:
            weight = np.ones((frequency.shape[-1], 1))

        midi = [self._frequency_to_midi(freq)/12 for freq in frequency] # 1 Octave => 1.0

        midi = torch.tensor(midi, dtype=torch.float32)
        weight = torch.tensor(weight.T, dtype=torch.float32)

        return midi, weight


    def _load_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        loaded_data = []
        for csv_file in tqdm(self.data_dir, desc='Load Data'):
            midi, weight = self._get_data(csv_file)
            loaded_data.append((midi, weight))

        return loaded_data


    def __len__(self) -> int:
        return len(self.loaded_data)


    def __getitem__(self, idx:int):
        raise NotImplementedError("Subclasses must implement this method")




class FullPitchDataset(BaseDataset):
    def __init__(self,
                 data_pth: str | list,
                 window: int,
                 sr: int | float=100,
                 weight: str | List[str] | bool=False,
                 is_valid: bool=False,
                 threshold: None | Dict=None,
                 threshold_q: None | Dict=None,
                 num_interpolate: int=2,) -> None:
        super().__init__(data_pth, window, sr, weight, is_valid)
        self.threshold = threshold if threshold is not None else {}
        self.threshold_q = threshold_q if threshold_q is not None else {}
        self.num_interpolate = num_interpolate
        self._validate_threshold_keys()


    def _validate_threshold_keys(self):
        t, tq = set(self.threshold), set(self.threshold_q)
        if t & tq:
            raise ValueError(f"key {t & tq} is in threshold, threshold_q")

        unknown = (t | tq) - set(self.weight_cols)
        if unknown:
            raise ValueError(f"threshold key {unknown} is not in self.weight")


    def _aggregate(self, 
                    w: torch.Tensor) -> torch.Tensor:
        if not self.weight_cols:
            is_filtered = torch.zeros(w.shape[-1], dtype=torch.bool)
            return is_filtered

        filtered = []
        for i, name in enumerate(self.weight_cols):
            if name in self.threshold:
                threshold = self.threshold[name]
            elif name in self.threshold_q:
                threshold = torch.quantile(w[i][w[i] > 0], self.threshold_q[name])
            else:
                continue
            filtered.append(w[i] < threshold)

        is_filtered = torch.stack(filtered, dim=0).any(dim=0) if filtered else torch.zeros(w.shape[-1], dtype=torch.bool)

        return is_filtered


    def aggregate_weight_filter(self, 
                                w:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method")


    def get_segment_by_split(self,
                             x: torch.Tensor,
                             w: torch.Tensor,
                             is_filtered: torch.Tensor) -> Tuple[torch.Tensor]:
        x_len = x.shape[-1]
        max_start = x_len - self.window

        if self.is_valid:
            n_samples = x_len//self.window
            x_seg = x[:self.window*n_samples].reshape(n_samples, self.window)
            w_seg = w[..., :self.window*n_samples].reshape(-1, n_samples, self.window).permute(1, 0, 2)
            f_seg = is_filtered[:self.window*n_samples].reshape(n_samples, self.window)
            x, w, is_filtered = x_seg, w_seg, f_seg # 나머지 frame은 버림 — 모든 window가 비중첩 stride=window 타일
        else:
            start = random.randint(0, max_start)
            n_samples = (x_len-start)//self.window
            x = x[start:start+self.window*n_samples].reshape(n_samples,self.window)
            w = w[...,start:start+self.window*n_samples].reshape(-1,n_samples,self.window).permute(1,0,2)
            is_filtered = is_filtered[start:start+self.window*n_samples].reshape(n_samples,self.window)

        return x, w, is_filtered


    def filter_receptive_field(self, 
                              is_filtered: torch.Tensor, 
                              return_ids: bool) -> torch.Tensor:
        # interpolate 시 만족 가능한 segment일 경우 True
        interpolate_mask = ~(is_filtered.unfold(dimension=-1, step=1, size=self.num_interpolate+1).float().sum(dim=-1) >= self.num_interpolate + 1).sum(dim=-1).bool()
        # start/end에서 filtered 되지 않을 시 True
        start_end_mask = ~(is_filtered[...,0] | is_filtered[...,-1])
        
        final_mask = interpolate_mask & start_end_mask
        use_ids = torch.where(final_mask)[0]

        if return_ids:
            return final_mask, use_ids

        return final_mask


    # def interpolate_filtered_point(self, 
    #                                x: torch.Tensor, 
    #                                is_filtered: torch.Tensor) -> torch.Tensor:
    #     all_idx = np.arange(self.window)
    #     pitch, is_filtered = x.numpy(), is_filtered.numpy()
    #     x_out = pitch.copy()

    #     for target_row_idx in np.where(is_filtered.any(axis=-1))[0]:
    #         valid_idx = np.where(~is_filtered[target_row_idx])[0]
    #         x_out[target_row_idx] = np.interp(all_idx, valid_idx, pitch[target_row_idx][valid_idx])

    #     return x_out


    def interpolate_filtered_point(self, 
                                   x: torch.Tensor, 
                                   is_filtered: torch.Tensor) -> torch.Tensor:
        # x: (N, W) torch.Tensor, is_filtered: (N, W) bool
        n, w = x.shape
        pos = torch.arange(w).expand(n, w)

        # left_idx[i,j] = 위치 j에서 좌측으로 가장 가까운 valid 위치
        left_src  = torch.where(is_filtered, torch.full_like(pos, -1), pos)
        left_idx  = torch.cummax(left_src, dim=-1).values

        # right_idx[i,j] = 위치 j에서 우측으로 가장 가까운 valid 위치
        right_src = torch.where(is_filtered, torch.full_like(pos, w), pos)
        right_idx = torch.flip(torch.cummin(torch.flip(right_src, [-1]), dim=-1).values, [-1])

        left_val  = torch.gather(x, -1, left_idx)
        right_val = torch.gather(x, -1, right_idx)

        span      = (right_idx - left_idx).clamp_min(1).to(x.dtype)
        frac      = (pos - left_idx).to(x.dtype) / span
        interp    = left_val + (right_val - left_val) * frac

        x_out = torch.where(is_filtered, interp, x)

        return x_out


    def _cache_filter(self) -> None:
        self.loaded_data = [(x, w, self.aggregate_weight_filter(w))
                            for x, w in self.loaded_data]


    def sample_receptive_field(self) -> Tuple[torch.Tensor, torch.Tensor]:
        sampled_x, sampled_w = [], []
        for x, w, is_filtered_full in self.loaded_data:
            if x.shape[-1] < self.window:
                continue

            x_seg, w_seg, is_filtered = self.get_segment_by_split(x, w, is_filtered_full)
            final_mask = self.filter_receptive_field(is_filtered, return_ids=False)

            x_seg = x_seg[final_mask]
            w_seg = w_seg[final_mask]
            is_filtered = is_filtered[final_mask]

            x_seg = self.interpolate_filtered_point(x_seg, is_filtered)

            sampled_x.append(x_seg)
            sampled_w.append(is_filtered.numpy()) # weight 자리에 is_filtered (mask 의미론) — 보간으로 채운 frame 표시

        sampled_data = (torch.from_numpy(np.concatenate(sampled_x, axis=0)),
                        torch.from_numpy(np.concatenate(sampled_w, axis=0)).unsqueeze(-2))

        return sampled_data


    def resample_receptive_field(self) -> None:
        self.sampled_data = self.sample_receptive_field()


    def __len__(self) -> int:
        return len(self.sampled_data[0])


    def __getitem__(self,
                    idx: int) -> Tuple[torch.Tensor]:
        x, w = self.sampled_data[0][idx], self.sampled_data[1][idx]
        return x.unsqueeze(-2), w




class FullPitchThresholdDataset(FullPitchDataset):
    def __init__(self,
                 data_pth: str | list,
                 window: int,
                 sr: int | float=100,
                 weight: str | List[str] | bool=False,
                 is_valid: bool=False,
                 threshold: None | Dict=None,
                 threshold_q: None | Dict=None,
                 num_interpolate: int=2,) -> None:
        super().__init__(data_pth, window, sr, weight, is_valid, threshold, threshold_q, num_interpolate)
        self._cache_filter()
        self.sampled_data = self.sample_receptive_field()


    def aggregate_weight_filter(self,
                                w:torch.Tensor) -> torch.Tensor:
        return self._aggregate(w)




class NIAFixedClsDataset(BaseDataset):
    def __init__(self,
                 data_pth: str,
                 data_info: pd.DataFrame,
                 label_map: dict,
                 window: int=128,
                 sr: int | float=100,
                 weight: str | List[str] | bool=False,
                 **kwargs) -> None:
        self.data_dir = Path(data_pth)
        self.data_info = data_info
        self.label_map = label_map

        self.sr = sr
        self.window = window
        self.weight = weight

        if isinstance(weight, str):
            self.weight_cols = [weight]
        elif hasattr(weight, "__iter__"):
            self.weight_cols = list(weight)
        else:
            self.weight_cols = []

        self.loaded_data = self._load_data()


    def _crop_segment(self,
                      onset: int,
                      offset: int) -> Tuple[int, int]:
        center = onset + (offset-onset)//2
        onset = center-self.window//2
        offset = onset+self.window

        return onset, offset


    def _expand_segment(self,
                        onset: int,
                        offset: int,
                        seg_df: pd.DataFrame,
                        total_len: int) -> Tuple[int, int]:
        need = self.window - (offset - onset) # 보완해야할 양

        prev_rows = seg_df[seg_df['offset']*100 <= onset]
        next_rows = seg_df[seg_df['onset']*100 >= offset]
        prev_boundary = int(round(prev_rows.iloc[-1]['offset'], 2)*100) if not prev_rows.empty else 0  # 이전 레이블의 offset
        next_boundary = int(round(next_rows.iloc[0]['onset'], 2)*100) if not next_rows.empty else total_len  # 다음 레이블의 onset

        left_space = onset - prev_boundary # 이전 label offset 까지의 잔여 길이
        right_space = next_boundary - offset # 다음 label onset 까지의 잔여 길이

        if left_space + right_space >= need: # 이전 label offset, 다음 label onset 까지의 영역으로 window를 채울 수 있을 경우
            if left_space >= right_space: # left_space가 더 많을 경우, 남은 영역은 right_space로 보완
                l_pad = min(left_space, need)
                r_pad = need - l_pad
            else: # right_space가 더 많을 경우, 남은 영역은 left_space로 보완
                r_pad = min(right_space, need)
                l_pad = need - r_pad

        else: # 이전 label offset, 다음 label onset 까지의 영역으로 window를 채울 수 없을 경우
            extra = need - (left_space + right_space) # 잔존 영역을 소진하고도 필요한 양
            l_extra = extra // 2
            r_extra = extra - l_extra
            l_pad = left_space + l_extra
            r_pad = right_space + r_extra

        onset = onset - l_pad
        offset = offset + r_pad

        if onset < 0:
            onset = 0
            offset = self.window
        elif offset > total_len:
            offset = total_len
            onset = offset-self.window

        return onset, offset


    def _load_data(self) -> List[Tuple[torch.Tensor, torch.Tensor, str]]:
        self.check_list = []
        loaded_data = []
        for fn in tqdm(self.data_info['filename'].unique().tolist(), desc='Load Data'):
            x, w = self._get_data((self.data_dir/fn).with_suffix('.csv'))
            seg_df = self.data_info[self.data_info['filename']==fn]
            
            for _, row in seg_df.iterrows():
                onset, offset, label = int(round(row['onset'],2)*100), int(round(row['offset'],2)*100), row['label']
                seg_len = offset - onset
                seg_info = []
                seg_info.extend([fn, label, onset, offset])

                if self.window < seg_len: # seg_len이 window보다 길 경우 center에서 crop
                    onset, offset = self._crop_segment(onset, offset)
                elif self.window > seg_len: # seg_len이 window보다 부족할 경우
                    onset, offset = self._expand_segment(onset, offset, seg_df, x.shape[-1])

                seg = x[...,onset:offset]
                seg_w = w[...,onset:offset]

                seg_info.extend([onset, offset])
                self.check_list.append(seg_info)
                loaded_data.append((seg, seg_w, label))

        return loaded_data


    def __len__(self) -> int:
        return len(self.loaded_data)


    def __getitem__(self,
                    idx: int) -> Tuple[torch.Tensor]:
        x, w, label = self.loaded_data[idx]
        y = torch.tensor(self.label_map[label], dtype=torch.long)
        # x, weight = x.chunk(chunks=2, dim=0)

        return x.unsqueeze(0), w, y
