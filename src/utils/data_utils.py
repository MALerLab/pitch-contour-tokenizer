import random
import unicodedata
from typing import Tuple
from pathlib import Path

import pandas as pd
from omegaconf.dictconfig import DictConfig

from torch.utils.data import Dataset

from datasets import dataset_zoo
from utils.train_utils import set_seed

class BaseDatasetMaker:
    def __init__(self,
                 config: DictConfig,) -> None:
        set_seed(config.random_seed)
        self.data_pth = sorted(Path(config.data.data_pth).rglob('*.csv')) # sort before any shuffle (rglob order is filesystem-dependent)
        self.data_root = config.data.data_pth

        self.dataset_class = getattr(dataset_zoo, config.dataset.name)
        self.dataset_params = config.dataset.params

        self.split_mode = config.split.name
        self.split_params = config.split.params


    def _get_random_split(self) -> Tuple[list | None]:
        rng = random.Random(self.split_params.get('split_seed', 42)) # random_seed와 분리, 고정 split
        rng.shuffle(self.data_pth)
        split_ratio = self.split_params.split_ratio
        n_total = len(self.data_pth)

        if self.split_params.with_test:
            n_train, n_valid = int(n_total * split_ratio[0]), int(n_total * split_ratio[1])
            train_pth, valid_pth, test_pth = self.data_pth[:n_train], self.data_pth[n_train:n_train+n_valid], self.data_pth[n_train+n_valid:]
        else:
            n_train = int(n_total * split_ratio[0])
            train_pth, valid_pth = self.data_pth[:n_train], self.data_pth[n_train:]
            test_pth = None
        
        return train_pth, valid_pth, test_pth


    def _get_manifest_split(self) -> Tuple[list | None]:
        # Explicit per-file split shipped with the repo (`splits/*.csv`, columns: path,artist,split).
        # split in {train, valid, test, exclude}; paths are relative to data.data_pth. Every file on disk must be listed.
        split_df = pd.read_csv(self.split_params.manifest_pth)
        split_map = {unicodedata.normalize('NFC', p): s
                     for p, s in zip(split_df['path'], split_df['split'])} # NFC: tolerate NFD-normalised filenames

        splits = {'train': [], 'valid': [], 'test': [], 'exclude': []}
        for pth in self.data_pth:
            rel = unicodedata.normalize('NFC', pth.relative_to(self.data_root).as_posix())
            split = split_map.pop(rel, None)
            if split is None:
                raise KeyError(f'file not in split manifest {self.split_params.manifest_pth}: {rel} '
                               f'(use split=random for custom data)')
            splits[split].append(pth)
        if split_map:
            raise FileNotFoundError(f'{len(split_map)} manifest entries not found on disk, e.g. {sorted(split_map)[:3]}')
        if splits['exclude']:
            print(f"[split] {len(splits['exclude'])} files marked 'exclude' in the manifest are skipped.")

        test_pth = splits['test'] if self.split_params.with_test else None
        return splits['train'], splits['valid'], test_pth


    def _get_split(self) -> Tuple[list | None]:
        if self.split_mode == 'random':
            train_pth, valid_pth, test_pth = self._get_random_split()
        elif self.split_mode == 'manifest':
            train_pth, valid_pth, test_pth = self._get_manifest_split()
        else:
            raise NotImplementedError

        return train_pth, valid_pth, test_pth


    def get_datasets(self) -> Tuple[Dataset | None]:
        train_pth, valid_pth, test_pth = self._get_split()
        trainset = self.dataset_class(train_pth, **self.dataset_params, is_valid=False)
        validset = self.dataset_class(valid_pth, **self.dataset_params, is_valid=True)
        testset = self.dataset_class(test_pth, **self.dataset_params, is_valid=True) if test_pth is not None else None

        return trainset, validset, testset




class ClsDatasetMaker(BaseDatasetMaker):
    def __init__(self, 
                 config: DictConfig,) -> None:
        set_seed(config.random_seed)
        self.data_pth = config.data.data_pth

        self.dataset_class = getattr(dataset_zoo, config.dataset.name)
        self.dataset_params = config.dataset.params

        self.split_mode = config.split.name
        self.split_params = config.split.params

        self.metadata = pd.read_csv(config.data.metadata_pth)
        self.label = self.metadata['label'].tolist()
        self.label_map = {k:idx for idx, k in enumerate(sorted(set(self.label)))}


    def _get_random_split(self) -> Tuple[list | None]:
        indices = list(range(len(self.data_pth)))
        random.shuffle(indices)
        
        self.data_pth = [self.data_pth[i] for i in indices]
        self.label = [self.label[i] for i in indices]
        
        split_ratio = self.split_params.split_ratio
        n_total = len(self.data_pth)

        if self.split_params.with_test:
            n_train, n_valid = int(n_total * split_ratio[0]), int(n_total * split_ratio[1])
            train_pth, valid_pth, test_pth = self.data_pth[:n_train], self.data_pth[n_train:n_train+n_valid], self.data_pth[n_train+n_valid:]
            train_label, valid_label, test_label = self.label[:n_train], self.label[n_train:n_train+n_valid], self.label[n_train+n_valid:]
        else:
            n_train = int(n_total * split_ratio[0])
            train_pth, valid_pth = self.data_pth[:n_train], self.data_pth[n_train:]
            train_label, valid_label = self.label[:n_train], self.label[n_train:]
            test_pth, test_label = None, None
        
        train_info, valid_info, test_info = (train_pth, train_label), (valid_pth, valid_label), (test_pth, test_label)

        return train_info, valid_info, test_info


    def _get_stratified_split(self) -> Tuple[list | None]:
        train_df = self.metadata[self.metadata['split']=='train']
        valid_df = self.metadata[self.metadata['split']=='valid']

        if self.split_params.with_test:
            test_df = self.metadata[self.metadata['split']=='test']
        else:
            test_df = None

        return train_df, valid_df, test_df


    def _get_split(self) -> Tuple[list | None]:
        if self.split_mode == 'random':
            train_info, valid_info, test_info = self._get_random_split()
        elif self.split_mode == 'stratified':
            train_info, valid_info, test_info = self._get_stratified_split()
        else:
            raise NotImplementedError

        return train_info, valid_info, test_info


    def get_datasets(self) -> Tuple[Dataset | None]:
        train_info, valid_info, test_info = self._get_split()
        trainset = self.dataset_class(self.data_pth, train_info, self.label_map, **self.dataset_params, is_valid=False)
        validset = self.dataset_class(self.data_pth, valid_info, self.label_map, **self.dataset_params, is_valid=True)
        testset = self.dataset_class(self.data_pth, test_info, self.label_map, **self.dataset_params, is_valid=True) if test_info is not None else None

        return trainset, validset, testset