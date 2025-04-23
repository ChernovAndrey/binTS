import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.base import TransformerMixin
import numpy as np
from sklearn.pipeline import Pipeline


class TSDataset(Dataset):
    def __init__(self, data: np.array, context_length: int, scaler: None | Pipeline | TransformerMixin):
        self.scaler = scaler
        self.context_length = context_length

        data = data.reshape(-1, 1) if (len(data.shape) == 1) and (self.scaler is not None) else data
        # self.data = scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)
        self.data = np.squeeze(scaler.fit_transform(data)) if self.scaler is not None else data

    def __len__(self):
        return len(self.data) - self.context_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.data[index:index + self.context_length], dtype=torch.float),
            torch.tensor(self.data[index + self.context_length], dtype=torch.float),
        )

    def get_scaler(self) -> None | Pipeline | TransformerMixin:
        return self.scaler


# TODO: redesign the logic
class TSDatasetTest(TSDataset):
    def __init__(self, data: np.array, context_length: int, scaler: None | Pipeline | TransformerMixin):
        self.scaler = scaler
        self.context_length = context_length

        data = data.reshape(-1, 1) if (len(data.shape) == 1) and (self.scaler is not None) else data
        # self.data = scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)
        self.data = np.squeeze(scaler.transform(data)) if self.scaler is not None else data

    def __getitem__(self, index):
        return torch.tensor(self.data[index:index + self.context_length], dtype=torch.float)
