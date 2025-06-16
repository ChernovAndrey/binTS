import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Literal
import torch


class OneHotQuantizer(BaseEstimator, TransformerMixin):
    def __init__(self, num_bins=1000, min_val=-10.0, max_val=10.0):
        super().__init__()
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val
        print(f'num bins: {num_bins}')
        print(f'min_val: {self.min_val}')
        print(f'max_val: {self.max_val}')
        self.bin_edges_ = torch.linspace(self.min_val, self.max_val, self.num_bins + 1)
        self.bin_values_ = 0.5 * (self.bin_edges_[:-1] + self.bin_edges_[1:])

    def fit(self, values):
        pass

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def transform(self, values):
        self.bin_edges_ = self.bin_edges_
        bin_indices = torch.bucketize(values, self.bin_edges_, right=False) - 1
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)  # ensure within bounds

        one_hot = torch.zeros(*values.shape, self.num_bins, device=values.device)
        one_hot.scatter_(-1, bin_indices.unsqueeze(-1), 1.0)
        return one_hot

    def inverse_transform(self, one_hot_values):
        self.bin_values_ = self.bin_values_
        indices = one_hot_values.argmax(dim=-1)
        return self.bin_values_[indices]


class BinaryQuantizer(BaseEstimator, TransformerMixin):
    def __init__(self, num_bins=1000, min_val=-10.0, max_val=10.0):
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val
        print(f'num bins: {num_bins}')
        print(f'min_val: {min_val}')
        print(f'max_val: {max_val}')

        # Bin edges and values
        self.bin_edges_ = np.linspace(self.min_val, self.max_val, self.num_bins + 1)
        self.bin_values_ = 0.5 * (self.bin_edges_[:-1] + self.bin_edges_[1:])

    def fit(self, values):
        # no-op for fixed binning
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def transform(self, values):
        """
        Convert each value into a binary vector based on the bin thresholds.
        Output shape will be: original_shape + (num_bins,)
        """
        values = np.expand_dims(values, axis=-1)  # Shape: (..., 1)
        bin_thresholds = self.bin_edges_[1:].reshape(*((1,) * values.ndim), -1)  # Broadcast shape: (..., num_bins)

        return (values >= bin_thresholds).astype(np.float32)

    def inverse_transform(self, binary_vectors):
        """
        Reconstruct values from binary quantized representation.
        For each vector, returns the midpoint of the highest active bin.
        If all bins are zero, returns the first bin value (edge case).
        """
        reversed_bin = np.flip(binary_vectors, axis=-1)
        idx_first_one_reversed = reversed_bin.argmax(axis=-1)
        idx_last_one = self.num_bins - 1 - idx_first_one_reversed
        reconstructed = self.bin_values_[idx_last_one]

        # Edge case: all zeros
        all_zero_mask = binary_vectors.sum(axis=-1) == 0
        reconstructed[all_zero_mask] = self.bin_values_[0]

        return reconstructed


class TimeSeriesMeanScaler(BaseEstimator, TransformerMixin):
    """
    Scales each time series by dividing by its mean.
    Assumes input shape is (n_samples, time_steps)
    """

    def fit(self, X, y=None):
        # self.means_ = np.mean(X, axis=1, keepdims=True)
        self.means_ = np.mean(X)
        # Avoid division by zero
        # self.means_[self.means_ == 0] = 1.0
        return self

    def transform(self, X):
        return X / self.means_

    def inverse_transform(self, X_scaled):
        return X_scaled * self.means_


def get_preprocessing_pipeline(num_bins=1000, min_val=-15, max_val=15,
                               enc_type: Literal['bce', 'onehot'] = 'bce') -> Pipeline:
    # Full pipeline
    if enc_type == 'bce':
        return Pipeline([
            # ('scaler', StandardScaler()),
            ('scaler', TimeSeriesMeanScaler()),
            ('quantizer', BinaryQuantizer(num_bins=num_bins, min_val=min_val, max_val=max_val))
        ])
    else:
        return Pipeline([
            # ('scaler', StandardScaler()),
            ('scaler', TimeSeriesMeanScaler()),
            ('quantizer', OneHotQuantizer(num_bins=num_bins, min_val=min_val, max_val=max_val))
        ])
