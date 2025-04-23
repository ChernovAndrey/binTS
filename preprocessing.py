import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class BinaryQuantizer(BaseEstimator, TransformerMixin):
    def __init__(self, num_bins=100, min_val=-5.0, max_val=5.0):
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, X, y=None):
        self.bin_values_ = np.linspace(self.min_val, self.max_val, self.num_bins)
        return self

    def transform(self, X):
        # X shape: (n_samples, n_features)
        bin_thresholds = self.bin_values_.reshape(1, 1, -1)
        X_expanded = X[:, :, np.newaxis]  # shape: (n_samples, n_features, 1)
        return (X_expanded >= bin_thresholds).astype(int)

    # def transform(self, X):
    #     # X shape: (n_samples, n_features)
    #     bin_thresholds = self.bin_values_.reshape(1, 1, -1)
    #     X_expanded = X[:, :, np.newaxis]  # shape: (n_samples, n_features, 1)
    #     return ((X_expanded >= bin_thresholds).astype(int) * 2 - 1)

    def inverse_transform(self, X_bin: np.ndarray) -> np.ndarray:
        # X_bin shape: (n_samples, n_features, num_bins)
        # Reverse bins along the last axis and find first 1 in reversed
        reversed_bin = X_bin[..., ::-1]
        idx_first_one_reversed = reversed_bin.argmax(axis=-1)

        # Convert reversed index to original index: last 1
        idx_last_one = self.num_bins - 1 - idx_first_one_reversed

        # Map indices back to bin values
        reconstructed = self.bin_values_[idx_last_one]
        return reconstructed


def get_preprocessing_pipeline(num_bins=1000, min_val=-15, max_val=15) -> Pipeline:
    # Full pipeline
    return Pipeline([
        ('scaler', StandardScaler()),
        ('quantizer', BinaryQuantizer(num_bins=num_bins, min_val=min_val, max_val=max_val))
    ])
