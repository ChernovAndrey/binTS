import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from typing import Literal, Union
from data_utils.data_scaler import TemporalScaler, StandardScaler, IdentityScaler, BinScaler, BinaryQuantizer
def repeat(tensor: torch.Tensor, n: int, dim: int = 0):
    return tensor.repeat_interleave(repeats=n, dim=dim)


def sliding_window_batch(x, L, H):
    if len(x.shape) == 4:
        x = x.squeeze()
    """
    x: Tensor of shape (B, L+H, C)
    Returns: Tensor of shape (B, H, L, C)
    """
    B, total_len, C = x.shape
    assert total_len >= L + H, "Not enough sequence length for given L and H"

    windows = [x[:, h:h + L, :].unsqueeze(1) for h in range(H)]  # list of (B, 1, L, C)
    return torch.cat(windows, dim=1)  # (B, H, L, C)


def get_sequence_from_prob(p: torch.Tensor, is_sample: bool, eps: float = 1e-6):
    """
    p: Tensor of shape (B, D) with probabilities
    Returns:
        best_sequences: Tensor of shape (B, D) with the most probable [1...1, 0...0] sequence
        best_probs: Tensor of shape (B,) with normalized probability of the best sequence
    """
    B, D = p.shape

    # Clamp p to avoid log(0) or log(1) instability
    p_clamped = p.clamp(min=eps, max=1 - eps)

    # Use log domain to compute cumulative products
    log_p = torch.log(p_clamped)
    log_1_minus_p = torch.log(1 - p_clamped)

    log_success = torch.cumsum(log_p, dim=1)  # shape (B, D)
    log_fail = torch.cumsum(log_1_minus_p.flip(dims=[1]), dim=1).flip(dims=[1])  # shape (B, D)

    # Pad with log(1) = 0 to align indexing
    zero = torch.zeros((B, 1), dtype=p.dtype, device=p.device)
    log_success = torch.cat([zero, log_success], dim=1)  # shape (B, D+1)
    log_fail = torch.cat([log_fail, zero], dim=1)  # shape (B, D+1)

    # Sum log-probs for each possible cutoff (index k: first 0 after all 1s)
    log_probs = log_success + log_fail  # shape (B, D+1)
    log_probs_max = torch.max(log_probs, dim=1, keepdim=True)[0]
    probs_normalized = torch.exp(log_probs - log_probs_max)
    probs_normalized = probs_normalized / probs_normalized.sum(dim=1, keepdim=True)

    # Sample or take the most probable index
    if is_sample:
        k = torch.multinomial(probs_normalized, num_samples=1)
    else:
        k = torch.argmax(probs_normalized, dim=1, keepdim=True)

    # Create the monotonic sequence [1,...,1,0,...,0]
    arange = torch.arange(D, device=p.device).unsqueeze(0)
    best_sequences = (arange < k).to(p.dtype)  # shape (B, D)

    best_probs = torch.gather(probs_normalized, dim=1, index=k).squeeze(1)

    return best_sequences, best_probs


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            # x = x * self.weight[:, None, None] + self.bias[:, None, None]
            x = x * self.weight[:, None] + self.bias[:, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"


def most_probable_monotonic_sequence(p: torch.Tensor):
    n = p.shape[0]

    # Cumulative products
    left_cumprod = torch.cumprod(p, dim=0)  # prod(p[0] ... p[k-1])
    right_cumprod = torch.cumprod((1 - p).flip(0), dim=0).flip(0)  # prod(1-p[k] ... 1-p[n-1])

    # Pad with 1 at ends to handle edge cases (k = 0 and k = n)
    left = torch.cat([torch.tensor([1.0], dtype=p.dtype, device=p.device), left_cumprod[:-1]])
    right = right_cumprod

    # Element-wise multiplication to get probability of each sequence
    probs = left * right
    probs_normalized = probs / probs.sum()

    best_k = torch.argmax(probs_normalized).item()
    best_sequence = torch.cat([
        torch.ones(best_k, dtype=p.dtype, device=p.device),
        torch.zeros(n - best_k, dtype=p.dtype, device=p.device)
    ])

    return best_sequence, probs_normalized[best_k]

class BinConv(nn.Module):
    def __init__(self, context_length: int, is_prob_forecast: bool, num_bins: int, min_bin_value=-10.0,
                 max_bin_value=10.0, kernel_size_across_bins_2d: int = 3,
                 kernel_size_across_bins_1d: int = 3, num_filters_2d: int = 8,
                 num_filters_1d: int = 32, is_cum_sum: bool = False, num_1d_layers: int = 2, num_blocks: int = 3,
                 kernel_size_ffn: int = 51, dropout: float = 0.2, target_dim=1, prediction_length:int = 36,
                 last_layer: Literal["conv", "fc"] = 'conv',
                 scaler_type: Union[Literal["standard", "temporal", "None"], None] = None) -> None:
        super().__init__()
        assert kernel_size_across_bins_2d % 2 == 1, "2D kernel size must be odd"
        assert kernel_size_across_bins_1d % 2 == 1, "1D kernel size must be odd"

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_bins = num_bins
        self.target_dim = target_dim
        print(f'dropout:{dropout}')
        print(f'target dim: {self.target_dim}')
        self.is_prob_forecast = is_prob_forecast
        self.num_filters_2d = num_filters_2d
        self.num_filters_1d = num_filters_1d
        self.kernel_size_across_bins_2d = kernel_size_across_bins_2d
        self.kernel_size_across_bins_1d = kernel_size_across_bins_1d
        self.is_cum_sum = is_cum_sum

        print(f'per sample scaler type is {scaler_type}')
        if scaler_type is None:
            self.scalers = None
        elif scaler_type == 'standard':
            self.scalers = [BinScaler(StandardScaler(var_specific=True),
                                      BinaryQuantizer(num_bins=num_bins, min_val=min_bin_value, max_val=max_bin_value))
                            for _ in range(self.target_dim)]
        elif scaler_type == 'temporal':
            self.scalers = [BinScaler(TemporalScaler(time_first=False),
                                      BinaryQuantizer(num_bins=num_bins, min_val=min_bin_value, max_val=max_bin_value))
                            for _ in range(self.target_dim)]
        else:
            assert False, f"The scaler type {scaler_type} is not supported"
        self.num_1d_layers = num_1d_layers
        self.num_blocks = num_blocks
        self.kernel_size_ffn = kernel_size_ffn
        self.dropout = nn.Dropout(dropout)
        # Conv2d over (context_length, num_bins)

        assert num_filters_2d == num_filters_1d, "todo: change the self.act shape if not"
        self.act = nn.ModuleList([
            nn.ModuleList([
                DynamicTanh(normalized_shape=num_filters_2d if i < self.num_1d_layers else context_length,
                            channels_last=False)
                for i in range(1)  # applied only after conv2d
                # for i in range(self.num_1d_layers + 1)  # applied only after conv2d
            ]) for _ in range(self.num_blocks)
        ])
        # self.act = nn.ModuleList([
        #     nn.ModuleList([
        #         nn.Sequential(
        #             nn.LayerNorm(
        #                 normalized_shape=(num_filters_2d if i < self.num_1d_layers else context_length)
        #             ),
        #             nn.ReLU()
        #         )
        #         for i in range(1)
        #     ])
        #     for _ in range(self.num_blocks)
        # ]) # used only for ablation study
        print('activation functions after conv2d:')
        print(self.act)

        layers = []
        for i in range(target_dim):
            conv2d = nn.ModuleList([nn.Conv2d(
                in_channels=1,
                out_channels=self.num_filters_2d,
                # kernel_size=(context_length if i == 0 else kernel_size_across_bins_2d, kernel_size_across_bins_2d),
                kernel_size=(context_length, kernel_size_across_bins_2d),
                bias=True
            ) for _ in range(num_blocks)
            ])
            conv1d = nn.ModuleList([
                nn.ModuleList([
                    nn.Conv1d(in_channels=num_filters_2d if i == 0 else num_filters_1d,
                              out_channels=context_length if i == num_1d_layers - 1 else num_filters_1d,
                              kernel_size=kernel_size_across_bins_1d, bias=True,
                              groups=num_filters_1d)
                    for i in range(num_1d_layers)
                ]) for _ in range(num_blocks)
            ])
            self.last_layer = last_layer
            if last_layer == 'conv':
                conv_ffn = nn.Conv1d(
                    in_channels=context_length,
                    out_channels=1,
                    kernel_size=kernel_size_ffn,  # large kernel size?
                    groups=1,
                    bias=True
                )
            elif last_layer == 'fc':
                class MeanOverChannel(nn.Module):
                    def __init__(self, dim=-2):
                        super().__init__()
                        self.dim = dim

                    def forward(self, x):
                        return x.mean(dim=self.dim)


                conv_ffn = nn.Sequential(
                    MeanOverChannel(dim=-2),  # Averages over the channel dimension
                    nn.Linear(in_features=self.num_bins, out_features=self.num_bins, bias=True)
                ) #was used for mlp ablation study
            else:
                assert False, f"last layer {last_layer} is not supported"
            assert num_filters_2d == num_filters_1d, "todo: change the self.act shape if not"
            act = nn.ModuleList([
                nn.ModuleList([
                    # DynamicTanh(normalized_shape=num_filters_2d if i == 0 else num_filters_1d, channels_last=False)
                    DynamicTanh(normalized_shape=num_filters_2d if i < self.num_1d_layers else context_length,
                                channels_last=False)
                    for i in range(1)
                    # applied after conv2d, and all conv1d including the last one
                ]) for _ in range(self.num_blocks)
            ])
            print(f'target dim: {i}')
            print('conv 2d layers:')
            print(conv2d)
            print('conv 1d layers:')
            print(conv1d)
            print('conv ffn layer:')
            print(conv_ffn)
            layers.append(nn.ModuleDict({
                'conv2d': conv2d,
                'conv1d': conv1d,
                'conv_ffn': conv_ffn,
                'act': act,
            }))
        self.layers = nn.ModuleList(layers)

    def _pad_channels(self, tensor: torch.Tensor, pad_size: int, pad_val_left=1.0, pad_val_right=0.0):
        if pad_size == 0:
            return tensor
        left = torch.full((*tensor.shape[:-1], pad_size), pad_val_left, device=tensor.device)
        right = torch.full((*tensor.shape[:-1], pad_size), pad_val_right, device=tensor.device)
        return torch.cat([left, tensor, right], dim=-1)

    def conv_layer(self, x: torch.Tensor, conv_func, act_func, kernel_size: int, is_2d: bool, ):
        # kernel_size = self.kernel_size_across_bins_2d if is_2d else self.kernel_size_across_bins_1d
        pad = kernel_size // 2 if kernel_size > 1 else 0
        x_padded = self._pad_channels(x, pad)
        if is_2d:
            x_padded = x_padded.unsqueeze(1)
        conv_out = conv_func(x_padded)  # (batch_size, num_filters_2d, num_bins)

        if is_2d:
            conv_out = conv_out.squeeze(2)
        if act_func is not None:
            conv_out = act_func(conv_out)
        return conv_out

    def forward(self, x, layer_id=0):
        if len(x.shape) == 4:
            x = x.squeeze()
        x = x.float()
        # x: (batch_size, context_length, num_bins)
        batch_size, context_length, num_bins = x.shape
        assert context_length == self.context_length, "Mismatch in context length"

        for j in range(self.num_blocks):
            residual = x
            x = self.conv_layer(x, self.layers[layer_id]["conv2d"][j], self.layers[layer_id]["act"][j][0],
                                self.kernel_size_across_bins_2d, True)
            for i in range(self.num_1d_layers):
                x = self.conv_layer(x, self.layers[layer_id]["conv1d"][j][i], F.relu,
                                    self.kernel_size_across_bins_1d, False)
            x = self.dropout(x)
            x = x + residual
        if self.last_layer == 'conv':
            out = self.conv_layer(x, self.layers[layer_id]["conv_ffn"], None, self.kernel_size_ffn, False).squeeze(1)
        else:
            out = self.layers[layer_id]["conv_ffn"](x)
        if self.is_cum_sum:
            assert False, "Do not use it, it degrades the performance"
            out = torch.flip(torch.cumsum(torch.flip(out, dims=[1]), dim=1), dims=[1])
        return out

    @torch.inference_mode()
    def forecast(self, batch_data, num_samples=None):
        do_sample = num_samples is not None and num_samples > 1 and self.is_prob_forecast
        inputs = batch_data.unsqueeze(2)
        print(f'batch_data shape:{inputs.shape}')
        forecasts_list = []
        for c in range(inputs.shape[2]):
            if self.scalers is not None:
                self.scalers[c].fit(inputs[:, :, c:c + 1].reshape(-1))
                c_inputs = self.scalers[c].transform(inputs[:, :, c:c + 1])
            else:
                c_inputs = inputs[:, :, c:c + 1]

            if do_sample:
                c_inputs = repeat(c_inputs.unsqueeze(1), num_samples, 1)  # (B, NS, T, D)
                batch_size = c_inputs.shape[0]
                c_inputs = c_inputs.view(-1, *c_inputs.shape[2:])
            current_context = c_inputs.clone()
            c_forecasts = []
            for _ in range(self.prediction_length):
                pred = F.sigmoid(self(current_context))  # (B, D)
                # print(pred.min(), pred.max())
                pred, _ = get_sequence_from_prob(pred, do_sample)
                pred = pred.int()
                c_forecasts.append(pred.unsqueeze(1))  # (B, 1, D)
                next_input = pred.unsqueeze(1)

                if len(current_context.shape) == 4:
                    next_input = next_input.unsqueeze(1)
                current_context = torch.cat([current_context[:, 1:], next_input], dim=1)

            c_forecasts = torch.cat(c_forecasts, dim=1)
            if self.scalers is not None:
                c_forecasts = self.scalers[c].inverse_transform(c_forecasts)
            if do_sample:
                c_forecasts = c_forecasts.view(batch_size, num_samples, *c_forecasts.shape[1:])
            else:
                c_forecasts = c_forecasts.unsqueeze(1)  # (B, 1,  T, D)
            if inputs.shape[2] > 1:
                c_forecasts = c_forecasts.unsqueeze(-2)  # (B, 1, T, D, num_bins)
            forecasts_list.append(c_forecasts)
        forecasts = torch.concat(forecasts_list, dim=-2)  # was 2
        return forecasts

    # def forward(self, x): the old one
    #     def pad_channels(tensor, pad_size: int, pad_val_left=1.0, pad_val_right=0.0):
    #         if pad_size == 0:
    #             return tensor
    #         left = torch.full((*tensor.shape[:-1], pad_size), pad_val_left, device=tensor.device)
    #         right = torch.full((*tensor.shape[:-1], pad_size), pad_val_right, device=tensor.device)
    #         return torch.cat([left, tensor, right], dim=-1)
    #
    #     # x: (batch_size, context_length, num_bins)
    #     batch_size, context_length, num_bins = x.shape
    #     assert context_length == self.context_length, "Mismatch in context length"
    #
    #     pad2d = self.kernel_size_across_bins_2d // 2 if self.kernel_size_across_bins_2d > 1 else 0
    #     x_padded = pad_channels(x, pad2d)
    #     x_conv_in = x_padded.unsqueeze(1)
    #     conv_out = F.relu(self.conv(x_conv_in).squeeze(2))  # (batch_size, num_filters_2d, num_bins)
    #
    #     pad1d = self.kernel_size_across_bins_1d // 2 if self.kernel_size_across_bins_1d > 1 else 0
    #     h_padded = pad_channels(conv_out, pad1d)
    #     h = F.relu(self.conv1d_1(h_padded))
    #
    #     h_padded = pad_channels(h, pad1d)
    #     out = self.conv1d_2(h_padded).mean(dim=1)  # (batch_size, num_bins)
    #
    #     if self.is_cum_sum:
    #         out = torch.flip(torch.cumsum(torch.flip(out, dims=[1]), dim=1), dims=[1])
    #
    #     return out



    def predict(self, x: torch.tensor, prediction_length: int) -> torch.Tensor:
        """
        Autoregressive prediction over `prediction_length` steps.

        Args:
            dl: DataLoader yielding (B, context_length, num_bins)
            prediction_length: number of future steps to forecast

        Returns:
            Tensor of shape (N, prediction_length, num_bins)
        """
        device = next(self.parameters()).device
        x = x.to(device)
        current_context = x.clone()
        forecasts = []
        for i in range(prediction_length):
            pred = F.sigmoid(self(current_context))  # (B, D)
            # pred = (pred >= 0.5).int()
            print(i)
            pred, _ = most_probable_monotonic_sequence(pred)
            pred = pred.int()
            forecasts.append(pred.unsqueeze(1))  # (B, 1, D)
            next_input = pred.unsqueeze(1)
            current_context = torch.cat([current_context[:, 1:], next_input], dim=1)

        return torch.cat(forecasts, dim=1)  # (B, T, D)


class LightningBinConv(BinConv, pl.LightningModule):
    def __init__(self, lr, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr
    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        # assert inputs.shape[-1] == self.context_length
        # assert targets.shape[-1] == self.prediction_length

        # distr_args, loc, scale = self(past_target)
        # distr = self.distr_output.distribution(distr_args, loc, scale)
        # loss = -distr.log_prob(future_target)
        #
        # return loss.mean()
        logits = self(inputs)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        # loss = F.softplus(-targets * logits).mean()
        self.log("train_loss", loss)  # TODO: use log properly
        print(loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
