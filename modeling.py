import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl


class BinConv(nn.Module):
    def __init__(self, context_length: int, num_bins: int, kernel_size_across_bins_2d: int = 3,
                 kernel_size_across_bins_1d: int = 3, num_filters_2d: int = 8,
                 num_filters_1d: int = 32, is_cum_sum: bool = False) -> None:
        super().__init__()
        assert kernel_size_across_bins_2d % 2 == 1, "2D kernel size must be odd"
        assert kernel_size_across_bins_1d % 2 == 1, "1D kernel size must be odd"

        self.context_length = context_length
        self.num_bins = num_bins
        self.num_filters_2d = num_filters_2d
        self.num_filters_1d = num_filters_1d
        self.kernel_size_across_bins_2d = kernel_size_across_bins_2d
        self.kernel_size_across_bins_1d = kernel_size_across_bins_1d
        self.is_cum_sum = is_cum_sum

        # Conv2d over (context_length, num_bins)
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.num_filters_2d,
            kernel_size=(context_length, kernel_size_across_bins_2d),
            bias=True
        )

        self.conv1d_1 = nn.Conv1d(
            in_channels=self.num_filters_2d,
            out_channels=self.num_filters_1d,
            kernel_size=kernel_size_across_bins_1d,
            bias=True
        )

        self.conv1d_2 = nn.Conv1d(
            in_channels= self.num_filters_1d,
            out_channels=self.num_bins,
            kernel_size=kernel_size_across_bins_1d,
            bias=True
        )

    def forward(self, x):
        def pad_channels(tensor, pad_size: int, pad_val_left=1.0, pad_val_right=0.0):
            if pad_size == 0:
                return tensor
            left = torch.full((*tensor.shape[:-1], pad_size), pad_val_left, device=tensor.device)
            right = torch.full((*tensor.shape[:-1], pad_size), pad_val_right, device=tensor.device)
            return torch.cat([left, tensor, right], dim=-1)

        # x: (batch_size, context_length, num_bins)
        batch_size, context_length, num_bins = x.shape
        assert context_length == self.context_length, "Mismatch in context length"

        pad2d = self.kernel_size_across_bins_2d // 2 if self.kernel_size_across_bins_2d > 1 else 0
        x_padded = pad_channels(x, pad2d)
        x_conv_in = x_padded.unsqueeze(1)
        conv_out = F.relu(self.conv(x_conv_in).squeeze(2))  # (batch_size, num_filters_2d, num_bins)

        pad1d = self.kernel_size_across_bins_1d // 2 if self.kernel_size_across_bins_1d > 1 else 0
        h_padded = pad_channels(conv_out, pad1d)
        h = F.relu(self.conv1d_1(h_padded))

        h_padded = pad_channels(h, pad1d)
        out = self.conv1d_2(h_padded).mean(dim=1)  # (batch_size, num_bins)

        if self.is_cum_sum:
            out = torch.flip(torch.cumsum(torch.flip(out, dims=[1]), dim=1), dims=[1])

        return out

    # class BinConv(nn.Module):
    #     def __init__(self, context_length: int, num_bins: int, kernel_size_across_bins_2d: int = 3,
    #                  kernel_size_across_bins_1d: int = 3, num_filters_2d: int = 8,
    #                  num_filters_1d: int = 32, is_cum_sum: bool = False) -> None:
    #         super().__init__()
    #         self.context_length = context_length
    #         self.num_bins = num_bins
    #         self.num_filters_2d = num_filters_2d
    #         self.num_filters_1d = num_filters_1d
    #         self.kernel_size_across_bins_2d = kernel_size_across_bins_2d
    #         self.kernel_size_across_bins_1d = kernel_size_across_bins_1d
    #         self.is_cum_sum = is_cum_sum
    #         # Convolution layer to collapse context_length dimension
    #         self.conv = nn.Conv2d(
    #             in_channels=1,
    #             out_channels=self.num_filters_2d,
    #             kernel_size=(context_length, kernel_size_across_bins_2d),
    #             bias=True
    #         )
    #
    #         # Shared MLP applied independently to each (8-dim) filter output vector
    #         # self.mlp = nn.Sequential(
    #         #     nn.Linear(num_bins, num_bins),
    #         #     nn.ReLU(),
    #         #     nn.Linear(num_bins, num_bins)
    #         # )
    #         self.conv1d_1 = nn.Conv1d(in_channels=self.num_filters_2d, out_channels=num_filters_1d, kernel_size=3,
    #                                   bias=True)
    #         self.conv1d_2 = nn.Conv1d(in_channels=4 * self.num_filters, out_channels=num_filters_1d, kernel_size=3,
    #                                   bias=True)
    #
    #     def forward(self, x):
    #         def pad_channels(tensor, pad_val_left=1.0, pad_val_right=0.0):
    #             left = torch.full((*tensor.shape[:-1], 1), pad_val_left, device=tensor.device)
    #             right = torch.full((*tensor.shape[:-1], 1), pad_val_right, device=tensor.device)
    #             return torch.cat([left, tensor, right], dim=-1)
    #
    #         # x: (batch_size, context_length, num_bins)
    #         batch_size, context_length, num_bins = x.shape
    #         assert context_length == self.context_length, "Mismatch in context length"
    #
    #         x_padded = pad_channels(x)
    #         x_conv_in = x_padded.unsqueeze(1)
    #         conv_out = F.relu(self.conv(x_conv_in).squeeze(2))  # (batch_size, num_filters, num_bins)
    #
    #         h_padded = pad_channels(conv_out)
    #         h = F.relu(self.conv1d_1(h_padded))
    #
    #         h_padded = pad_channels(h)
    #         out = self.conv1d_2(h_padded).mean(dim=1)  # (batch_size, num_bins)
    #
    #         if self.is_cum_sum:
    #             out = torch.flip(torch.cumsum(torch.flip(out, dims=[1]), dim=1), dims=[1])
    #
    #         return out
    # def forward(self, x):
    #     # x: (batch_size, context_length, num_bins)
    #     batch_size, context_length, num_bins = x.shape
    #     assert context_length == self.context_length, "Mismatch in context length"
    #
    #     # Left padding with ones, right padding with zeros
    #     pad_left = torch.ones(batch_size, context_length, 1, device=x.device)
    #     pad_right = torch.zeros(batch_size, context_length, 1, device=x.device)
    #     x_padded = torch.cat([pad_left, x, pad_right], dim=2)  # (batch_size, context_length+2, num_bins)
    #
    #     # Reshape for Conv2d: (batch_size, 1, context_length+2, num_bins)
    #     x_conv_in = x_padded.unsqueeze(1)
    #     # Apply convolution → (batch_size, 8, 1, num_bins)
    #     conv_out = F.relu(self.conv(x_conv_in).squeeze(2))  # → (batch_size, 8, num_bins)
    #     # ----
    #
    #     pad_left = torch.ones(batch_size, self.num_filters_2d, 1, device=x.device)
    #     pad_right = torch.zeros(batch_size, self.num_filters_2d, 1, device=x.device)
    #     h_padded = torch.cat([pad_left, conv_out, pad_right], dim=2)  # (batch_size, context_length+2, num_bins)
    #     h = F.relu(self.conv1d_1(h_padded))
    #     pad_left = torch.ones(batch_size, self.num_filters_1d, 1, device=x.device)
    #     pad_right = torch.zeros(batch_size, self.num_filters_1d, 1, device=x.device)
    #     h_padded = torch.cat([pad_left, h, pad_right], dim=2)  # (batch_size, context_length+2, num_bins)
    #     out = self.conv1d_2(h_padded).mean(dim=1)
    #     if self.is_cum_sum:
    #         # Apply reverse cumsum for monotonic decreasing
    #         out = torch.flip(out, dims=[1])  # flip along bins axis
    #         out = torch.cumsum(out, dim=1)
    #         out = torch.flip(out, dims=[1])
    #     return out

    # def forward(self, x):
    #     # x: (batch_size, context_length, num_bins)
    #     batch_size, context_length, num_bins = x.shape
    #     assert context_length == self.context_length, "Mismatch in context length"
    #
    #     # Left padding with ones, right padding with zeros
    #     pad_left = torch.ones(batch_size, context_length, 1, device=x.device)
    #     pad_right = torch.zeros(batch_size, context_length, 1, device=x.device)
    #     x_padded = torch.cat([pad_left, x, pad_right], dim=2)  # (batch_size, context_length+2, num_bins)
    #
    #     # Reshape for Conv2d: (batch_size, 1, context_length+2, num_bins)
    #     x_conv_in = x_padded.unsqueeze(1)
    #     # Apply convolution → (batch_size, 8, 1, num_bins)
    #     conv_out = F.relu(self.conv(x_conv_in).squeeze(2))  # → (batch_size, 8, num_bins)

    ## Prepare for MLP: flatten batch and num_bins dimensions
    # conv_out_flat = conv_out.reshape(-1, self.num_bins)  # (batch_size * 8, num_bins)
    #
    # # Apply shared MLP
    # mlp_out = self.mlp(conv_out_flat)  # (batch_size * 8, num_bins)
    #
    # # Reshape back to (batch_size, 8, num_bins)
    # mlp_out = mlp_out.view(batch_size, self.num_filters, self.num_bins)
    #
    # # Average over filters (i.e., average over num_bins dimension)
    # output = mlp_out.mean(dim=1)  # → (batch_size, num_bins)
    #
    # return output

    @torch.inference_mode()
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
        for _ in range(prediction_length):
            pred = F.sigmoid(self(current_context))  # (B, D)
            pred = (pred >= 0.5).int()
            forecasts.append(pred.unsqueeze(1))  # (B, 1, D)
            next_input = pred.unsqueeze(1)
            current_context = torch.cat([current_context[:, 1:], next_input], dim=1)

        return torch.cat(forecasts, dim=1)  # (B, T, D)


class LightningBinConv(BinConv, pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer
