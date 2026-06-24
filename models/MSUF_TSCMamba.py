import copy

import torch
import torch.nn.functional as F
from mamba_ssm import Mamba


class MultiScaleUpsamplingFusion(torch.nn.Module):
    def __init__(self, input_dim, d_model):
        super(MultiScaleUpsamplingFusion, self).__init__()
        self.input_dim = input_dim
        self.projection = torch.nn.Linear(input_dim * 3, d_model)

    def _to_batch_length_channel(self, x):
        if x.dim() != 3:
            raise ValueError(f"MSUF expects a 3D raw input tensor, got shape {tuple(x.shape)}")

        if x.size(1) == self.input_dim:
            return x.permute(0, 2, 1)
        if x.size(2) == self.input_dim:
            return x

        raise ValueError(
            "MSUF could not infer raw input layout. "
            f"Expected channel dimension {self.input_dim}, got shape {tuple(x.shape)}"
        )

    def forward(self, x):
        x0 = self._to_batch_length_channel(x)
        seq_len = x0.size(1)

        x0_channels_first = x0.permute(0, 2, 1)
        x1 = F.avg_pool1d(x0_channels_first, kernel_size=2, stride=2, ceil_mode=True)
        x2 = F.avg_pool1d(x1, kernel_size=2, stride=2, ceil_mode=True)

        up_x1 = F.interpolate(x1, size=seq_len, mode="linear", align_corners=False).permute(0, 2, 1)
        up_x2 = F.interpolate(x2, size=seq_len, mode="linear", align_corners=False).permute(0, 2, 1)

        fused = torch.cat([x0, up_x1, up_x2], dim=2)
        return self.projection(fused)


class Model(torch.nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.configs_copy = copy.deepcopy(self.configs)

        if self.configs.task_name == "classification":
            self.msuf = MultiScaleUpsamplingFusion(
                input_dim=self.configs.enc_in,
                d_model=self.configs.d_model,
            )
            self.ln = torch.nn.LayerNorm(self.configs.d_model)
            self.dropout = torch.nn.Dropout(self.configs.dropout)

            self.mamba1 = torch.nn.ModuleList([
                Mamba(
                    d_model=self.configs.d_model,
                    d_state=self.configs.d_state,
                    d_conv=self.configs.dconv,
                    expand=self.configs.e_fact,
                ) for _ in range(self.configs.num_mambas)
            ])
            self.mamba2 = torch.nn.ModuleList([
                Mamba(
                    d_model=self.configs.seq_len,
                    d_state=self.configs.d_state,
                    d_conv=self.configs.dconv,
                    expand=self.configs.e_fact,
                ) for _ in range(self.configs.num_mambas)
            ])

            self.flatten = torch.nn.Flatten(start_dim=1)
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(self.configs.d_model, self.configs.d_model // 2),
                torch.nn.Dropout(self.configs.dropout),
                torch.nn.Linear(self.configs.d_model // 2, self.configs.num_class),
            )

    def _flip(self, x):
        flip_dir = self.configs.flip_dir
        if flip_dir >= x.dim():
            flip_dir = 1
        return torch.flip(x, dims=[flip_dir])

    def _reverse_merge(self, forward_x, reverse_x):
        if self.configs.reverse_flip == 0:
            return forward_x + reverse_x
        if self.configs.reverse_flip == 1:
            return forward_x + self._flip(reverse_x)
        return forward_x + reverse_x

    def classification(self, x_cwt, batch_x_rocket, batch_x_raw):
        ms_feature = self.msuf(batch_x_raw)
        ms_feature = self.dropout(self.ln(ms_feature))

        if self.configs.num_mambas != 0:
            x1 = ms_feature.clone()
            for i in range(self.configs.num_mambas):
                x1 = self.mamba1[i](x1) + x1.clone()

            if self.configs.only_forward_scan == 0:
                ms_feature_flipped = self._flip(ms_feature)
                x1_flipped = ms_feature_flipped.clone()
                for i in range(self.configs.num_mambas):
                    x1_flipped = self.mamba1[i](x1_flipped) + x1_flipped.clone()
                x1 = self._reverse_merge(x1, x1_flipped)

            channel_axis_feature = ms_feature.permute(0, 2, 1)
            x2 = channel_axis_feature.clone()
            for i in range(self.configs.num_mambas):
                x2 = self.mamba2[i](x2) + x2.clone()
            x2 = x2.permute(0, 2, 1)

            if self.configs.only_forward_scan == 0:
                x2_flipped = self._flip(channel_axis_feature.clone())
                for i in range(self.configs.num_mambas):
                    x2_flipped = self.mamba2[i](x2_flipped) + x2_flipped.clone()
                x2_flipped = x2_flipped.permute(0, 2, 1)
                x2 = self._reverse_merge(x2, x2_flipped)

            x3 = x1 + x2
        else:
            x3 = ms_feature

        if self.configs.max_pooling == 0:
            x3 = x3.mean(1)
        else:
            x3, _ = x3.max(1)

        x3 = self.flatten(x3)
        return self.classifier(x3)

    def forward(self, x_cwt, x_rocket, x_raw):
        if self.configs.task_name == "classification":
            return self.classification(x_cwt, x_rocket, x_raw)
        raise NotImplementedError("MSUF_TSCMamba currently supports classification only.")
