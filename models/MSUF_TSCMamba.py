import copy

import torch
import torch.nn.functional as F
from mamba_ssm import Mamba


class SeriesDecomp(torch.nn.Module):
    def __init__(self, moving_avg=3):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = moving_avg

    def forward(self, x):
        seq_len = x.size(1)
        kernel_size = min(self.moving_avg, seq_len)
        if kernel_size <= 1:
            trend = x
        else:
            left_pad = (kernel_size - 1) // 2
            right_pad = kernel_size - 1 - left_pad
            x_channels_first = x.permute(0, 2, 1)
            padded = F.pad(x_channels_first, (left_pad, right_pad), mode="replicate")
            trend = F.avg_pool1d(padded, kernel_size=kernel_size, stride=1).permute(0, 2, 1)

        seasonal = x - trend
        return seasonal, trend


class TemporalMLP(torch.nn.Module):
    def __init__(self, l_in, l_out):
        super(TemporalMLP, self).__init__()
        self.l_in = l_in
        self.l_out = l_out
        self.temporal_projection = torch.nn.Sequential(
            torch.nn.Linear(l_in, l_out),
            torch.nn.GELU(),
            torch.nn.Linear(l_out, l_out),
        )

    def forward(self, x):
        if x.size(1) != self.l_in:
            raise ValueError(
                f"TemporalMLP expected length {self.l_in}, got {x.size(1)}"
            )
        x = x.permute(0, 2, 1)
        x = self.temporal_projection(x)
        return x.permute(0, 2, 1)


class PDMBlock(torch.nn.Module):
    def __init__(self, d_model, lengths, dropout=0.0, moving_avg=3):
        super(PDMBlock, self).__init__()
        l0, l1, l2 = lengths

        self.decomp = SeriesDecomp(moving_avg=moving_avg)
        self.bu_0_to_1 = TemporalMLP(l0, l1)
        self.bu_1_to_2 = TemporalMLP(l1, l2)
        self.td_2_to_1 = TemporalMLP(l2, l1)
        self.td_1_to_0 = TemporalMLP(l1, l0)

        self.ffn0 = self._build_ffn(d_model, dropout)
        self.ffn1 = self._build_ffn(d_model, dropout)
        self.ffn2 = self._build_ffn(d_model, dropout)

        self.norm0 = torch.nn.LayerNorm(d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)

    def _build_ffn(self, d_model, dropout):
        return torch.nn.Sequential(
            torch.nn.Linear(d_model, 4 * d_model),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(4 * d_model, d_model),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x_list):
        x0, x1, x2 = x_list

        s0, t0 = self.decomp(x0)
        s1, t1 = self.decomp(x1)
        s2, t2 = self.decomp(x2)

        s1 = s1 + self.bu_0_to_1(s0)
        s2 = s2 + self.bu_1_to_2(s1)

        t1 = t1 + self.td_2_to_1(t2)
        t0 = t0 + self.td_1_to_0(t1)

        m0 = s0 + t0
        m1 = s1 + t1
        m2 = s2 + t2

        out0 = self.norm0(x0 + self.ffn0(m0))
        out1 = self.norm1(x1 + self.ffn1(m1))
        out2 = self.norm2(x2 + self.ffn2(m2))

        return [out0, out1, out2]


class MultiScaleUpsamplingFusion(torch.nn.Module):
    def __init__(self, input_dim, d_model, seq_len, dropout=0.0, moving_avg=3, pdm_layers=1):
        super(MultiScaleUpsamplingFusion, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len
        self.l0 = seq_len
        self.l1 = (self.l0 + 1) // 2
        self.l2 = (self.l1 + 1) // 2

        self.input_projection = torch.nn.Linear(input_dim, d_model)
        self.pdm_blocks = torch.nn.ModuleList([
            PDMBlock(
                d_model=d_model,
                lengths=(self.l0, self.l1, self.l2),
                dropout=dropout,
                moving_avg=moving_avg,
            )
            for _ in range(pdm_layers)
        ])
        self.output_projection = torch.nn.Linear(3 * d_model, d_model)
        self.output_norm = torch.nn.LayerNorm(d_model)
        self.output_dropout = torch.nn.Dropout(dropout)

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
        x0_blc = self._to_batch_length_channel(x)
        if x0_blc.size(1) != self.seq_len:
            raise ValueError(
                "MSUF received a sequence length that does not match configs.seq_len. "
                f"Expected {self.seq_len}, got {x0_blc.size(1)}"
            )

        x0_raw = x0_blc.permute(0, 2, 1)
        x1_raw = F.avg_pool1d(x0_raw, kernel_size=2, stride=2, ceil_mode=True)
        x2_raw = F.avg_pool1d(x1_raw, kernel_size=2, stride=2, ceil_mode=True)

        x0 = self.input_projection(x0_raw.permute(0, 2, 1))
        x1 = self.input_projection(x1_raw.permute(0, 2, 1))
        x2 = self.input_projection(x2_raw.permute(0, 2, 1))

        x_list = [x0, x1, x2]
        for block in self.pdm_blocks:
            x_list = block(x_list)
        out0, out1, out2 = x_list

        up_out1 = F.interpolate(
            out1.permute(0, 2, 1),
            size=self.l0,
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)
        up_out2 = F.interpolate(
            out2.permute(0, 2, 1),
            size=self.l0,
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)

        fused = torch.cat([out0, up_out1, up_out2], dim=-1)
        ms_feature = self.output_projection(fused)
        return self.output_dropout(self.output_norm(ms_feature))


class Model(torch.nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.configs_copy = copy.deepcopy(self.configs)

        if self.configs.task_name == "classification":
            self.msuf = MultiScaleUpsamplingFusion(
                input_dim=self.configs.enc_in,
                d_model=self.configs.d_model,
                seq_len=self.configs.seq_len,
                dropout=self.configs.dropout,
                moving_avg=getattr(self.configs, "moving_avg", 3),
                pdm_layers=getattr(self.configs, "pdm_layers", 1),
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
