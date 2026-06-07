# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import math
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange, repeat
from timm.models.layers import DropPath
from models.csms6s import SelectiveScanMamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

# Import ODE functionality
try:
    from torchdiffeq import odeint
    ODE_AVAILABLE = True
except ImportError:
    ODE_AVAILABLE = False
    print("Warning: torchdiffeq not available. ODE functionality will be disabled.")


MODEL_PATH = 'your_model_path'
_MODELS = {
    "videomamba_t16_in1k": os.path.join(MODEL_PATH, "videomamba_t16_in1k_res224.pth"),
    "videomamba_s16_in1k": os.path.join(MODEL_PATH, "videomamba_s16_in1k_res224.pth"),
    "videomamba_m16_in1k": os.path.join(MODEL_PATH, "videomamba_m16_in1k_res224.pth"),
}


class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# ===================== ODE Components =====================
class ODEFunc1D(nn.Module):
    """Spectral ODE Function"""
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, in_channels, 1),
            nn.BatchNorm1d(in_channels),
        )
        # 添加时间调制参数
        self.time_scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, t, x):
        # 如果t是None，说明是fallback模式，直接执行纯CNN
        if t is None:
            return self.net(x)
            
        # 获取时间值
        if isinstance(t, torch.Tensor):
            t_val = t.item() if t.numel() == 1 else t.mean().item()
        else:
            t_val = float(t)
        
        # 基础网络输出
        base_output = self.net(x)
        
        # 时间调制：让输出随时间变化
        time_factor = 1.0 + self.time_scale * torch.tanh(torch.tensor(t_val, device=x.device, dtype=x.dtype))
        
        return base_output * time_factor


class ODEExpert(nn.Module):
    """Spectral ODE Expert (1D only)"""
    def __init__(self, dim, hidden_dim=None, integration_time=1.0, method="euler"):
        super().__init__()
        hidden_dim = hidden_dim or dim // 2
        self.integration_time = integration_time
        self.method = method
        self.spectral_odefunc = ODEFunc1D(dim, hidden_dim)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        x_spec = x.view(B, C, -1)  # → [B, C, H*W]
        
        if not ODE_AVAILABLE:
            x_spec = self.spectral_odefunc(None, x_spec)
        else:
            t = torch.tensor([0.0, self.integration_time], dtype=x.dtype, device=x.device)
            try:
                x_spec = odeint(self.spectral_odefunc, x_spec, t, method=self.method, rtol=1e-3, atol=1e-4)[-1]
            except Exception as e:
                if "out of memory" in str(e).lower() or "must be strictly increasing" in str(e):
                    print(f"ODE failed ({type(e).__name__}: {e}), falling back to convolution...")
                    x_spec = self.spectral_odefunc(None, x_spec)
                else:
                    raise e
        
        return x_spec.view(B, C, H, W)


class SequenceToSpatialAdapter(nn.Module):
    """将序列格式转换为空间格式以适配ODE Expert"""
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
    def forward(self, x, target_h, target_w):
        """
        Args:
            x: [B, seq_len, embed_dim] - 序列格式
            target_h, target_w: 目标空间尺寸
        Returns:
            [B, embed_dim, target_h, target_w] - 空间格式
        """
        B, seq_len, embed_dim = x.shape
        
        # 检查序列长度是否匹配空间尺寸
        if seq_len != target_h * target_w:
            # 如果不匹配，使用插值
            x_2d = x.transpose(1, 2)  # [B, embed_dim, seq_len]
            x_interp = nn.functional.interpolate(
                x_2d, size=target_h * target_w, mode='linear', align_corners=False
            )
            x = x_interp.transpose(1, 2)  # [B, target_h*target_w, embed_dim]
        
        # 重塑为空间格式
        x_spatial = x.transpose(1, 2).view(B, embed_dim, target_h, target_w)
        return x_spatial
    
    def reverse(self, x):
        """
        Args:
            x: [B, embed_dim, H, W] - 空间格式
        Returns:
            [B, H*W, embed_dim] - 序列格式
        """
        B, embed_dim, H, W = x.shape
        x_seq = x.view(B, embed_dim, -1).transpose(1, 2)  # [B, H*W, embed_dim]
        return x_seq

class SequenceODEBlock(nn.Module):
    """序列→空间→Spectral ODE(1D)→空间→序列 (适配Mamba序列格式)"""
    def __init__(self, embed_dim, hidden_dim=None, integration_time=1, method="euler", dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ode_expert = ODEExpert(embed_dim, hidden_dim, integration_time, method)
        self.adapter = SequenceToSpatialAdapter(embed_dim)

    def forward(self, x, spatial_size=None):
        B, seq_len, embed_dim = x.shape
        
        # --- 推断空间尺寸 ---
        if spatial_size is None:
            h = w = int(seq_len ** 0.5)
            if h * w != seq_len:
                h = int(math.sqrt(seq_len))
                w = max(1, seq_len // h)
                if h * w != seq_len:
                    h = w = int(math.ceil(seq_len ** 0.5))
        else:
            h, w = spatial_size
            
        # --- 序列 → 空间 ---
        x_spatial = self.adapter(x, h, w)   # [B, C, H, W]

        # --- ODE ---
        x_ode = self.ode_expert(x_spatial)  # [B, C, H, W]

        # --- 空间 → 序列 ---
        x_out = self.adapter.reverse(x_ode) # [B, H*W, C]

        # --- 处理长度不一致 ---
        if x_out.shape[1] != seq_len:
            x_out = x_out[:, :seq_len, :]
            if x_out.shape[1] < seq_len:
                pad = seq_len - x_out.shape[1]
                x_out = torch.cat([x_out, x_out.new_zeros(B, pad, embed_dim)], dim=1)

        return x_out




# ===================== Original Components (kept for compatibility) =====================
class Block(nn.Module, mamba_init):
    def __init__(self,
                 scan_type=None,
                 group_type = None,
                 k_group = None,
                 dim=None,
                 dt_rank="auto",
                 d_state = None,
                 d_model = None,
                 ssm_ratio = None,
                 bimamba=None,
                 seq=False,
                 force_fp32=True,
                 dropout=0.0,
                 **kwargs):
        super().__init__()
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        self.force_fp32 = force_fp32
        self.seq = seq
        self.k_group = k_group
        self.group_type = group_type
        self.scan_type = scan_type
        self.dim = dim
        self.d_model = d_model
        self.d_state = d_state
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # in proj ============================
        self.in_proj = nn.Linear(dim, d_inner * 2, bias=bias, **kwargs)
        self.act: nn.Module = act_layer()
        self.forward_conv1d = nn.Conv1d(
            in_channels=d_inner, out_channels=d_inner, kernel_size=1
        )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **kwargs)
            for _ in range(k_group)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=k_group, merge=True)  # (K * D)

        # out proj =======================================
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, dim, bias=bias, **kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def scan(self, x, scan_type=None, group_type=None, route=None):
        if group_type == 'Patch':
            x_hwwh = torch.stack([x.view(self.B, -1, self.L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(self.B, -1, self.L)], dim=1).view(self.B, 2, -1, self.L)
            xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        elif group_type == 'Linear':
            xs = torch.stack([x, torch.flip(x, dims=[-1])], dim=1)  # [B, 2, D, L]
        else:
            xs = torch.stack([x, torch.flip(x, dims=[-1])], dim=1)  # [B, 2, D, L]
        return xs

    def forward(self, x: Tensor, route=None, SelectiveScan = SelectiveScanMamba):
        x = self.in_proj(x)  # [B, L, d_in * 2]
        x, z = x.chunk(2, dim=-1)  # [B, L, d_in]
        z = self.act(z)

        # forward conv1d
        if self.group_type == 'Linear':
            x1_rearranged = rearrange(x, "b s d -> b d s").contiguous()  # [B, d_in, L]
            x = self.forward_conv1d(x1_rearranged)  # [B, d_in, L]
            x = self.act(x)

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, False)

        if len(x.size()) == 3:
            B, D_inner, L = x.shape  # [B, d_inner, L]
        else:
            raise ValueError(f"Expected 3D tensor, got {len(x.size())}D")
        self.B = B
        self.L = L
        D_total, N = self.A_logs.shape  # D_total = k_group * d_inner, N = d_state
        K, D_proj, R = self.dt_projs_weight.shape  # K = k_group, D_proj = d_inner, R = dt_rank

        # scan
        xs = self.scan(x, scan_type=self.scan_type, group_type=self.group_type, route=route)  # [B, K, D, L]

        # x_proj: 计算 delta, B, C
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)  # [B, K, dt_rank+d_state*2, L]
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)  # dts: [B, K, R, L], Bs/Cs: [B, K, N, L]
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)  # [B, K, d_inner, L]

        xs = xs.view(B, -1, L)  # [B, K*d_inner, L]
        dts = dts.contiguous().view(B, -1, L)  # [B, K*d_inner, L]
        Bs = Bs.contiguous()  # [B, K, N, L]
        Cs = Cs.contiguous()  # [B, K, N, L]

        As = -torch.exp(self.A_logs.float())   # [K*d_inner, N]
        Ds = self.Ds.float()  # [K*d_inner]
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # [K*d_inner]

        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        if self.force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        # 执行 SelectiveScan
        if self.seq:
            out_y = []
            for i in range(self.k_group):
                yi = selective_scan(
                    xs.view(B, K, -1, L)[:, i], dts.view(B, K, -1, L)[:, i],
                    As.view(K, -1, N)[i], Bs[:, i].unsqueeze(1), Cs[:, i].unsqueeze(1), Ds.view(K, -1)[i],
                    delta_bias=dt_projs_bias.view(K, -1)[i],
                    delta_softplus=True,
                ).view(B, -1, L)
                out_y.append(yi)
            out_y = torch.stack(out_y, dim=1)  # [B, K, d_inner, L]
        else:
            out_y = selective_scan(
                xs, dts,
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)  # [B, K, d_inner, L]
        
        assert out_y.dtype == torch.float

        # 合并方向
        if out_y.size(1) == 2:
            y = out_y[:, 0] + torch.flip(out_y[:, 1], dims=[-1])  # [B, d_inner, L]
            y = y.transpose(dim0=1, dim1=2).contiguous()  # [B, L, d_inner]
            y = self.out_norm(y)  # [B, L, d_inner]

        y = y * z  # [B, L, d_inner]
        out = self.dropout(self.out_proj(y))  # [B, L, dim]
        return out

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return


class MLP_Block(nn.Module):
    def __init__(self, in_features, hidden_features, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ===================== Cross-scale Fusion =====================
class CrossScaleFusion(nn.Module):
    """跨尺度融合：让小尺度接受大尺度的信息"""
    def __init__(self, embed_dim, reduction=4):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 大尺度特征投影
        self.large_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, embed_dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // reduction, embed_dim, 1),
            nn.Sigmoid()
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x_small, x_large):
        """
        Args:
            x_small: [B, C, H_s, W_s] - 小尺度特征
            x_large: [B, C, H_l, W_l] - 大尺度特征
        Returns:
            [B, C, H_s, W_s] - 融合后的特征
        """
        x_large_down = torch.nn.functional.interpolate(
            x_large, size=x_small.shape[2:], mode='bilinear', align_corners=False
        )
        large_global = self.large_proj(x_large_down)  # [B, C, 1, 1]
        x_concat = torch.cat([x_small, x_large_down], dim=1)
        x_fused = self.fusion(x_concat)
        
        # 加权融合
        alpha = torch.sigmoid(self.fusion_weight)
        x_out = alpha * x_fused + (1 - alpha) * (x_small + large_global)
        
        return x_out


class MultiScaleMambaBlock(nn.Module):
    """多尺度Mamba Block: 每个分支独立处理 + 跨尺度融合"""
    def __init__(
        self,
        embed_dim,
        d_state=16,
        d_model=None,
        ssm_ratio=1,
        bimamba=True,
        drop_path=0.1,
        use_cross_scale=True,
        **kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_cross_scale = use_cross_scale
        d_model = d_model or embed_dim
        
        # 每个分支独立的Mamba Block
        self.mamba_1 = Block(
            group_type='Linear',
            k_group=2,
            dim=embed_dim,
            d_state=d_state,
            d_model=d_model,
            ssm_ratio=ssm_ratio,
            bimamba=bimamba,
            **kwargs
        )
        self.mamba_2 = Block(
            group_type='Linear',
            k_group=2,
            dim=embed_dim,
            d_state=d_state,
            d_model=d_model,
            ssm_ratio=ssm_ratio,
            bimamba=bimamba,
            **kwargs
        )
        self.mamba_3 = Block(
            group_type='Linear',
            k_group=2,
            dim=embed_dim,
            d_state=d_state,
            d_model=d_model,
            ssm_ratio=ssm_ratio,
            bimamba=bimamba,
            **kwargs
        )
        
        # 跨尺度融合模块
        if use_cross_scale:
            self.cross_scale_fusion_2to1 = CrossScaleFusion(embed_dim, reduction=4)
            self.cross_scale_fusion_3to1 = CrossScaleFusion(embed_dim, reduction=4)
            self.cross_scale_fusion_3to2 = CrossScaleFusion(embed_dim, reduction=4)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, LG_1, LG_2, LG_3, spatial_sizes):
        """
        Args:
            LG_1: [B, n1, C] - 分支1特征
            LG_2: [B, n2, C] - 分支2特征
            LG_3: [B, n3, C] - 分支3特征
            spatial_sizes: [(H1,W1), (H2,W2), (H3,W3)]
        Returns:
            LG_1, LG_2, LG_3: 增强后的特征
        """
        # 1. 每个分支独立Mamba处理
        LG_1_mamba = self.mamba_1(self.norm(LG_1))  # [B, n1, C]
        LG_2_mamba = self.mamba_2(self.norm(LG_2))  # [B, n2, C]
        LG_3_mamba = self.mamba_3(self.norm(LG_3))  # [B, n3, C]
        
        # 2. 转换为空间格式进行跨尺度融合
        B = LG_1_mamba.shape[0]
        H1, W1 = spatial_sizes[0]
        H2, W2 = spatial_sizes[1]
        H3, W3 = spatial_sizes[2]
        
        # 确保尺寸匹配
        if LG_1_mamba.shape[1] != H1 * W1:
            # Padding或截断
            target_len = H1 * W1
            if LG_1_mamba.shape[1] < target_len:
                pad_size = target_len - LG_1_mamba.shape[1]
                LG_1_mamba = torch.cat([
                    LG_1_mamba,
                    torch.zeros(B, pad_size, self.embed_dim, device=LG_1_mamba.device, dtype=LG_1_mamba.dtype)
                ], dim=1)
            else:
                LG_1_mamba = LG_1_mamba[:, :target_len, :]
        
        if LG_2_mamba.shape[1] != H2 * W2:
            target_len = H2 * W2
            if LG_2_mamba.shape[1] < target_len:
                pad_size = target_len - LG_2_mamba.shape[1]
                LG_2_mamba = torch.cat([
                    LG_2_mamba,
                    torch.zeros(B, pad_size, self.embed_dim, device=LG_2_mamba.device, dtype=LG_2_mamba.dtype)
                ], dim=1)
            else:
                LG_2_mamba = LG_2_mamba[:, :target_len, :]
        
        if LG_3_mamba.shape[1] != H3 * W3:
            target_len = H3 * W3
            if LG_3_mamba.shape[1] < target_len:
                pad_size = target_len - LG_3_mamba.shape[1]
                LG_3_mamba = torch.cat([
                    LG_3_mamba,
                    torch.zeros(B, pad_size, self.embed_dim, device=LG_3_mamba.device, dtype=LG_3_mamba.dtype)
                ], dim=1)
            else:
                LG_3_mamba = LG_3_mamba[:, :target_len, :]
        
        LG_1_spatial = LG_1_mamba.view(B, H1, W1, self.embed_dim).permute(0, 3, 1, 2)  # [B, C, H1, W1]
        LG_2_spatial = LG_2_mamba.view(B, H2, W2, self.embed_dim).permute(0, 3, 1, 2)  # [B, C, H2, W2]
        LG_3_spatial = LG_3_mamba.view(B, H3, W3, self.embed_dim).permute(0, 3, 1, 2)  # [B, C, H3, W3]
        
        # 3. 跨尺度融合：小尺度接受大尺度信息
        if self.use_cross_scale:
            # 分支2接受分支1的信息
            LG_2_enhanced = self.cross_scale_fusion_2to1(LG_2_spatial, LG_1_spatial)
            
            # 分支3接受分支1和分支2的信息
            LG_3_from1 = self.cross_scale_fusion_3to1(LG_3_spatial, LG_1_spatial)
            LG_3_from2 = self.cross_scale_fusion_3to2(LG_3_spatial, LG_2_spatial)
            # 融合分支1和分支2的信息
            LG_3_enhanced = 0.5 * LG_3_from1 + 0.5 * LG_3_from2 + LG_3_spatial
        else:
            LG_2_enhanced = LG_2_spatial
            LG_3_enhanced = LG_3_spatial
        
        # 5. 转换回序列格式
        LG_1_final = LG_1_spatial.permute(0, 2, 3, 1).view(B, H1 * W1, self.embed_dim)
        LG_2_final = LG_2_enhanced.permute(0, 2, 3, 1).view(B, H2 * W2, self.embed_dim)
        LG_3_final = LG_3_enhanced.permute(0, 2, 3, 1).view(B, H3 * W3, self.embed_dim)
        
        # 6. 残差连接
        # 恢复原始长度
        n1_orig = LG_1.shape[1]
        n2_orig = LG_2.shape[1]
        n3_orig = LG_3.shape[1]
        
        LG_1_out = LG_1 + self.drop_path(LG_1_final[:, :n1_orig, :])
        LG_2_out = LG_2 + self.drop_path(LG_2_final[:, :n2_orig, :])
        LG_3_out = LG_3 + self.drop_path(LG_3_final[:, :n3_orig, :])
        
        return LG_1_out, LG_2_out, LG_3_out


class VisionMambaODE(nn.Module):
    """VisionMamba with ODE: Mamba + Spectral ODE 多尺度结构，无 Transformer"""
    def __init__(
            self,
            k_group=None,
            depth=None,
            embed_dim=None,
            d_state: int = None,
            ssm_ratio: int = None,
            num_classes: int = None,
            drop_rate=0.,
            drop_path_rate=0.1,
            residual_in_fp32=True,
            bimamba=True,
            # video
            fc_drop_rate=0.,
            Pos_Cls = False,
            pos: str = None,
            cls: str = None,
            conv3D_channel: int = None,
            conv3D_kernel_1: int = None,
            conv3D_kernel_2: int = None,
            conv3D_kernel_3: int = None,
            dim_patch: int = None,
            dim_linear_1: int = None,
            dim_linear_2: int = None,
            dim_linear_3: int = None,
            # ODE specific parameters
            ode_integration_time: float = 1.0,
            ode_method: str = "euler",
            ode_hidden_dim: int = None,
            **kwargs,
        ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.Pos_Cls = Pos_Cls
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  
        self.k_group = k_group
        self.depth = depth

        # 3D Conv features (unchanged)
        self.conv3d_features_1 = nn.Sequential(
            nn.Conv3d(1, out_channels=conv3D_channel, kernel_size=conv3D_kernel_1),
            nn.BatchNorm3d(conv3D_channel),
            nn.ReLU(),
        )
        self.conv3d_features_2 = nn.Sequential(
            nn.Conv3d(1, out_channels=conv3D_channel, kernel_size=conv3D_kernel_2),
            nn.BatchNorm3d(conv3D_channel),
            nn.ReLU(),
        )
        self.conv3d_features_3 = nn.Sequential(
            nn.Conv3d(1, out_channels=conv3D_channel, kernel_size=conv3D_kernel_3),
            nn.BatchNorm3d(conv3D_channel),
            nn.ReLU(),
        )

        self.embedding_spatial_1 = nn.LazyConv2d(embed_dim, kernel_size=1)
        self.embedding_spatial_2 = nn.LazyConv2d(embed_dim, kernel_size=1)
        self.embedding_spatial_3 = nn.LazyConv2d(embed_dim, kernel_size=1)


        self.norm = nn.LayerNorm(embed_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Classification head (unchanged)
        self.head_drop = nn.Dropout(fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        # ODE blocks
        ode_hidden_dim = ode_hidden_dim or embed_dim // 2
        self.ode_1 = nn.ModuleList([
            Residual(LayerNormalize(embed_dim, 
                SequenceODEBlock(
                    embed_dim, 
                    hidden_dim=ode_hidden_dim,
                    integration_time=ode_integration_time, 
                    method=ode_method,
                    dropout=drop_path_rate,
                )
            ))
            for i in range(depth)
        ])
        self.ode_2 = nn.ModuleList([
            Residual(LayerNormalize(embed_dim, 
                SequenceODEBlock(
                    embed_dim, 
                    hidden_dim=ode_hidden_dim,
                    integration_time=ode_integration_time, 
                    method=ode_method,
                    dropout=drop_path_rate,
                )
            ))
            for i in range(depth)
        ])
        self.ode_3 = nn.ModuleList([
            Residual(LayerNormalize(embed_dim, 
                SequenceODEBlock(
                    embed_dim, 
                    hidden_dim=ode_hidden_dim,
                    integration_time=ode_integration_time, 
                    method=ode_method,
                    dropout=drop_path_rate,
                )
            ))
            for i in range(depth)
        ])

        # FFN (unchanged)
        self.FFN = nn.ModuleList([Residual(
                LayerNormalize(
                embed_dim, MLP_Block(embed_dim, embed_dim, dropout=drop_path_rate)))
                for i in range(depth)])

        # Mamba blocks (unchanged)
        self.layers = nn.ModuleList([Block(
                group_type='Linear',
                k_group=2,
                dim=embed_dim,
                d_state=d_state,
                d_model=embed_dim,
                ssm_ratio=ssm_ratio,
                bimamba=bimamba,
                **kwargs, )
                for i in range(depth)])
        
        # Multi-scale Mamba blocks (for Enhanced Interval MT)
        # 每个分支独立的Mamba处理 + 跨尺度融合
        self.multi_scale_mamba = nn.ModuleList([
            MultiScaleMambaBlock(
                embed_dim=embed_dim,
                d_state=d_state,
                d_model=embed_dim,
                ssm_ratio=ssm_ratio,
                bimamba=bimamba,
                drop_path=drop_path_rate,
                use_cross_scale=True,
                **kwargs
            )
            for i in range(depth)
        ])

    def get_num_layers(self):
        return len(self.layers)

    def scan(self, x, scan_type=None, group_type=None):
        x = rearrange(x, 'b c t h w -> b (c t) h w')  
        x = rearrange(x, 'b c h w -> b h w c')  
        return x

    def forward_features(self, x, inference_params=None):

        # ---------- 3D Conv ----------
        x_1 = self.conv3d_features_1(x)
        x_2 = self.conv3d_features_2(x)
        x_3 = self.conv3d_features_3(x)

        # ---------- Scan ----------
        x_1 = self.scan(x_1)   # [B, H, W, C]
        x_2 = self.scan(x_2)
        x_3 = self.scan(x_3)

        # ---------- Conv embedding ----------
        x_1 = self.embedding_spatial_1(x_1.permute(0,3,1,2)).permute(0,2,3,1)
        x_2 = self.embedding_spatial_2(x_2.permute(0,3,1,2)).permute(0,2,3,1)
        x_3 = self.embedding_spatial_3(x_3.permute(0,3,1,2)).permute(0,2,3,1)

        # ---------- Positional dropout ----------
        x_1 = self.pos_drop(x_1)
        x_2 = self.pos_drop(x_2)
        x_3 = self.pos_drop(x_3)

        # ---------- flatten ----------
        LG_1 = rearrange(x_1, 'b h w c -> b (h w) c')
        LG_2 = rearrange(x_2, 'b h w c -> b (h w) c')
        LG_3 = rearrange(x_3, 'b h w c -> b (h w) c')

        spatial_sizes = [
            (x_1.shape[1], x_1.shape[2]),
            (x_2.shape[1], x_2.shape[2]),
            (x_3.shape[1], x_3.shape[2]),
        ]

        LG = torch.cat([LG_1, LG_2, LG_3], dim=1)

        # ---------- Interval MT (default): ODE -> FFN -> Multi-scale Mamba ----------
        for i in range(self.depth):
            LG_1 = self.ode_1[i](LG_1, spatial_size=spatial_sizes[0])
            LG_2 = self.ode_2[i](LG_2, spatial_size=spatial_sizes[1])
            LG_3 = self.ode_3[i](LG_3, spatial_size=spatial_sizes[2])

            LG_T = torch.cat([LG_1, LG_2, LG_3], dim=1)
            LG_T = self.FFN[i](LG_T)

            LG_1, LG_2, LG_3 = self.multi_scale_mamba[i](
                LG_T[:, :LG_1.shape[1]],
                LG_T[:, LG_1.shape[1]:LG_1.shape[1]+LG_2.shape[1]],
                LG_T[:, LG_1.shape[1]+LG_2.shape[1]:],
                spatial_sizes
            )

            LG = torch.cat([LG_1, LG_2, LG_3], dim=1)

        feature = LG.mean(dim=1)
        return feature

    def forward(self, x, inference_params=None):
        feature = self.forward_features(x, inference_params)  
        x = self.head(self.head_drop(feature))   
        return x, feature 