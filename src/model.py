import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, in_channels: int, base: int = 32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class TemporalDifferenceFusion(nn.Sequential):
    """Fuse event features with their absolute change from the pre-event image."""

    def __init__(self, channels: int):
        super().__init__(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, pre, event):
        return super().forward(torch.cat([event, torch.abs(event - pre)], dim=1))


class DirectionalTemporalFusion(nn.Sequential):
    """Preserve the direction and magnitude of change from one historical flood."""

    def __init__(self, channels: int):
        super().__init__(
            nn.Conv2d(channels * 4, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, pre, event):
        signed_change = event - pre
        return super().forward(
            torch.cat([pre, event, signed_change, torch.abs(signed_change)], dim=1)
        )


class SiameseChangeUNet(nn.Module):
    """Shared temporal encoder with multi-scale event/change fusion."""

    def __init__(self, in_channels: int = 4, base: int = 32):
        super().__init__()
        if in_channels != 4:
            raise ValueError("SiameseChangeUNet expects pre/event VV/VH (4 channels)")
        self.enc1 = DoubleConv(2, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.bottleneck = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.fuse1 = TemporalDifferenceFusion(base)
        self.fuse2 = TemporalDifferenceFusion(base * 2)
        self.fuse3 = TemporalDifferenceFusion(base * 4)
        self.fuse_bottleneck = TemporalDifferenceFusion(base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    @staticmethod
    def _split_temporal(features):
        return features.chunk(2, dim=0)

    def encode_temporal(self, x):
        if x.shape[1] != 4:
            raise ValueError(f"Expected 4 channels, received {x.shape[1]}")
        # Put both dates in one batch so shared BatchNorm sees them together.
        paired = torch.cat([x[:, :2], x[:, 2:4]], dim=0)
        z1 = self.enc1(paired)
        z2 = self.enc2(self.pool(z1))
        z3 = self.enc3(self.pool(z2))
        zb = self.bottleneck(self.pool(z3))
        pre1, event1 = self._split_temporal(z1)
        pre2, event2 = self._split_temporal(z2)
        pre3, event3 = self._split_temporal(z3)
        preb, eventb = self._split_temporal(zb)
        f1 = self.fuse1(pre1, event1)
        f2 = self.fuse2(pre2, event2)
        f3 = self.fuse3(pre3, event3)
        fb = self.fuse_bottleneck(preb, eventb)
        return f1, f2, f3, fb

    def forward_features(self, x):
        f1, f2, f3, fb = self.encode_temporal(x)
        d3 = self.dec3(torch.cat([self.up3(fb), f3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), f2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), f1], dim=1))
        return self.head(d1), fb

    def forward(self, x):
        logits, _ = self.forward_features(x)
        return logits


class SiameseChangeUNetWithPresence(SiameseChangeUNet):
    """Siamese change U-Net with an image-level flood-presence gate."""

    def __init__(self, in_channels: int = 4, base: int = 32):
        super().__init__(in_channels, base)
        self.presence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(base * 8, 1),
        )

    def forward_with_aux(self, x):
        segmentation_logits, bottleneck = self.forward_features(x)
        presence_logits = self.presence_head(bottleneck).squeeze(1)
        return segmentation_logits, presence_logits

    @staticmethod
    def apply_presence_gate(segmentation_logits, presence_logits):
        presence = torch.sigmoid(presence_logits)[:, None, None, None]
        probability = torch.sigmoid(segmentation_logits) * presence
        probability = probability.float().clamp(1e-6, 1 - 1e-6)
        return torch.logit(probability).to(segmentation_logits.dtype)

    def forward(self, x):
        segmentation_logits, presence_logits = self.forward_with_aux(x)
        return self.apply_presence_gate(segmentation_logits, presence_logits)


class AtrousDepthwiseBranch(nn.Sequential):
    """Low-cost atrous context branch used at the Siamese bottleneck."""

    def __init__(self, channels: int, out_channels: int, dilation: int):
        super().__init__(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class MultiScaleResidualContext(nn.Module):
    """Residual multi-scale context without the cost of a full ASPP block."""

    def __init__(self, channels: int):
        super().__init__()
        reduced = max(channels // 2, 8)
        self.local = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.BatchNorm2d(reduced),
            nn.ReLU(inplace=True),
        )
        self.atrous = nn.ModuleList(
            AtrousDepthwiseBranch(channels, reduced, dilation) for dilation in (1, 2, 4)
        )
        self.project = nn.Sequential(
            nn.Conv2d(reduced * 4, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        context = torch.cat([self.local(x), *(branch(x) for branch in self.atrous)], dim=1)
        return x + self.project(context)


class DecoderGuidedSkipAttention(nn.Module):
    """Use coarse decoder semantics to reweight a temporal skip connection."""

    def __init__(self, skip_channels: int, gate_channels: int):
        super().__init__()
        hidden = max(skip_channels // 2, 8)
        self.skip_project = nn.Conv2d(skip_channels, hidden, 1, bias=False)
        self.gate_project = nn.Conv2d(gate_channels, hidden, 1, bias=False)
        self.score = nn.Sequential(
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, skip, gate):
        attention = self.score(self.skip_project(skip) + self.gate_project(gate))
        # The residual range [0.5, 1.5] keeps early optimization close to the baseline.
        return skip * (0.5 + attention)


class SiameseChangeUNetPresenceRefine(SiameseChangeUNetWithPresence):
    """Presence-gated Siamese U-Net with context, skip attention and edge refinement."""

    def __init__(self, in_channels: int = 4, base: int = 32):
        super().__init__(in_channels, base)
        self.context = MultiScaleResidualContext(base * 8)
        self.attention3 = DecoderGuidedSkipAttention(base * 4, base * 4)
        self.attention2 = DecoderGuidedSkipAttention(base * 2, base * 2)
        self.attention1 = DecoderGuidedSkipAttention(base, base)
        self.boundary_head = nn.Conv2d(base, 1, 1)
        self.boundary_refine = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )

    def forward_with_aux(self, x):
        f1, f2, f3, fb = self.encode_temporal(x)
        fb = self.context(fb)
        u3 = self.up3(fb)
        d3 = self.dec3(torch.cat([u3, self.attention3(f3, u3)], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, self.attention2(f2, u2)], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, self.attention1(f1, u1)], dim=1))
        boundary_logits = self.boundary_head(d1)
        refined = d1 + self.boundary_refine(d1) * torch.sigmoid(boundary_logits)
        segmentation_logits = self.head(refined)
        presence_logits = self.presence_head(fb).squeeze(1)
        return segmentation_logits, presence_logits, boundary_logits

    def forward(self, x):
        segmentation_logits, presence_logits, _ = self.forward_with_aux(x)
        return self.apply_presence_gate(segmentation_logits, presence_logits)


class HistoricalFloodForecastNet(SiameseChangeUNetPresenceRefine):
    """Forecast the next-event flood footprint from one previous flood observation pair."""

    def __init__(self, in_channels: int = 4, base: int = 32):
        super().__init__(in_channels, base)
        self.fuse1 = DirectionalTemporalFusion(base)
        self.fuse2 = DirectionalTemporalFusion(base * 2)
        self.fuse3 = DirectionalTemporalFusion(base * 4)
        self.fuse_bottleneck = DirectionalTemporalFusion(base * 8)


class HistoricalFloodResidualForecastNet(SiameseChangeUNetPresenceRefine):
    """Start from historical flood mapping and learn only the next-event residual."""

    def __init__(self, in_channels: int = 4, base: int = 32):
        super().__init__(in_channels, base)
        self.forecast_residual = nn.Sequential(
            nn.Conv2d(in_channels + 1, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, 1, 3, padding=1),
        )
        nn.init.zeros_(self.forecast_residual[-1].weight)
        nn.init.zeros_(self.forecast_residual[-1].bias)

    def forward_with_aux(self, x):
        historical_logits, presence_logits, boundary_logits = super().forward_with_aux(x)
        historical_probability = torch.sigmoid(historical_logits)
        residual = self.forecast_residual(torch.cat([x, historical_probability], dim=1))
        return historical_logits + residual, presence_logits, boundary_logits


class ASPPBranch(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        kernel_size = 1 if dilation == 1 else 3
        padding = 0 if dilation == 1 else dilation
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 128):
        super().__init__()
        self.branches = nn.ModuleList(
            ASPPBranch(in_channels, out_channels, rate) for rate in (1, 6, 12, 18)
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        pooled = self.pool(x)
        pooled = F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.project(torch.cat([*(branch(x) for branch in self.branches), pooled], dim=1))


class DeepLabV3PlusMobileNet(nn.Module):
    """DeepLabV3+ with a lightweight MobileNetV3-Small encoder."""

    def __init__(self, in_channels: int = 4):
        super().__init__()
        from torchvision.models import mobilenet_v3_small

        encoder = mobilenet_v3_small(weights=None).features
        first = encoder[0][0]
        encoder[0][0] = nn.Conv2d(
            in_channels,
            first.out_channels,
            first.kernel_size,
            first.stride,
            first.padding,
            bias=False,
        )
        self.encoder = encoder
        self.aspp = ASPP(576, 128)
        self.low_project = nn.Sequential(
            nn.Conv2d(24, 24, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            DoubleConv(152, 128),
            nn.Conv2d(128, 1, 1),
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        low = None
        for index, block in enumerate(self.encoder):
            x = block(x)
            if index == 3:
                low = x
        if low is None:
            raise RuntimeError("MobileNet encoder did not emit a low-level feature map")
        x = self.aspp(x)
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decoder(torch.cat([x, self.low_project(low)], dim=1))
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)


class OverlapPatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, kernel_size: int, stride: int):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        height, width = x.shape[-2:]
        return self.norm(x.flatten(2).transpose(1, 2)), height, width


class EfficientSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, sr_ratio: int):
        super().__init__()
        if dim % heads:
            raise ValueError(f"Embedding dimension {dim} must be divisible by {heads} heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, height: int, width: int):
        batch, tokens, channels = x.shape
        q = self.q(x).reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        reduced = x
        if self.sr_ratio > 1:
            reduced = x.transpose(1, 2).reshape(batch, channels, height, width)
            reduced = self.sr(reduced).flatten(2).transpose(1, 2)
            reduced = self.norm(reduced)
        kv = self.kv(reduced).reshape(batch, -1, 2, self.heads, self.head_dim)
        key, value = kv.permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, key, value, dropout_p=0.0)
        x = x.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(x)


class MixFFN(nn.Module):
    def __init__(self, dim: int, expansion: int):
        super().__init__()
        hidden_dim = dim * expansion
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.depthwise = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x, height: int, width: int):
        batch, _, _ = x.shape
        x = self.fc1(x)
        channels = x.shape[-1]
        x = x.transpose(1, 2).reshape(batch, channels, height, width)
        x = self.depthwise(x).flatten(2).transpose(1, 2)
        return self.fc2(F.gelu(x))


class MixTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, sr_ratio: int, expansion: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = EfficientSelfAttention(dim, heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim, expansion)

    def forward(self, x, height: int, width: int):
        x = x + self.attention(self.norm1(x), height, width)
        return x + self.ffn(self.norm2(x), height, width)


class MixTransformerStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dim: int,
        depth: int,
        heads: int,
        sr_ratio: int,
        expansion: int,
        first: bool = False,
    ):
        super().__init__()
        self.patch = OverlapPatchEmbedding(
            in_channels,
            dim,
            kernel_size=7 if first else 3,
            stride=4 if first else 2,
        )
        self.blocks = nn.ModuleList(
            MixTransformerBlock(dim, heads, sr_ratio, expansion) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x, height, width = self.patch(x)
        for block in self.blocks:
            x = block(x, height, width)
        x = self.norm(x)
        return x.transpose(1, 2).reshape(x.shape[0], -1, height, width)


class SegFormerB0(nn.Module):
    """Dependency-free SegFormer-B0 for four-channel SAR segmentation."""

    def __init__(self, in_channels: int = 4, decoder_dim: int = 256):
        super().__init__()
        dims = (32, 64, 160, 256)
        depths = (2, 2, 2, 2)
        heads = (1, 2, 5, 8)
        sr_ratios = (8, 4, 2, 1)
        expansions = (8, 8, 4, 4)
        stages = []
        stage_in = in_channels
        for index, values in enumerate(zip(dims, depths, heads, sr_ratios, expansions)):
            dim, depth, stage_heads, sr_ratio, expansion = values
            stages.append(
                MixTransformerStage(
                    stage_in,
                    dim,
                    depth,
                    stage_heads,
                    sr_ratio,
                    expansion,
                    first=index == 0,
                )
            )
            stage_in = dim
        self.stages = nn.ModuleList(stages)
        self.projections = nn.ModuleList(nn.Conv2d(dim, decoder_dim, 1) for dim in dims)
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, 1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(decoder_dim, 1, 1),
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        target_size = features[0].shape[-2:]
        decoded = [
            F.interpolate(project(feature), target_size, mode="bilinear", align_corners=False)
            for project, feature in zip(self.projections, features)
        ]
        x = self.fuse(torch.cat(decoded, dim=1))
        return F.interpolate(x, input_size, mode="bilinear", align_corners=False)


class SiameseSegFormerB0(nn.Module):
    """Shared MiT-B0 encoder with multi-scale temporal-difference fusion."""

    def __init__(self, in_channels: int = 4, decoder_dim: int = 256):
        super().__init__()
        if in_channels != 4:
            raise ValueError("SiameseSegFormerB0 expects pre/event VV/VH (4 channels)")
        dims = (32, 64, 160, 256)
        depths = (2, 2, 2, 2)
        heads = (1, 2, 5, 8)
        sr_ratios = (8, 4, 2, 1)
        expansions = (8, 8, 4, 4)
        stages = []
        stage_in = 2
        for index, values in enumerate(zip(dims, depths, heads, sr_ratios, expansions)):
            dim, depth, stage_heads, sr_ratio, expansion = values
            stages.append(
                MixTransformerStage(
                    stage_in,
                    dim,
                    depth,
                    stage_heads,
                    sr_ratio,
                    expansion,
                    first=index == 0,
                )
            )
            stage_in = dim
        self.stages = nn.ModuleList(stages)
        self.fusions = nn.ModuleList(TemporalDifferenceFusion(dim) for dim in dims)
        self.projections = nn.ModuleList(nn.Conv2d(dim, decoder_dim, 1) for dim in dims)
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, 1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(decoder_dim, 1, 1),
        )

    def forward(self, x):
        if x.shape[1] != 4:
            raise ValueError(f"Expected 4 channels, received {x.shape[1]}")
        input_size = x.shape[-2:]
        paired = torch.cat([x[:, :2], x[:, 2:4]], dim=0)
        features = []
        for stage, fusion in zip(self.stages, self.fusions):
            paired = stage(paired)
            pre, event = paired.chunk(2, dim=0)
            features.append(fusion(pre, event))
        target_size = features[0].shape[-2:]
        decoded = [
            F.interpolate(project(feature), target_size, mode="bilinear", align_corners=False)
            for project, feature in zip(self.projections, features)
        ]
        logits = self.fuse(torch.cat(decoded, dim=1))
        return F.interpolate(logits, input_size, mode="bilinear", align_corners=False)


def build_model(name: str, in_channels: int, base: int):
    name = (name or "unet").lower()
    if name == "unet":
        return UNet(in_channels, base)
    if name == "siamese_change_unet":
        return SiameseChangeUNet(in_channels, base)
    if name == "siamese_change_unet_presence":
        return SiameseChangeUNetWithPresence(in_channels, base)
    if name == "siamese_change_unet_presence_refine":
        return SiameseChangeUNetPresenceRefine(in_channels, base)
    if name == "historical_flood_forecast_net":
        return HistoricalFloodForecastNet(in_channels, base)
    if name == "historical_flood_residual_forecast_net":
        return HistoricalFloodResidualForecastNet(in_channels, base)
    if name == "deeplabv3plus_mobilenet":
        return DeepLabV3PlusMobileNet(in_channels)
    if name == "segformer_b0":
        return SegFormerB0(in_channels)
    if name == "siamese_segformer_b0":
        return SiameseSegFormerB0(in_channels)
    raise ValueError(f"Unknown model: {name}")
