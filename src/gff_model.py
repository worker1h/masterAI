"""Horizon-conditioned Transformer model for GFF flood-footprint forecasting."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .model import MixTransformerStage


class WeatherSpatialEncoder(nn.Module):
    """Compress each day's coarse meteorological fields into one token."""

    def __init__(self, in_channels: int, dim: int):
        super().__init__()
        hidden = max(dim // 2, 32)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, stride=2, padding=1, groups=dim, bias=False),
            nn.GroupNorm(8, dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = x.shape
        x = x.reshape(batch * steps, channels, height, width)
        x = self.encoder(x).flatten(1)
        return x.reshape(batch, steps, -1)


class TemporalForcingTransformer(nn.Module):
    """Encode observed/future slots for a requested lead time."""

    def __init__(
        self,
        in_channels: int,
        dim: int = 128,
        depth: int = 3,
        heads: int = 4,
        max_steps: int = 32,
        max_horizon: int = 3,
    ):
        super().__init__()
        self.spatial = WeatherSpatialEncoder(in_channels, dim)
        self.position = nn.Parameter(torch.zeros(1, max_steps + 1, dim))
        self.forecast_state = nn.Embedding(2, dim)
        self.horizon = nn.Embedding(max_horizon + 1, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 3,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(
        self,
        weather: torch.Tensor,
        horizon: torch.Tensor,
        forecast_mask: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.spatial(weather)
        if tokens.shape[1] + 1 > self.position.shape[1]:
            raise ValueError(
                f"Weather sequence has {tokens.shape[1]} steps; maximum is "
                f"{self.position.shape[1] - 1}"
            )
        horizon = horizon.long().clamp(0, self.horizon.num_embeddings - 1)
        horizon_token = self.horizon(horizon).unsqueeze(1)
        tokens = tokens + self.forecast_state(forecast_mask.long())
        sequence = torch.cat([horizon_token, tokens], dim=1)
        sequence = sequence + self.position[:, : sequence.shape[1]]
        return self.norm(self.transformer(sequence)[:, 0])


class FeatureWiseModulation(nn.Module):
    """Inject basin forcing and lead-time context into a spatial feature map."""

    def __init__(self, context_dim: int, channels: int):
        super().__init__()
        self.affine = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, channels * 2),
        )
        nn.init.zeros_(self.affine[-1].weight)
        nn.init.zeros_(self.affine[-1].bias)

    def forward(self, feature: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        scale, bias = self.affine(context).chunk(2, dim=1)
        scale = 0.25 * torch.tanh(scale)[:, :, None, None]
        bias = bias[:, :, None, None]
        return feature * (1.0 + scale) + bias


class GFFHorizonFormer(nn.Module):
    """MiT-B0 spatial encoder fused with a temporal forcing Transformer.

    Inputs
    ------
    spatial:
        Pre-event Sentinel-1 VV/VH, DEM and HAND, shaped ``B x 4 x H x W``.
    weather:
        Daily ERA5/ERA5-Land/GloFAS fields, shaped ``B x T x C x h x w``.
    horizon:
        Requested lead in days (1, 2 or 3).
    forecast_mask:
        Boolean ``B x T`` mask marking slots after forecast issue time. In
        causal mode those slots contain normalized climatological means.
    """

    def __init__(
        self,
        spatial_channels: int = 4,
        weather_channels: int = 26,
        decoder_dim: int = 192,
        temporal_dim: int = 128,
        temporal_depth: int = 3,
    ):
        super().__init__()
        dims = (32, 64, 160, 256)
        depths = (2, 2, 2, 2)
        heads = (1, 2, 5, 8)
        sr_ratios = (8, 4, 2, 1)
        expansions = (8, 8, 4, 4)
        self.temporal = TemporalForcingTransformer(
            weather_channels, temporal_dim, temporal_depth
        )
        stages = []
        stage_in = spatial_channels
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
        self.modulations = nn.ModuleList(
            FeatureWiseModulation(temporal_dim, dim) for dim in dims
        )
        self.projections = nn.ModuleList(
            nn.Conv2d(dim, decoder_dim, 1) for dim in dims
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, 1, bias=False),
            nn.GroupNorm(16, decoder_dim),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_dim, decoder_dim // 2, 3, padding=1, bias=False),
            nn.GroupNorm(8, decoder_dim // 2),
            nn.GELU(),
        )
        head_channels = decoder_dim // 2
        self.segmentation_head = nn.Conv2d(head_channels, 1, 1)
        self.boundary_head = nn.Conv2d(head_channels, 1, 1)
        self.presence_head = nn.Sequential(
            nn.LayerNorm(temporal_dim + dims[-1]),
            nn.Linear(temporal_dim + dims[-1], 1),
        )

    def forward(
        self,
        spatial: torch.Tensor,
        weather: torch.Tensor,
        horizon: torch.Tensor,
        forecast_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        input_size = spatial.shape[-2:]
        context = self.temporal(weather, horizon, forecast_mask)
        features = []
        x = spatial
        for stage, modulation in zip(self.stages, self.modulations):
            x = modulation(stage(x), context)
            features.append(x)

        target_size = features[0].shape[-2:]
        decoded = [
            F.interpolate(project(feature), target_size, mode="bilinear", align_corners=False)
            for project, feature in zip(self.projections, features)
        ]
        decoded = self.decoder(torch.cat(decoded, dim=1))
        segmentation = F.interpolate(
            self.segmentation_head(decoded), input_size, mode="bilinear", align_corners=False
        )
        boundary = F.interpolate(
            self.boundary_head(decoded), input_size, mode="bilinear", align_corners=False
        )
        deepest = F.adaptive_avg_pool2d(features[-1], 1).flatten(1)
        presence = self.presence_head(torch.cat([context, deepest], dim=1)).squeeze(1)
        return {
            "segmentation": segmentation,
            "boundary": boundary,
            "presence": presence,
        }


def _conv_norm_gelu(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        ),
        nn.GroupNorm(8, out_channels),
        nn.GELU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(8, out_channels),
        nn.GELU(),
    )


class GFFViTHorizonFormer(nn.Module):
    """ImageNet-pretrained ViT-B/16 with a U-shaped flood decoder.

    The spatial stream consumes SU-Net-inspired original/enhanced SAR channels
    plus DEM/HAND. A ViT-B/16 models global spatial dependencies at 14x14 token
    resolution, while convolutional skips restore thin flood boundaries. The
    same causal weather Transformer and horizon conditioning as the baseline
    are retained for a controlled architecture comparison.
    """

    def __init__(
        self,
        spatial_channels: int = 6,
        weather_channels: int = 26,
        temporal_dim: int = 128,
        temporal_depth: int = 3,
        pretrained: bool = True,
        freeze_blocks: int = 6,
    ):
        super().__init__()
        from torchvision.models import ViT_B_16_Weights, vit_b_16

        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = vit_b_16(weights=weights, image_size=224)
        hidden_dim = int(self.vit.hidden_dim)
        old_projection = self.vit.conv_proj
        projection = nn.Conv2d(
            spatial_channels,
            hidden_dim,
            kernel_size=old_projection.kernel_size,
            stride=old_projection.stride,
            bias=old_projection.bias is not None,
        )
        with torch.no_grad():
            if pretrained:
                mean_kernel = old_projection.weight.mean(dim=1, keepdim=True)
                projection.weight.copy_(
                    mean_kernel.repeat(1, spatial_channels, 1, 1)
                    * (3.0 / float(spatial_channels))
                )
                if projection.bias is not None and old_projection.bias is not None:
                    projection.bias.copy_(old_projection.bias)
            else:
                nn.init.trunc_normal_(projection.weight, std=0.02)
                if projection.bias is not None:
                    nn.init.zeros_(projection.bias)
        self.vit.conv_proj = projection
        for index, block in enumerate(self.vit.encoder.layers):
            if index < int(freeze_blocks):
                for parameter in block.parameters():
                    parameter.requires_grad = False

        self.temporal = TemporalForcingTransformer(
            weather_channels, temporal_dim, temporal_depth
        )
        self.token_modulation = FeatureWiseModulation(temporal_dim, hidden_dim)
        self.full_skip = _conv_norm_gelu(spatial_channels, 32)
        self.half_skip = _conv_norm_gelu(32, 64, stride=2)
        self.quarter_skip = _conv_norm_gelu(64, 128, stride=2)
        self.token_projection = nn.Conv2d(hidden_dim, 256, 1)
        self.up_28 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.up_56 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.fuse_56 = _conv_norm_gelu(256, 128)
        self.up_112 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.fuse_112 = _conv_norm_gelu(128, 64)
        self.up_224 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.fuse_224 = _conv_norm_gelu(64, 64)
        self.decoder_modulations = nn.ModuleList(
            [
                FeatureWiseModulation(temporal_dim, 128),
                FeatureWiseModulation(temporal_dim, 64),
                FeatureWiseModulation(temporal_dim, 64),
            ]
        )
        self.segmentation_head = nn.Conv2d(64, 1, 1)
        self.boundary_head = nn.Conv2d(64, 1, 1)
        self.presence_head = nn.Sequential(
            nn.LayerNorm(temporal_dim + hidden_dim),
            nn.Linear(temporal_dim + hidden_dim, 1),
        )

    def _vit_features(self, spatial: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = spatial.shape[0]
        tokens = self.vit._process_input(spatial)
        class_token = self.vit.class_token.expand(batch, -1, -1)
        sequence = self.vit.encoder(torch.cat([class_token, tokens], dim=1))
        cls = sequence[:, 0]
        patches = sequence[:, 1:].transpose(1, 2).reshape(batch, self.vit.hidden_dim, 14, 14)
        return patches, cls

    def forward(
        self,
        spatial: torch.Tensor,
        weather: torch.Tensor,
        horizon: torch.Tensor,
        forecast_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if spatial.shape[-2:] != (224, 224):
            raise ValueError("ViT-B/16 configuration requires 224x224 spatial tiles")
        context = self.temporal(weather, horizon, forecast_mask)
        full = self.full_skip(spatial)
        half = self.half_skip(full)
        quarter = self.quarter_skip(half)
        tokens, cls = self._vit_features(spatial)
        tokens = self.token_modulation(tokens, context)
        decoded = self.up_56(self.up_28(self.token_projection(tokens)))
        decoded = self.fuse_56(torch.cat([decoded, quarter], dim=1))
        decoded = self.decoder_modulations[0](decoded, context)
        decoded = self.up_112(decoded)
        decoded = self.fuse_112(torch.cat([decoded, half], dim=1))
        decoded = self.decoder_modulations[1](decoded, context)
        decoded = self.up_224(decoded)
        decoded = self.fuse_224(torch.cat([decoded, full], dim=1))
        decoded = self.decoder_modulations[2](decoded, context)
        presence = self.presence_head(torch.cat([context, cls], dim=1)).squeeze(1)
        return {
            "segmentation": self.segmentation_head(decoded),
            "boundary": self.boundary_head(decoded),
            "presence": presence,
        }


def build_gff_model(
    config: dict, spatial_channels: int, weather_channels: int = 26
) -> nn.Module:
    """Construct the configured GFF spatial/temporal Transformer."""

    model_config = config.get("model", {})
    name = str(model_config.get("name", "mit_segformer")).lower()
    if name in {"vit", "vit_b_16", "vit-b/16"}:
        return GFFViTHorizonFormer(
            spatial_channels=spatial_channels,
            weather_channels=weather_channels,
            temporal_dim=int(model_config.get("temporal_dim", 128)),
            temporal_depth=int(model_config.get("temporal_depth", 3)),
            pretrained=bool(model_config.get("pretrained", True)),
            freeze_blocks=int(model_config.get("freeze_blocks", 6)),
        )
    if name not in {"mit", "mit_segformer", "horizonformer"}:
        raise ValueError(f"Unknown GFF model name: {name}")
    return GFFHorizonFormer(
        spatial_channels=spatial_channels,
        weather_channels=weather_channels,
        decoder_dim=int(model_config.get("decoder_dim", 192)),
        temporal_dim=int(model_config.get("temporal_dim", 128)),
        temporal_depth=int(model_config.get("temporal_depth", 3)),
    )
