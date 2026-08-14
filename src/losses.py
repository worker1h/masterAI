import torch
import torch.nn.functional as F


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dims)
    denominator = probs.sum(dims) + targets.sum(dims)
    return (1 - (2 * intersection + eps) / (denominator + eps)).mean()


def bce_dice_loss(logits, targets, pos_weight=None):
    weight = None if pos_weight is None else torch.as_tensor(pos_weight,device=logits.device,dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight) + dice_loss(logits, targets)


def tversky_loss(logits, targets, alpha=0.5, beta=0.5, eps=1e-6):
    """Region loss with independently controllable false-positive/negative costs."""
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    tp = (probs * targets).sum(dims)
    fp = (probs * (1 - targets)).sum(dims)
    fn = ((1 - probs) * targets).sum(dims)
    score = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return (1 - score).mean()


def boundary_band(targets, kernel_size=5):
    """Return a two-sided band around binary target boundaries."""
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("boundary kernel_size must be an odd integer >= 3")
    pad = kernel_size // 2
    dilated = F.max_pool2d(targets, kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-targets, kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp_(0, 1)


def focal_bce_loss(logits, targets, pos_weight=None, gamma=2.0, pixel_weight=None):
    """Numerically stable focal BCE, optionally emphasizing a boundary band."""
    weight = None if pos_weight is None else torch.as_tensor(
        pos_weight, device=logits.device, dtype=logits.dtype
    )
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=weight, reduction="none"
    )
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1 - probs) * (1 - targets)
    loss = bce * (1 - pt).pow(gamma)
    if pixel_weight is not None:
        loss = loss * pixel_weight
    return loss.mean()


def segmentation_loss(logits, targets, pos_weight=None, config=None):
    """Configurable loss while retaining the original BCE+Dice baseline."""
    config = config or {}
    loss_type = config.get("type", "bce_dice")
    boundary_weight = float(config.get("boundary_weight", 0.0))
    pixel_weight = None
    if boundary_weight > 0:
        band = boundary_band(targets, int(config.get("boundary_kernel_size", 5)))
        pixel_weight = 1 + boundary_weight * band

    if loss_type == "bce_dice":
        weight = None if pos_weight is None else torch.as_tensor(
            pos_weight, device=logits.device, dtype=logits.dtype
        )
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=weight, reduction="none"
        )
        if pixel_weight is not None:
            bce = bce * pixel_weight
        return float(config.get("bce_weight", 1.0)) * bce.mean() + float(
            config.get("dice_weight", 1.0)
        ) * dice_loss(logits, targets)

    if loss_type == "focal_tversky":
        focal = focal_bce_loss(
            logits,
            targets,
            pos_weight=pos_weight,
            gamma=float(config.get("focal_gamma", 2.0)),
            pixel_weight=pixel_weight,
        )
        tversky = tversky_loss(
            logits,
            targets,
            alpha=float(config.get("tversky_alpha", 0.6)),
            beta=float(config.get("tversky_beta", 0.4)),
        )
        return float(config.get("focal_weight", 1.0)) * focal + float(
            config.get("tversky_weight", 1.0)
        ) * tversky

    raise ValueError(f"Unknown loss type: {loss_type}")
