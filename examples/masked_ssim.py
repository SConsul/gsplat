import torch

import torch.nn.functional as F

def masked_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor, 
    mask: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,  
) -> torch.Tensor:
    """
    Compute SSIM loss only on masked (valid) regions.
    
    Args:
        img1: Rendered image [B, C, H, W]
        img2: Ground truth image [B, C, H, W]
        mask: Valid region mask [B, H, W], True = valid
        window_size: SSIM window size
        C1, C2: SSIM constants for stability
    
    Returns:
        Scalar SSIM loss (1 - SSIM) averaged over valid pixels only
    """
    B, C, H, W = img1.shape
    device = img1.device
    
    # Create Gaussian window
    def gaussian_window(size, sigma=1.5):
        coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.outer(g)
    
    window = gaussian_window(window_size)
    window = window.expand(C, 1, window_size, window_size)
    
    pad = window_size // 2
    
    # Compute local means
    mu1 = F.conv2d(img1, window, padding=pad, groups=C)
    mu2 = F.conv2d(img2, window, padding=pad, groups=C)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute local variances and covariance
    sigma1_sq = F.conv2d(img1 ** 2, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=C) - mu1_mu2
    
    # Clamp variances to avoid numerical issues
    sigma1_sq = torch.clamp(sigma1_sq, min=0)
    sigma2_sq = torch.clamp(sigma2_sq, min=0)
    
    # SSIM formula
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator  # [B, C, H, W]
    
    # Average over channels to get per-pixel SSIM
    ssim_map = ssim_map.mean(dim=1)  # [B, H, W]
    
    # Also compute mask weight map: how much of each SSIM window is valid
    # This downweights SSIM values at mask boundaries
    mask_float = mask.float().unsqueeze(1)  # [B, 1, H, W]
    window_1ch = window[0:1]  # [1, 1, window_size, window_size]
    mask_weight = F.conv2d(mask_float, window_1ch, padding=pad)  # [B, 1, H, W]
    mask_weight = mask_weight.squeeze(1)  # [B, H, W]
    
    # Only include pixels where the window is mostly valid (>50% valid pixels)
    valid_ssim_mask = (mask_weight > 0.5) & mask
    
    if valid_ssim_mask.sum() == 0:
        # No valid pixels, return 0 loss
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Compute weighted SSIM loss only on valid regions
    # Weight by how much of the window is valid
    weights = mask_weight[valid_ssim_mask]
    ssim_values = ssim_map[valid_ssim_mask]
    
    # Weighted average
    ssim_score = (ssim_values * weights).sum() / weights.sum()
    
    return 1.0 - ssim_score