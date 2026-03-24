import torch


def frame_to_latent_index(frame_idx: int) -> int:
    """Convert frame index to latent index according to WanVideoVAE38 compression pattern."""
    if frame_idx == 0:
        return 0
    else:
        return 1 + (frame_idx - 1) // 4


def _build_absolute_freqs(
    original_freqs, 
    f: int, h: int, w: int, 
    device: torch.device,
    abs_frame_start: int = 0,
    abs_h_start: int = 0, 
    abs_w_start: int = 0,
    vae_spatial_compression: int = 16
):
    """Build frequency embeddings using absolute coordinates in the full video (latent space)."""
    # Convert frame indices to latent indices according to VAE compression pattern
    frame_indices = range(abs_frame_start, abs_frame_start + f)
    t_indices = torch.tensor([frame_to_latent_index(idx) for idx in frame_indices], device=device)
    
    # Convert spatial coordinates to latent space (divided by VAE compression factor)
    h_indices = torch.arange(abs_h_start // vae_spatial_compression, 
                            (abs_h_start + h * vae_spatial_compression) // vae_spatial_compression, 
                            device=device)
    w_indices = torch.arange(abs_w_start // vae_spatial_compression, 
                            (abs_w_start + w * vae_spatial_compression) // vae_spatial_compression, 
                            device=device)
    
    # Ensure indices don't exceed the original frequency table bounds
    max_t = original_freqs[0].shape[0] - 1
    max_h = original_freqs[1].shape[0] - 1
    max_w = original_freqs[2].shape[0] - 1
    
    t_indices = torch.clamp(t_indices, 0, max_t)
    h_indices = torch.clamp(h_indices, 0, max_h)
    w_indices = torch.clamp(w_indices, 0, max_w)
    
    # Handle case where h_indices or w_indices might be empty due to compression
    if h_indices.numel() == 0:
        h_indices = torch.tensor([abs_h_start // vae_spatial_compression], device=device)
        h_indices = torch.clamp(h_indices, 0, max_h)
    if w_indices.numel() == 0:
        w_indices = torch.tensor([abs_w_start // vae_spatial_compression], device=device)
        w_indices = torch.clamp(w_indices, 0, max_w)
    
    # Ensure we have the right number of spatial indices
    if len(h_indices) < h:
        h_indices = h_indices.repeat((h + len(h_indices) - 1) // len(h_indices))[:h]
    if len(w_indices) < w:
        w_indices = w_indices.repeat((w + len(w_indices) - 1) // len(w_indices))[:w]
    
    freqs = torch.cat([
        original_freqs[0][t_indices].view(f, 1, 1, -1).expand(f, h, w, -1),
        original_freqs[1][h_indices[:h]].view(1, h, 1, -1).expand(f, h, w, -1),
        original_freqs[2][w_indices[:w]].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(device)
    return freqs


def compute_cube_map_freqs_cross(
    original_freqs, 
    f: int, cube_h: int, cube_w: int, 
    device: torch.device,
    abs_frame_start: int = 0,
    abs_h_start: int = 0,
    abs_w_start: int = 0,
    vae_spatial_compression: int = 16
):
    """
    Cross T-shaped positional mapping for 6-face horizon input ordered as [F, R, B, L, U, D].
    Returns (f*cube_h*cube_w, 1, dim_total).
    
    Args:
        abs_frame_start: Absolute frame start index in full video
        abs_h_start: Absolute height start index (for cube map positioning)
        abs_w_start: Absolute width start index (for cube map positioning)
        vae_spatial_compression: VAE spatial compression factor
    """
    assert cube_w == cube_h * 6, "cube_w must be 6 times cube_h"
    C = cube_h
    original_freqs_h = original_freqs[1].to(device)
    original_freqs_w = original_freqs[2].to(device)

    # Convert spatial coordinates to latent space
    latent_h_start = abs_h_start // vae_spatial_compression
    latent_w_start = abs_w_start // vae_spatial_compression
    latent_C = C  # This is already in latent space dimensions

    # Build virtual index grids (C, 6C) with absolute positioning in latent space
    hv = torch.full((latent_C, 6 * latent_C), fill_value=latent_C + latent_h_start, device=device, dtype=torch.long)
    rows = torch.arange(latent_C, device=device) + latent_h_start
    # Middle band for F,R,B,L blocks
    hv[:, 0:4*latent_C] = (latent_C + rows.view(latent_C, 1)).expand(latent_C, 4*latent_C)
    # Up band
    hv[:, 4*latent_C:5*latent_C] = rows.view(latent_C, 1).expand(latent_C, latent_C)
    # Down band
    hv[:, 5*latent_C:6*latent_C] = (2*latent_C + rows.view(latent_C, 1)).expand(latent_C, latent_C)

    # Virtual W indices so that U/D align over F (with absolute positioning in latent space)
    wv = torch.empty((latent_C, 6 * latent_C), device=device, dtype=torch.long)
    base_w = latent_w_start
    wv[:, 0:latent_C]     = torch.arange(base_w,   base_w+latent_C, device=device).view(1, latent_C).expand(latent_C, latent_C)     # F
    wv[:, latent_C:2*latent_C]   = torch.arange(base_w+latent_C, base_w+2*latent_C, device=device).view(1, latent_C).expand(latent_C, latent_C)   # R
    wv[:, 2*latent_C:3*latent_C] = torch.arange(base_w+2*latent_C, base_w+3*latent_C, device=device).view(1, latent_C).expand(latent_C, latent_C) # B
    wv[:, 3*latent_C:4*latent_C] = torch.arange(base_w+3*latent_C, base_w+4*latent_C, device=device).view(1, latent_C).expand(latent_C, latent_C) # L
    wv[:, 4*latent_C:5*latent_C] = torch.arange(base_w,   base_w+latent_C, device=device).view(1, latent_C).expand(latent_C, latent_C)     # U -> align with F
    wv[:, 5*latent_C:6*latent_C] = torch.arange(base_w,   base_w+latent_C, device=device).view(1, latent_C).expand(latent_C, latent_C)     # D -> align with F

    # Clamp indices to valid ranges
    max_h = original_freqs_h.shape[0] - 1
    max_w = original_freqs_w.shape[0] - 1
    hv = torch.clamp(hv, 0, max_h)
    wv = torch.clamp(wv, 0, max_w)

    freqs_h = original_freqs_h[hv]  # (C,6C,dim_h)
    freqs_w = original_freqs_w[wv]  # (C,6C,dim_w)
    
    # Convert frame indices to latent indices
    frame_indices = range(abs_frame_start, abs_frame_start + f)
    t_indices = torch.tensor([frame_to_latent_index(idx) for idx in frame_indices], device=device)
    max_t = original_freqs[0].shape[0] - 1
    t_indices = torch.clamp(t_indices, 0, max_t)
    
    freqs_f = original_freqs[0][t_indices].view(f, 1, 1, -1).expand(f, latent_C, 6 * latent_C, -1).to(device)
    freqs_h = freqs_h.view(1, latent_C, 6*latent_C, -1).expand(f, latent_C, 6*latent_C, -1)
    freqs_w = freqs_w.view(1, latent_C, 6*latent_C, -1).expand(f, latent_C, 6*latent_C, -1)
    cross_freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1)
    cross_freqs = cross_freqs.reshape(f * latent_C * 6 * latent_C, 1, -1)
    return cross_freqs


def build_cube_rope_freqs(
    original_freqs, 
    f: int, h: int, w: int, 
    device: torch.device, 
    mapping: str = "cross",
    abs_frame_start: int = 0,
    abs_h_start: int = 0,
    abs_w_start: int = 0,
    vae_spatial_compression: int = 16
):
    """
    Build RoPE frequencies for cube map with absolute positioning support.
    
    Args:
        abs_frame_start: Absolute frame start index in full video
        abs_h_start: Absolute height start index  
        abs_w_start: Absolute width start index
        vae_spatial_compression: VAE spatial compression factor
    """
    if mapping == "cross":
        return compute_cube_map_freqs_cross(
            original_freqs, f, h, w, device, 
            abs_frame_start=abs_frame_start, 
            abs_h_start=abs_h_start, 
            abs_w_start=abs_w_start,
            vae_spatial_compression=vae_spatial_compression
        )
    else:
        return _build_absolute_freqs(
            original_freqs, f, h, w, device,
            abs_frame_start=abs_frame_start,
            abs_h_start=abs_h_start, 
            abs_w_start=abs_w_start,
            vae_spatial_compression=vae_spatial_compression
        )
