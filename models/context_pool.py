import torch
import os
import torchvision
from typing import Dict, List, Tuple, Optional
from einops import rearrange
import logging
import traceback

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(funcName)s - %(levelname)s - %(message)s")


def _save_context_video(context_bcthw: torch.Tensor, save_dir: str, 
                        save_prefix: str = "context", step: Optional[int] = None, 
                        rank: Optional[int] = None, face: Optional[str] = None,
                        order_idx: Optional[int] = None):
    """
    Internal method to safely save context tensor as video.
    
    Args:
        context_bcthw: Context tensor in BCTHW format
        save_dir: Directory to save the video file
        save_prefix: Prefix for the filename
        step: Training/validation step number
        rank: Process rank (for multi-GPU)
        face: Current face being processed
        order_idx: Current order index
    """
    try:
        if context_bcthw is None or not isinstance(context_bcthw, torch.Tensor):
            return
        
        # Build filename with all available info
        filename_parts = [save_prefix]
        if step is not None:
            filename_parts.append(f"step{step}")
        if rank is not None:
            filename_parts.append(f"rank{rank}")
        if order_idx is not None:
            filename_parts.append(f"order{order_idx}")
        if face is not None:
            filename_parts.append(f"{face}")
        
        filename = "_".join(filename_parts) + ".mp4"
        file_path = os.path.join(save_dir, filename)
        
        # Ensure we're working with a copy on CPU to avoid GPU memory issues
        ctx = context_bcthw.detach().cpu().clone()
        
        
        if ctx.ndim == 5 and ctx.shape[0] == 1:
            ctx = ctx[0]  # Remove batch dimension: CTHW
        
        if ctx.ndim == 4:
            if ctx.shape[0] > 3:
                ctx = ctx[:3, :, :, :]
            # normalize ctx to [0, 1]
            ctx = ctx - ctx.min()
            ctx = ctx / ctx.max()
                
            # Convert CTHW to TCHW
            ctx_tchw = ctx.permute(1, 0, 2, 3).contiguous()
            
            # Handle grayscale by repeating channels
            if ctx_tchw.shape[1] == 1:
                ctx_tchw = ctx_tchw.repeat(1, 3, 1, 1)
            
            # Convert to uint8 and prepare for video writing (THWC)
            ctx_thwc = (ctx_tchw.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
            
            # Create output directory if needed
            os.makedirs(save_dir, exist_ok=True)
            
            # Save video
            torchvision.io.write_video(file_path, ctx_thwc, fps=8)
    except Exception as e:
        logging.warning(f"Failed to save context video to {file_path}: {e}")
        traceback.print_exc()

# Find the nearest 4N+1 to actual_length
# 4N+1 = k, where k <= actual_length or k >= actual_length, and k = 4*n + 1
def nearest_4n1(val, direction, max_len):
    if direction == "up":
        # Smallest 4N+1 >= val
        n = (val + 3) // 4
        k = 4 * n + 1
        if k <= max_len:
            return k
        else:
            return None
    else:
        # Largest 4N+1 <= val
        n = (val - 1) // 4
        k = 4 * n + 1
        if k >= 4:
            return k
        else:
            return None


class PositionalEmbeddingPadder:
    """
    Applies padding to positional embeddings using the same logic as CubeMapPadder.
    This ensures that positional embeddings match the pixel padding layout.
    """
    def __init__(self, padding_width_latent: int):
        self.padding_width_latent = padding_width_latent
        if padding_width_latent == 0:
            logging.warning(f"[PositionalEmbeddingPadder] padding_width_latent is 0, no padding will be applied")
        
        # Face adjacency mapping (same as CubeMapPadder)
        self.face_map = {
            'F': {'top': ('U', 'bottom', 0, False), 'bottom': ('D', 'top', 0, False),
                  'left': ('L', 'right', 0, False), 'right': ('R', 'left', 0, False)},
            'R': {'top': ('U', 'right', -90, False), 'bottom': ('D', 'right', 90, False),
                  'left': ('F', 'right', 0, False), 'right': ('B', 'left', 0, False)},
            'B': {'top': ('U', 'top', 180, False), 'bottom': ('D', 'bottom', 180, False),
                  'left': ('R', 'right', 0, False), 'right': ('L', 'left', 0, False)},
            'L': {'top': ('U', 'left', 90, False), 'bottom': ('D', 'left', -90, False),
                  'left': ('B', 'right', 0, False), 'right': ('F', 'left', 0, False)},
            'U': {'top': ('B', 'top', 180, False), 'bottom': ('F', 'top', 0, False),
                  'left': ('L', 'top', -90, False), 'right': ('R', 'top', 90, False)},
            'D': {'top': ('F', 'bottom', 0, False), 'bottom': ('B', 'bottom', 180, False),
                  'left': ('L', 'bottom', 90, False), 'right': ('R', 'bottom', -90, False)}
        }

    def _extract_edge_strip(self, face_pos_embs: torch.Tensor, edge: str) -> torch.Tensor:
        """Extract edge strip from face positional embeddings. face_pos_embs: [T, H, W, dim]"""
        p = self.padding_width_latent
        if edge == 'top':
            return face_pos_embs[:, :p, :, :]
        elif edge == 'bottom':
            return face_pos_embs[:, -p:, :, :]
        elif edge == 'left':
            return face_pos_embs[:, :, :p, :]
        elif edge == 'right':
            return face_pos_embs[:, :, -p:, :]
        else:
            raise ValueError(f"Unknown edge {edge}")

    def pad_face_pos_embs(
        self, 
        face_pos_embs_dict: Dict[str, torch.Tensor], 
        face_id: str, 
        original_height_latent: int, 
        original_width_latent: int
    ) -> torch.Tensor:
        """
        Pad a single face's positional embeddings with data from adjacent faces.
        
        Args:
            face_pos_embs_dict: {face: [T, H, W, dim]} - positional embeddings for all faces
            face_id: Target face to pad ('F','R','B','L','U','D')
            original_height_latent: Original face height in latent space
            original_width_latent: Original face width in latent space
            
        Returns:
            [T, H+2p, W+2p, dim] - Padded positional embeddings
        """
        p = self.padding_width_latent
        if p == 0:
            return face_pos_embs_dict[face_id]
        
        face_pos_embs = face_pos_embs_dict[face_id]  # [T, H, W, dim]
        T, H, W, dim = face_pos_embs.shape
        
        if H > original_height_latent or W > original_width_latent:
            logging.info(f"Face: {face_id}, Shape: {face_pos_embs.shape}, Original shape: {original_height_latent}, {original_width_latent}")
            logging.info(f"Already padded, returning original face.")
            return face_pos_embs

        # Create padded tensor
        padded = torch.zeros(T, H + 2 * p, W + 2 * p, dim, device=face_pos_embs.device, dtype=face_pos_embs.dtype)
        padded[:, p:-p, p:-p, :] = face_pos_embs

        # Apply padding from adjacent faces
        for edge, (adj_face, adj_edge, rot, flip) in self.face_map[face_id].items():
            if adj_face not in face_pos_embs_dict:
                continue
                
            adj_pos_embs = face_pos_embs_dict[adj_face]
            strip = self._extract_edge_strip(adj_pos_embs, adj_edge)
            
            if rot != 0:
                # Rotate over spatial dims (H,W) - dims [T,H,W,dim], rotate over H,W
                k = (rot // 90) % 4
                strip = torch.rot90(strip, k=k, dims=[1, 2])
            if flip:
                # Flip over spatial dimensions
                if adj_edge in ['top', 'bottom']:
                    strip = torch.flip(strip, dims=[2])  # flip width
                else:
                    strip = torch.flip(strip, dims=[1])  # flip height
            
            # Place strip into padded tensor
            if edge == 'top':
                if strip.shape[2] != padded.shape[2]:
                    padded[:, :p, p:-p, :] = strip
                else:
                    padded[:, :p, :, :] = strip
            elif edge == 'bottom':
                if strip.shape[2] != padded.shape[2]:
                    padded[:, -p:, p:-p, :] = strip
                else:
                    padded[:, -p:, :, :] = strip
            elif edge == 'left':
                if strip.shape[1] != padded.shape[1]:
                    padded[:, p:-p, :p, :] = strip
                else:
                    padded[:, :, :p, :] = strip
            elif edge == 'right':
                if strip.shape[1] != padded.shape[1]:
                    padded[:, p:-p, -p:, :] = strip
                else:
                    padded[:, :, -p:, :] = strip
            else:
                raise ValueError(f"Unknown edge {edge}")

        return padded



class ContextPool:
    """
    Manages context pool for efficient autoregressive cube map video generation.
    
    Handles two types of context:
    1. Current local context: pixel content from adjacent faces with coverage in current time window
    2. History local context: content generated in previous time windows
    
    Maintains absolute positional coordinates for proper embedding assignment.
    """
    
    def __init__(self, cube_map_size: int = 512, padding_width: int = 16, max_history_windows: int = 2,
                 dit_patch_size: List[int] = [1, 2, 2], dit_precompted_freqs: torch.Tensor = None,
                 vae_spatial_compression: int = 16, vae_temporal_compression: int = 4, active_faces: List[str] = None,
                 use_vanilla_pos_embs: bool = False, window_length: int = 8,
                 fragment_future_context: bool = False,
                 coverage_threshold: float = 0.1,
                 always_full_context: bool = False,
                 use_global_sink_token: bool = False,
                 use_tiled_vae: bool = False):
        self.cube_map_size = cube_map_size
        self.padding_width = padding_width
        self.max_history_windows = max_history_windows
        self.window_length = window_length
        self.always_full_context = always_full_context
        self.use_global_sink_token = use_global_sink_token
        self.use_tiled_vae = use_tiled_vae
        
        # Set active faces
        if active_faces is None:
            self.active_faces = ['F', 'R', 'B', 'L', 'U', 'D']
        else:
            self.active_faces = active_faces
        
        # Face adjacency mapping
        self.face_adj_map = {
            'F': ['R', 'L', 'U', 'D'],
            'R': ['F', 'B', 'U', 'D'], 
            'B': ['R', 'L', 'U', 'D'],
            'L': ['F', 'B', 'U', 'D'],
            'U': ['F', 'R', 'B', 'L'],
            'D': ['F', 'R', 'B', 'L'],
        }
        
        # History storage: {face: List[(start_frame, end_frame, tensor)]}
        self.history: Dict[str, List[Tuple[int, int, torch.Tensor]]] = {
            f: [] for f in self.active_faces
        }

        self.dit_patch_size = dit_patch_size
        assert self.dit_patch_size[1] == self.dit_patch_size[2], "dit_patch_size[1] and dit_patch_size[2] should be the same"
        
        self.dit_precompted_freqs = dit_precompted_freqs
        self.vae_spatial_compression = vae_spatial_compression
        self.spatial_compression_factor = vae_spatial_compression * self.dit_patch_size[1]
        self.vae_temporal_compression = vae_temporal_compression
        
        # Track generation order for current timestep context
        self.current_order: List[Tuple[str, int, int]] = []
        self.current_order_idx: int = 0

        self.use_vanilla_pos_embs = use_vanilla_pos_embs
        
        # Fragment future context settings
        self.fragment_future_context = fragment_future_context
        self.coverage_threshold = coverage_threshold
        
        # Cache for detected fragments: {face: List[(start_frame, end_frame, avg_coverage)]}
        self.fragment_cache: Dict[str, List[Tuple[int, int, float]]] = {
            f: [] for f in self.active_faces
        }
        
        # Initialize positional embedding padder
        padding_width_latent = self.padding_width // self.spatial_compression_factor
        self.padder_pos_emb = PositionalEmbeddingPadder(padding_width_latent)
        
        # Global sink token storage (will be populated after first window is generated)
        self.global_sink_token_latents = None  # [B, C, 1, H, W] - first frame of all 6 faces
        self.global_sink_token_pos_embs = None  # [1, H, W_total, dim] - pos embs for first frame

        
    def clear_history(self):
        """Clear all history."""
        for face in self.history:
            self.history[face] = []
        for face in self.fragment_cache:
            self.fragment_cache[face] = []
        self.current_order = []
        self.current_order_idx = 0
        # Clear global sink token
        self.global_sink_token_latents = None
        self.global_sink_token_pos_embs = None
    
    def set_generation_order(self, order: List[Tuple[str, int, int]], order_idx: int = 0):
        """Set the generation order for current inference/training step."""
        self.current_order = order
        self.current_order_idx = order_idx
    
    def advance_order(self):
        """Advance to next item in generation order."""
        self.current_order_idx += 1
    
    
    def _build_global_sink_token(
        self,
        vae,
        device: torch.device,
        torch_dtype: torch.dtype,
        actual_resolution: int = None
    ):
        """
        Build global sink token from the first frame of all faces in history.
        This should be called when current_start > 0 (not the first window).
        
        Args:
            vae: VAE encoder
            device: Target device
            torch_dtype: Target dtype
            actual_resolution: Actual resolution of the content
            
        Returns:
            None (updates self.global_sink_token_latents and self.global_sink_token_pos_embs)
        """
        if not self.use_global_sink_token:
            return
        
        # Check if global sink token is already built
        if self.global_sink_token_latents is not None:
            return
        
        # Check if we have the first frame for all faces in history
        first_frame_available = True
        for face in self.active_faces:
            if face not in self.history or len(self.history[face]) == 0:
                first_frame_available = False
                break
            # Check if history contains frame 0
            has_frame_0 = False
            for start_frame, end_frame, _ in self.history[face]:
                if start_frame == 0:
                    has_frame_0 = True
                    break
            if not has_frame_0:
                first_frame_available = False
                break
        
        if not first_frame_available:
            logging.info("[Global Sink Token] First frame not available for all faces yet, skipping")
            return
        
        # Use actual resolution if provided
        if actual_resolution is None:
            actual_resolution = self.cube_map_size
        
        H = actual_resolution
        W = actual_resolution * len(self.active_faces)
        
        # Extract first frame from all faces
        first_frame_content = None
        for face_idx, face in enumerate(self.active_faces):
            # Get first frame content from history
            face_first_frame = None
            for start_frame, end_frame, content in self.history[face]:
                if start_frame == 0:
                    # Extract first frame: content shape is [T, C, H, W]
                    face_first_frame = content[0:1]  # [1, C, H, W]
                    break
            
            if face_first_frame is None:
                logging.warning(f"[Global Sink Token] First frame not found for face {face}, aborting")
                return
            
            # Initialize or concatenate
            if first_frame_content is None:
                C = face_first_frame.shape[1]
                first_frame_content = torch.zeros((1, C, H, W), device=device, dtype=torch_dtype)
            
            # Place this face's first frame in the appropriate location
            w_start = face_idx * H
            w_end = w_start + H
            first_frame_content[:, :, :, w_start:w_end] = face_first_frame.to(device=device, dtype=torch_dtype)
        
        # Encode to latents: [1, C, H, W] -> [B, C, T, H, W] -> [B, C_latent, T_latent, H_latent, W_latent]
        first_frame_bcthw = first_frame_content.permute(1, 0, 2, 3).unsqueeze(0)  # [B=1, C, T=1, H, W]
        
        self.global_sink_token_latents = vae.encode(
            first_frame_bcthw.to(device=device, dtype=torch_dtype),
            tiled=self.use_tiled_vae,
            device=device
        ).to(torch_dtype)  # [B, C_latent, T_latent, H_latent, W_latent]
        
        # Build positional embeddings for first frame (t=0)
        # We need pos embs for all 6 faces at t=0
        pos_embs_list = []
        for face in self.active_faces:
            # Compute pos embs for this face at t=0 to t=1 (single frame in latent space)
            # Since VAE temporal compression is 4, first frame maps to t_latent=0
            face_pos_embs = self.compute_absolute_latent_pos_embs(
                face, 0, 1, with_padding=False, actual_resolution=actual_resolution
            )  # [T_latent=1, H_latent, W_latent, dim]
            pos_embs_list.append(face_pos_embs)
        
        # Concatenate along width dimension to form horizontal layout
        self.global_sink_token_pos_embs = torch.cat(pos_embs_list, dim=2)  # [T_latent=1, H_latent, W_total_latent, dim]
        
        logging.info(f"[Global Sink Token] Built global sink token from first frame: "
                    f"latents shape={self.global_sink_token_latents.shape}, "
                    f"pos_embs shape={self.global_sink_token_pos_embs.shape}")
    
    
    
    
    def get_current_face_clean_latents_with_padding(
        self,
        current_face: str,
        current_start: int,
        current_end: int,
        cond_cubemap: Dict[str, torch.Tensor],
        vae,
        device: torch.device,
        torch_dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Get clean latents for current face with padding from adjacent faces.
        This is used to initialize the denoising process with condition-based latents
        rather than pure noise.
        
        IMPORTANT: This method operates in PIXEL SPACE before VAE encoding because
        VAE latent rotation is NOT equivalent to pixel space rotation. We must:
        1. Extract pixel content for current face and adjacent faces
        2. Apply padding in pixel space using CubeMapPadder logic
        3. Encode the padded pixel content to latents via VAE
        
        Args:
            current_face: Current face being generated ('F','R','B','L','U','D')
            current_start: Start frame of current generation window (absolute)
            current_end: End frame of current generation window (absolute)
            cond_cubemap: Conditional cube map data {face: [T, C, H, W]} in pixel space
            vae: VAE encoder
            device: Target device
            torch_dtype: Target dtype
            
        Returns:
            clean_latents_with_padding: [B, C_latent, T_latent, H_latent+2p, W_latent+2p]
        """
        if cond_cubemap is None or current_face not in cond_cubemap:
            return None
        
        # Extract current time window for all faces in pixel space
        face_contents = {}
        for face in self.active_faces:
            if face not in cond_cubemap:
                continue
            # Extract: [T, C, H, W]
            face_content = cond_cubemap[face][current_start:current_end]
            face_contents[face] = face_content
        
        if current_face not in face_contents:
            return None
        
        # Now apply padding in PIXEL SPACE using CubeMapPadder logic
        # This is crucial because VAE rotation != pixel rotation
        current_content = face_contents[current_face]  # [T, C, H, W]
        T, C, H, W = current_content.shape
        
        p = self.padding_width
        
        if p == 0:
            # No padding, just encode current content
            current_bcthw = current_content.permute(1, 0, 2, 3).unsqueeze(0)  # [B, C, T, H, W]
            return vae.encode(
                current_bcthw.to(device=device, dtype=torch_dtype),
                tiled=self.use_tiled_vae,
                device=device
            ).to(torch_dtype)
        
        # Create padded tensor in pixel space
        padded = torch.zeros(T, C, H + 2 * p, W + 2 * p, 
                            device=current_content.device, dtype=current_content.dtype)
        padded[:, :, p:-p, p:-p] = current_content
        
        # Face adjacency mapping (same as CubeMapPadder)
        face_map = {
            'F': {'top': ('U', 'bottom', 0, False), 'bottom': ('D', 'top', 0, False),
                  'left': ('L', 'right', 0, False), 'right': ('R', 'left', 0, False)},
            'R': {'top': ('U', 'right', -90, False), 'bottom': ('D', 'right', 90, False),
                  'left': ('F', 'right', 0, False), 'right': ('B', 'left', 0, False)},
            'B': {'top': ('U', 'top', 180, False), 'bottom': ('D', 'bottom', 180, False),
                  'left': ('R', 'right', 0, False), 'right': ('L', 'left', 0, False)},
            'L': {'top': ('U', 'left', 90, False), 'bottom': ('D', 'left', -90, False),
                  'left': ('B', 'right', 0, False), 'right': ('F', 'left', 0, False)},
            'U': {'top': ('B', 'top', 180, False), 'bottom': ('F', 'top', 0, False),
                  'left': ('L', 'top', -90, False), 'right': ('R', 'top', 90, False)},
            'D': {'top': ('F', 'bottom', 0, False), 'bottom': ('B', 'bottom', 180, False),
                  'left': ('L', 'bottom', 90, False), 'right': ('R', 'bottom', -90, False)}
        }
        
        # Helper function to extract edge strip in pixel space
        def extract_edge_strip(face_tensor: torch.Tensor, edge: str) -> torch.Tensor:
            """Extract edge strip from face tensor. face_tensor: [T, C, H, W]"""
            if edge == 'top':
                return face_tensor[:, :, :p, :]
            elif edge == 'bottom':
                return face_tensor[:, :, -p:, :]
            elif edge == 'left':
                return face_tensor[:, :, :, :p]
            elif edge == 'right':
                return face_tensor[:, :, :, -p:]
            else:
                raise ValueError(f"Unknown edge {edge}")
        
        # Apply padding from adjacent faces in pixel space
        for edge, (adj_face, adj_edge, rot, flip) in face_map[current_face].items():
            if adj_face not in face_contents:
                continue
            
            adj_content = face_contents[adj_face]
            strip = extract_edge_strip(adj_content, adj_edge)
            
            # Apply rotation if needed (rotate over spatial dims H, W which are dims 2, 3)
            if rot != 0:
                k = (rot // 90) % 4
                strip = torch.rot90(strip, k=k, dims=[2, 3])
            
            # Apply flip if needed
            if flip:
                if adj_edge in ['top', 'bottom']:
                    strip = torch.flip(strip, dims=[3])  # flip width
                else:
                    strip = torch.flip(strip, dims=[2])  # flip height
            
            # Place strip into padded tensor
            if edge == 'top':
                if strip.shape[3] != padded.shape[3]:
                    padded[:, :, :p, p:-p] = strip
                else:
                    padded[:, :, :p, :] = strip
            elif edge == 'bottom':
                if strip.shape[3] != padded.shape[3]:
                    padded[:, :, -p:, p:-p] = strip
                else:
                    padded[:, :, -p:, :] = strip
            elif edge == 'left':
                if strip.shape[2] != padded.shape[2]:
                    padded[:, :, p:-p, :p] = strip
                else:
                    padded[:, :, :, :p] = strip
            elif edge == 'right':
                if strip.shape[2] != padded.shape[2]:
                    padded[:, :, p:-p, -p:] = strip
                else:
                    padded[:, :, :, -p:] = strip
            else:
                raise ValueError(f"Unknown edge {edge}")
        
        # Now encode the padded pixel content to latents
        padded_bcthw = padded.permute(1, 0, 2, 3).unsqueeze(0)  # [B, C, T, H+2p, W+2p]
        
        clean_latents_with_padding = vae.encode(
            padded_bcthw.to(device=device, dtype=torch_dtype),
            tiled=self.use_tiled_vae,
            device=device
        ).to(torch_dtype)  # [B, C_latent, T_latent, H_latent+2p, W_latent+2p]
        
        return clean_latents_with_padding
    
    def crop_padding(self, content: torch.Tensor) -> torch.Tensor:
        """
        Crop padding from content tensor.
        
        Args:
            content: Content tensor with shape [T, C, H, W] that may have padding
            
        Returns:
            Cropped content tensor with shape [T, C, H_cropped, W_cropped]
        """
        if self.padding_width == 0:
            return content
        
        # Check if the content has padding by examining if dimensions are consistent with having padding
        # We check if: actual_size = some_base_size + 2*padding_width
        # This is more flexible for multi-resolution training
        actual_h, actual_w = content.shape[-2], content.shape[-1]
        p = self.padding_width
        
        # Calculate what the base size would be if this content has padding
        expected_base_h = actual_h - 2 * p
        expected_base_w = actual_w - 2 * p
        
        # If removing padding gives us a reasonable size (positive and matches both dimensions)
        # then we assume it has padding
        if expected_base_h > 0 and expected_base_w > 0 and expected_base_h == expected_base_w:
            # Check if this looks like it has padding (dimensions are larger than typical base sizes)
            # and the base size is divisible by 32 (typical for cube map sizes)
            if expected_base_h % 32 == 0:
                # Has padding, crop it
                return content[:, :, p:-p, p:-p]
        
        # No padding detected, return as is
        return content
            
    def add_to_history(self, face: str, start_frame: int, end_frame: int, content: torch.Tensor):
        """
        Add generated content to history.

        Args:
            face: Face name ('F', 'R', 'B', 'L', 'U', 'D')
            start_frame: Starting frame index (absolute in full video)
            end_frame: Ending frame index (absolute in full video)
            content: Generated content tensor [T, C, H, W] (without padding)
        """
        self.history[face].append((start_frame, end_frame, content.clone()))

    def set_context_from_generated(self, face: str, start_frame: int, end_frame: int,
                                   pixel_tensor: torch.Tensor):
        """
        Inject a student-generated pixel-space face into history, replacing any
        overlapping entry.  Used during distillation rollout training so that
        subsequent faces receive the student's own (lower-quality) output as
        context rather than ground-truth, thereby correcting exposure bias.

        In non-latent mode the history stores pixel-space tensors [T, C, H, W].
        build_context_latents() will VAE-encode them on demand, so we just need
        to insert the decoded pixel frames here.

        Args:
            face:         Face name ('F', 'R', 'B', 'L', 'U', 'D')
            start_frame:  Starting pixel-frame index (absolute, inclusive)
            end_frame:    Ending   pixel-frame index (absolute, exclusive)
            pixel_tensor: Student-decoded content [T, C, H, W] in pixel space,
                          without padding.  Will be detached and stored on CPU.
        """
        if face not in self.history:
            logging.warning(f"[set_context_from_generated] Face '{face}' not in "
                            f"history dict – ignoring.")
            return

        # Remove existing entries that overlap with [start_frame, end_frame)
        self.history[face] = [
            (s, e, t) for s, e, t in self.history[face]
            if not (s < end_frame and e > start_frame)
        ]
        # Store on CPU, consistent with how add_to_history is used elsewhere
        self.history[face].append(
            (start_frame, end_frame, pixel_tensor.detach().cpu())
        )
        logging.debug(
            f"[set_context_from_generated] Injected generated face '{face}' "
            f"[{start_frame}, {end_frame}) shape={tuple(pixel_tensor.shape)} "
            f"into context history."
        )
        
        # Keep only recent windows
        # if len(self.history[face]) > self.max_history_windows:
        #     self.history[face] = self.history[face][-self.max_history_windows:]
    
    def get_face_content_from_history(self, face: str, start_frame: int, end_frame: int) -> Optional[torch.Tensor]:
        """
        Get face content from history for the specified time range.
        
        Args:
            face: Face name ('F', 'R', 'B', 'L', 'U', 'D')
            start_frame: Starting frame index (absolute in full video)
            end_frame: Ending frame index (absolute in full video)
            
        Returns:
            Content tensor [T, C, H, W] or None if not found
        """
        if face not in self.history or len(self.history[face]) == 0:
            return None
        
        # Find the history entry that overlaps with [start_frame, end_frame)
        for hist_start, hist_end, hist_content in self.history[face]:
            # Check if the history entry overlaps with requested range
            if hist_start <= start_frame and hist_end >= end_frame:
                # Extract the requested slice
                content_start = start_frame - hist_start
                content_end = content_start + (end_frame - start_frame)
                return hist_content[content_start:content_end].clone()
        
        return None
    
    def update_history(self, face: str, start_frame: int, end_frame: int, updated_content: torch.Tensor):
        """
        Update face content in history for the specified time range.
        If the entry doesn't exist, add it as a new entry.
        
        Args:
            face: Face name ('F', 'R', 'B', 'L', 'U', 'D')
            start_frame: Starting frame index (absolute in full video)
            end_frame: Ending frame index (absolute in full video)
            updated_content: Updated content tensor [T, C, H, W] (without padding)
        """
        if face not in self.history:
            return
        
        # Find the history entry that overlaps with [start_frame, end_frame)
        for i, (hist_start, hist_end, hist_content) in enumerate(self.history[face]):
            # Check if the history entry overlaps with requested range
            if hist_start <= start_frame and hist_end >= end_frame:
                # Update the content in place
                content_start = start_frame - hist_start
                content_end = content_start + (end_frame - start_frame)
                hist_content[content_start:content_end] = updated_content.clone()
                # Update the tuple with the modified content
                self.history[face][i] = (hist_start, hist_end, hist_content)
                return
        
        # If not found in existing history, add as new entry
        self.add_to_history(face, start_frame, end_frame, updated_content)

    def detect_fragments(self, mask_cubemap: Dict[str, torch.Tensor]):
        """
        Detect continuous fragments across all cube faces based on coverage threshold.
        
        Args:
            mask_cubemap: Dict mapping face names to mask tensors [T, C, H, W] where 1=content, 0=empty
        """
        if not self.fragment_future_context:
            logging.warning(f"fragment_future_context is disabled. Detecting fragments is disabled.")
            return
        
        # Debug: check mask_cubemap content
        if not mask_cubemap:
            logging.warning(f"mask_cubemap is empty!")
            return
        
        # logging.info(f"Processing mask_cubemap with faces: {list(mask_cubemap.keys())}")
            
        # Clear existing fragment cache
        for face in self.fragment_cache:
            self.fragment_cache[face] = []

        # logging.info(f"Cleared fragment cache")
            
        for face in self.active_faces:
            if face not in mask_cubemap:
                logging.info(f"Face {face} not in mask_cubemap, skipping")
                continue
                
            mask = mask_cubemap[face]  # [T, C, H, W]
            T = mask.shape[0]
            
            # Calculate per-frame coverage
            # Handle both binary (0/1) and continuous (0-1) masks
            per_frame_coverage = []
            for t in range(T):
                frame_mask = mask[t]  # [C, H, W]
                # Use mean of mask values as coverage (works for both binary and continuous masks)
                coverage = frame_mask.mean().item()
                per_frame_coverage.append(coverage)
            
            # Detect continuous fragments where coverage stays above threshold
            fragments = []
            start_frame = None
            
            for t, coverage in enumerate(per_frame_coverage):
                if coverage >= self.coverage_threshold:
                    if start_frame is None:
                        start_frame = t
                        # logging.info(f"Face {face}: Fragment starts at frame {t}, coverage={coverage:.3f}")
                else:
                    if start_frame is not None:
                        # End of a fragment
                        end_frame = t
                        avg_coverage = sum(per_frame_coverage[start_frame:end_frame]) / (end_frame - start_frame)
                        fragments.append((start_frame, end_frame, avg_coverage))
                        # logging.info(f"Face {face}: Fragment detected [{start_frame}, {end_frame}), avg_coverage={avg_coverage:.3f}")
                        start_frame = None
            
            # Handle case where last fragment extends to end of video
            if start_frame is not None:
                end_frame = T
                avg_coverage = sum(per_frame_coverage[start_frame:end_frame]) / (end_frame - start_frame)
                fragments.append((start_frame, end_frame, avg_coverage))
                
            self.fragment_cache[face] = fragments
            
    def get_fragment_future_context(
        self,
        current_face: str,
        current_end: int,
        cond_cubemap: Dict[str, torch.Tensor],
        vae,
        device: torch.device,
        torch_dtype: torch.dtype,
        save_fragments: bool = False,
        save_dir: Optional[str] = None,
        save_prefix: str = "fragment",
        step: Optional[int] = None,
        rank: Optional[int] = None,
        order_idx: Optional[int] = None,
        actual_resolution: int = None
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get future fragments for the current face and adjacent faces beyond the current time window.
        
        Strategy:
        - Get the nearest future fragment for the current face
        - Get the nearest future fragment for each adjacent face
        
        Args:
            current_face: Face currently being generated
            current_end: End frame of current generation window (absolute)
            cond_cubemap: Conditional cube map data
            vae: VAE encoder for converting to latents
            device: Target device
            torch_dtype: Target dtype
            save_fragments: Whether to save fragment videos for debugging
            save_dir: Directory to save fragment videos
            save_prefix: Prefix for fragment video filenames
            step: Training/test step number
            rank: Process rank (for multi-GPU)
            order_idx: Current order index
            actual_resolution: Actual resolution of the content
            
        Returns:
            List of (fragment_latents, fragment_pos_embs) tuples for future fragments
        """
        future_fragments = []
        
        if not self.fragment_future_context:
            logging.warning(f"fragment_future_context is disabled")
            return future_fragments
        
        # Collect faces to process: current face + adjacent faces
        faces_to_process = [current_face]
        if current_face in self.face_adj_map:
            adjacent_faces = self.face_adj_map[current_face]
            # Only include adjacent faces that are in active_faces
            faces_to_process.extend([f for f in adjacent_faces if f in self.active_faces])
        
        # Process each face (current + adjacent)
        for face_name in faces_to_process:
            # Check if this face has fragments and conditional data
            if face_name not in self.fragment_cache:
                continue
            if face_name not in cond_cubemap:
                continue
            
            face_cond = cond_cubemap[face_name]  # [T, C, H, W]
            face_fragments = self.fragment_cache[face_name]
            
            # Find the nearest future fragment for this face
            nearest_fragment = None
            for start_frame, end_frame, avg_coverage in face_fragments:
                # Skip fragments that are completely in the past or current window
                if end_frame <= current_end:
                    continue
                
                # This is a future fragment, record it
                nearest_fragment = (start_frame, end_frame, avg_coverage)
                break  # Take only the first (nearest) future fragment
            
            if nearest_fragment is None:
                continue  # No future fragment found for this face
            
            start_frame, end_frame, avg_coverage = nearest_fragment
            
            # Calculate the future portion of this fragment
            frag_start = max(start_frame, current_end)
            actual_length = end_frame - frag_start
            
            # If less than 4 frames, skip this fragment (too short for VAE temporal compression)
            if actual_length < 4:
                continue

            # Ensure length satisfies 4N+1 constraint for VAE temporal processing
            if ((actual_length - 1) % 4) != 0:
                max_possible = face_cond.shape[0] - frag_start
                # Try to floor to nearest valid length
                down_k = nearest_4n1(actual_length, "down", max_possible)
                if down_k is not None and down_k >= 4:
                    actual_length = down_k
                else:
                    # No valid length available, skip this fragment
                    continue

            # Set the fragment range
            frag_end = frag_start + actual_length
            
            # Extract fragment content
            fragment_content = face_cond[frag_start:frag_end]
            
            # Convert to VAE latents
            fragment_bcthw = fragment_content.permute(1, 0, 2, 3).unsqueeze(0)
            fragment_latents = vae.encode(
                fragment_bcthw.to(device=device, dtype=torch_dtype),
                tiled=self.use_tiled_vae,
                device=device
            ).to(torch_dtype)
            
            # Compute positional embeddings for this fragment (using face_name, not current_face)
            fragment_pos_embs = self.compute_absolute_latent_pos_embs(
                face_name, frag_start, frag_end, with_padding=False, actual_resolution=actual_resolution
            )
            
            future_fragments.append((fragment_latents, fragment_pos_embs))
            
            # Save fragment video for debugging
            if save_fragments and save_dir:
                face_label = "current" if face_name == current_face else "adjacent"
                filename_parts = [save_prefix]
                if step is not None:
                    filename_parts.append(f"step{step}")
                if rank is not None:
                    filename_parts.append(f"rank{rank}")
                if order_idx is not None:
                    filename_parts.append(f"order{order_idx}")
                filename_parts.append(f"{face_name}")
                filename_parts.append(f"{face_label}")
                filename_parts.append(f"t{frag_start}-{frag_end}")
                filename_parts.append(f"cov{avg_coverage:.2f}")
                
                custom_prefix = "_".join(filename_parts)
                _save_context_video(
                    fragment_bcthw.detach().cpu(), 
                    save_dir, 
                    custom_prefix,
                    step=None,  # Already included in prefix
                    rank=None,
                    face=None,
                    order_idx=None
                )
                logging.info(f"Saved {face_label} fragment for face {face_name}: [{frag_start}, {frag_end}), coverage={avg_coverage:.3f}")
        
        if save_fragments and save_dir:
            if len(future_fragments) > 0:
                logging.info(f"Total {len(future_fragments)} future fragments extracted "
                           f"(current face: {current_face}, processed faces: {faces_to_process}), "
                           f"current_end={current_end}")
            else:
                logging.info(f"No future fragments found for current face {current_face} and adjacent faces, "
                           f"current_end={current_end}")
        
        return future_fragments

    def _compute_padded_current_face_pos_embs(
        self, 
        current_face: str, 
        current_start: int, 
        current_end: int, 
        all_faces_pos_embs: Dict[str, torch.Tensor],
        min_time: int,
        max_time: int,
        actual_resolution: int = None
    ) -> torch.Tensor:
        """
        Compute padded positional embeddings for the current face by reusing context pos embs.
        
        Args:
            current_face: Current face being generated
            current_start: Start frame of current generation window
            current_end: End frame of current generation window  
            all_faces_pos_embs: Dict of all faces' pos embs {face: [T, H, W, dim]}
            min_time: Minimum time in the context
            max_time: Maximum time in the context
            actual_resolution: Actual resolution of the content (for multi-resolution training)
            
        Returns:
            Padded positional embeddings [T_current, H+2p, W+2p, dim]
        """
        # Extract the time slice for current generation window from all faces
        t_start_rel = (current_start // 4) - (min_time // 4)
        t_end_rel = ((current_end - 1) // 4 + 1) - (min_time // 4)
        
        # Get cube map size in latent space (without padding)
        if actual_resolution is None:
            actual_resolution = self.cube_map_size
        cube_map_latent_size = actual_resolution // self.spatial_compression_factor
        original_height_latent = cube_map_latent_size
        original_width_latent = cube_map_latent_size
        
        # Extract time slices for all faces  
        current_window_faces_pos_embs = {}
        for face_name, face_pos_embs in all_faces_pos_embs.items():
            # Extract current time window: [T_current, H, W, dim]
            current_window_faces_pos_embs[face_name] = face_pos_embs[t_start_rel:t_end_rel, :, :, :]
        
        # Apply padding using the padder
        padded_current_face_pos_embs = self.padder_pos_emb.pad_face_pos_embs(
            current_window_faces_pos_embs, 
            current_face, 
            original_height_latent, 
            original_width_latent
        )
        
        return padded_current_face_pos_embs
    
    def simulate_latent_from_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simulate VAE latent from image tensor without applying real VAE. Used for converting single-channel binary mask to latent space.
        input: [B, C, T, H, W]
        output: [B, C * 16, T // 4, H // 16, W // 16] (Compression rate = 64x)
        """
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1, 1)

        target_height = x.shape[-2] // self.vae_spatial_compression
        target_width = x.shape[-1] // self.vae_spatial_compression
        # (1) Apply VAE spatial to channel conversion
        vae_patch_size = 2
        x = rearrange(x,
                      "b c f (h q) (w r) -> b (c r q) f h w",
                      q=vae_patch_size, r=vae_patch_size)

        import torch.nn.functional as F
        x = F.interpolate(
            x, size=(x.shape[-3], target_height, target_width),
            mode='nearest'
        )  # [B, C=4, T, H', W']
        
        # (2) Apply VAE temporal-to-channel conversion
        # First frame is copied 4 times for first temporal latent position, then other frames follow
        if x.shape[2] == 1:
            # Only one frame, repeat it 4 times in channel dimension
            x = torch.repeat_interleave(x, repeats=4, dim=2)  # [B, C=4, T, H', W']
        else:
            # Multiple frames: first frame repeated 4 times, others as single channels
            x = torch.concat([torch.repeat_interleave(x[:, :, 0:1], repeats=4, dim=2), x[:, :, 1:]], dim=2)

        x = x.view(x.shape[0], x.shape[1], x.shape[2] // 4, 4, x.shape[3], x.shape[4])
        x = rearrange(x, "b c t1 t2 h w -> b (c t2) t1 h w")
        return x
    
    def build_context_latents(
        self,
        current_face: str,
        current_start: int, 
        current_end: int,
        cond_cubemap: Dict[str, torch.Tensor],
        mask_cubemap: Dict[str, torch.Tensor],
        vae,
        device: torch.device,
        torch_dtype: torch.dtype,
        save_context: bool = False,
        save_dir: Optional[str] = None,
        save_prefix: str = "context",
        step: Optional[int] = None,
        rank: Optional[int] = None,
        order_idx: Optional[int] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Build context latents with proper context window length management.
        
        Context Window Strategy (when always_full_context=False):
        - Context spans [min_time, max_time) where max_time = current_end
        - Context length is limited to: max_history_windows * current_window_length
        - min_time = max(0, current_end - max_context_length)
        - This ensures context includes current window + limited history
        - Future content beyond current_end is handled separately via fragment_future_context
        
        Context Window Strategy (when always_full_context=True):
        - Context spans [0, total_video_length) for all face generations
        - No length limitation from max_history_windows
        - All future perspective video content is included in context (no fragment_future_context)
        - Context is gradually updated from perspective video to generated face content
        
        Context Sources:
        1. self.history: All historical content (previous windows + previous faces in current window)
           - Only history within [min_time, max_time) is included
           - History entries are clipped to fit the context window
        2. cond_cubemap: Conditional content for context window [min_time, max_time)
           - Used to fill gaps where history doesn't exist
           - When always_full_context=True, includes ALL frames from perspective video
        
        The difference between training and inference is handled by the calling code:
        - Training: GT data is added to history via add_to_history() in preprocessing
        - Inference: History gets populated naturally during generation
        
        Returns:
            context_latents: Encoded context latents or None if no context
            context_pos_embs: Context positional embeddings
            current_face_pos_embs: Current face positional embeddings (with padding)
            current_face_cond_latents: Current face conditional latents for fuse_vae_embedding_in_latents
            current_face_cond_mask: Current face conditional mask in latent space
            fragment_future_context: List of (fragment_latents, fragment_pos_embs) for future fragments
        """
        # logging.info(f"Building context latents for [order_idx={order_idx}] face {current_face} from {current_start} to {current_end}")
        # Build face horizon layout using active faces
        slot_map = {f_name: idx for idx, f_name in enumerate(self.active_faces)}
        
        # Get sample tensor to determine dimensions
        sample_tensor = None
        if cond_cubemap:
            sample_tensor = next(iter(cond_cubemap.values()))
        
        if sample_tensor is None:
            return None, None, None, None, None, None
            
        # sample_tensor: [T, C, H, W]
        C = sample_tensor.shape[1]
        # Use actual resolution from input data (for multi-resolution training support)
        actual_resolution = sample_tensor.shape[2]  # H dimension
        H = actual_resolution
        W = actual_resolution * len(self.active_faces)

        # Calculate context time range with length limit
        if self.always_full_context:
            # Always full context mode: context spans entire video length
            # No limitation from max_history_windows
            # All future perspective video content is included
            min_time = 0
            max_time = sample_tensor.shape[0]  # Total video length
            T_total_context = max_time - min_time
            logging.info(f"[always_full_context=True] Face={current_face}, Current=[{current_start},{current_end}), "
                        f"Context=[{min_time},{max_time}), Context_len={T_total_context}")
        else:
            # Standard mode: limited context window
            # Context includes: current window + history windows (up to max_history_windows)
            # Future content beyond current_end is handled by fragment_future_context
            current_window_length = current_end - current_start
            max_context_length = self.max_history_windows * current_window_length
            
            # Context spans from min_time to current_end (not beyond)
            # Go back at most max_context_length frames from current_end
            min_time = max(0, current_end - max_context_length)
            max_time = current_end
            T_total_context = max_time - min_time
        
        # Initialize context video
        context_video = torch.zeros((T_total_context, C, H, W), device=device, dtype=torch_dtype)
        # logging.info(f"context_video shape: {context_video.shape}, time range: [{min_time}, {max_time})")
        
        if self.always_full_context:
            # Always full context mode: match historical full context implementation
            # 1. Fill all historical content from self.history
            for face in self.active_faces:
                if face in self.history:
                    slot_idx = slot_map[face]
                    w_start = slot_idx * H
                    w_end = w_start + H
                    
                    for start_frame, end_frame, content in self.history[face]:
                        t_start_rel = start_frame - min_time
                        t_end_rel = end_frame - min_time
                        # Only place if within our context video bounds
                        if t_start_rel >= 0 and t_end_rel <= T_total_context:
                            context_video[t_start_rel:t_end_rel, :, :, w_start:w_end] = content.to(device=device, dtype=torch_dtype)
            
            # 2. Fill with cond_cubemap for all faces (perspective video)
            # For each face: check if current window is already filled by history
            for _face in self.active_faces:
                if _face not in cond_cubemap:
                    continue
                    
                slot_idx = slot_map[_face]
                w_start = slot_idx * H
                w_end = w_start + H
                t_start_rel = current_start - min_time
                t_end_rel = current_end - min_time
                
                # Check if current window [current_start, current_end) is already filled by history
                if context_video[t_start_rel:t_end_rel, :, :, w_start:w_end].sum() == 0:
                    # Not filled by history, fill from current_start onwards
                    cond_content = cond_cubemap[_face][current_start:]
                    context_video[t_start_rel:, :, :, w_start:w_end] = cond_content.to(device=device, dtype=torch_dtype)
                else:
                    # Already filled by history, only fill future windows from current_end onwards
                    cond_content = cond_cubemap[_face][current_end:]
                    context_video[t_end_rel:, :, :, w_start:w_end] = cond_content.to(device=device, dtype=torch_dtype)
        else:
            # Standard mode: fill history first, then fill gaps with cond_cubemap
            # 1. Fill historical content from self.history (only within [min_time, max_time))
            for face in self.active_faces:
                if face in self.history:
                    slot_idx = slot_map[face]
                    w_start = slot_idx * H
                    w_end = w_start + H
                    
                    for start_frame, end_frame, content in self.history[face]:
                        # Only include history that overlaps with [min_time, max_time)
                        if end_frame <= min_time or start_frame >= max_time:
                            continue  # Outside context window, skip
                        
                        # Clip to context window bounds
                        clipped_start = max(start_frame, min_time)
                        clipped_end = min(end_frame, max_time)
                        
                        # Calculate relative indices in context video
                        t_start_rel = clipped_start - min_time
                        t_end_rel = clipped_end - min_time
                        
                        # Calculate corresponding slice in content tensor
                        content_start = clipped_start - start_frame
                        content_end = content_start + (clipped_end - clipped_start)
                        
                        # logging.info(f"Adding history content for face {face} from {clipped_start} (rel{t_start_rel}) to {clipped_end} (rel{t_end_rel})")
                        context_video[t_start_rel:t_end_rel, :, :, w_start:w_end] = content[content_start:content_end].to(device=device, dtype=torch_dtype)
            
            # 2. Fill with cond_cubemap for current generation window [current_start, current_end) where not already filled by history
            # This includes both the period before current window (if min_time < current_start) and the current window
            for _face in self.active_faces:
                if _face not in cond_cubemap:
                    continue
                    
                slot_idx = slot_map[_face]
                w_start = slot_idx * H
                w_end = w_start + H
                
                # Extract the relevant slice from cond_cubemap for this context window
                t_start_rel = current_start - min_time
                t_end_rel = current_end - min_time
                
                # Only add if not already filled by history (only fill in current face and other ungenerated faces in current window)
                if context_video[t_start_rel:, :, :, w_start:w_end].sum() == 0:
                    # Have not been filled by history, put cond content at current time window and future windows
                    cond_content = cond_cubemap[_face][current_start:current_end]
                    context_video[t_start_rel:, :, :, w_start:w_end] = cond_content.to(device=device, dtype=torch_dtype)
        
        context_bcthw = context_video.permute(1, 0, 2, 3).unsqueeze(0)

        if self.use_tiled_vae:
            logging.info(f"[Tiled VAE] Encoding context with shape {context_bcthw.shape}")
            
        context_latents = vae.encode(
            context_bcthw.to(device=device, dtype=torch_dtype), 
            tiled=self.use_tiled_vae, 
            tile_size=(self.cube_map_size, self.cube_map_size), 
            tile_stride=(self.cube_map_size, self.cube_map_size),
            device=device
        ).to(torch_dtype)  # (B, C, T', H', W')
        
        # Build positional embeddings along the width dimension
        context_pos_embs = []
        all_faces_pos_embs = {}  # Store individual face pos embs for reuse
        for _face in self.active_faces:
            current_freqs = self.compute_absolute_latent_pos_embs(
                _face, min_time, max_time, with_padding=False, actual_resolution=actual_resolution
            )
            context_pos_embs.append(current_freqs)
            all_faces_pos_embs[_face] = current_freqs

        context_pos_embs = torch.cat(context_pos_embs, dim=2)
        t_lat = context_latents.shape[2]
        t_pos = context_pos_embs.shape[0]
        if t_pos > t_lat:
            logging.info(
                f"[ContextPool] Trimming context_pos_embs temporal dim from {t_pos} to {t_lat} "
                f"to match context_latents (min_time={min_time}, max_time={max_time})"
            )
            context_pos_embs = context_pos_embs[:t_lat]
        elif t_pos < t_lat:
            logging.info(
                f"[ContextPool] Padding context_pos_embs temporal dim from {t_pos} to {t_lat} "
                f"to match context_latents (min_time={min_time}, max_time={max_time})"
            )
            pad = context_pos_embs[-1:].expand(t_lat - t_pos, -1, -1, -1)
            context_pos_embs = torch.cat([context_pos_embs, pad], dim=0)
        
        # Build global sink token if enabled and not the first window
        if self.use_global_sink_token and current_start > 0:
            self._build_global_sink_token(vae, device, torch_dtype, actual_resolution=actual_resolution)
            
            # Prepend global sink token to context if available
            if self.global_sink_token_latents is not None and self.global_sink_token_pos_embs is not None:
                context_latents = torch.cat([self.global_sink_token_latents, context_latents], dim=2)
                context_pos_embs = torch.cat([self.global_sink_token_pos_embs, context_pos_embs], dim=0)
                
                # Re-align after adding global sink token (in case T_sink mismatches)
                t_lat_after = context_latents.shape[2]
                t_pos_after = context_pos_embs.shape[0]
                if t_pos_after != t_lat_after:
                    logging.warning(
                        f"[ContextPool] Mismatch after global sink token: "
                        f"context_latents T={t_lat_after}, context_pos_embs T={t_pos_after}, "
                        f"aligning to {t_lat_after}"
                    )
                    if t_pos_after > t_lat_after:
                        context_pos_embs = context_pos_embs[:t_lat_after]
                    else:
                        pad = context_pos_embs[-1:].expand(t_lat_after - t_pos_after, -1, -1, -1)
                        context_pos_embs = torch.cat([context_pos_embs, pad], dim=0)
        
        # Build padded current face pos embs by reusing data from all_faces_pos_embs
        current_face_pos_embs = self._compute_padded_current_face_pos_embs(
            current_face, current_start, current_end, all_faces_pos_embs, min_time, max_time, actual_resolution=actual_resolution
        )

        # Encode current face conditional latents for fuse_vae_embedding_in_latents
        current_face_latents = None
        current_face_mask = None
        if current_face in cond_cubemap and current_face in mask_cubemap:
            current_face_cond_content = cond_cubemap[current_face][current_start:current_end]  # [T, C, H, W]
            
            current_face_cond_bcthw = current_face_cond_content.permute(1, 0, 2, 3).unsqueeze(0)  # [B, C, T, H, W]
            
            current_face_latents = vae.encode(
                current_face_cond_bcthw.to(device=device, dtype=torch_dtype),
                tiled=self.use_tiled_vae,
                device=device
            ).to(torch_dtype)  # (B, C, T', H', W')
        # Save context if requested
        if save_context and save_dir:
            _save_context_video(context_bcthw.detach().cpu(), save_dir, save_prefix, step, rank, current_face, order_idx)
        
        # Get fragment future context if enabled
        # Note: When always_full_context=True, we don't use fragment_future_context
        # because all future content is already included in the main context
        fragment_future_context = None
        if self.fragment_future_context and not self.always_full_context:
            fragment_future_context = self.get_fragment_future_context(
                current_face, current_end, cond_cubemap, vae, device, torch_dtype,
                save_fragments=save_context,  # Use same flag as context saving
                save_dir=save_dir,
                save_prefix="fragment",
                step=step,
                rank=rank,
                order_idx=order_idx,
                actual_resolution=actual_resolution
            )
        elif self.always_full_context:
            logging.info(f"[always_full_context=True] Skipping fragment_future_context since all future content is already included in the main context")
        
        if context_latents is not None and context_pos_embs is not None:
            t_lat_final = context_latents.shape[2]
            t_pos_final = context_pos_embs.shape[0]
            if t_pos_final != t_lat_final:
                logging.warning(
                    f"[ContextPool] Final mismatch detected before return: "
                    f"context_latents T={t_lat_final}, context_pos_embs T={t_pos_final}, "
                    f"aligning to {t_lat_final} (min_time={min_time}, max_time={max_time})"
                )
                if t_pos_final > t_lat_final:
                    context_pos_embs = context_pos_embs[:t_lat_final]
                else:
                    pad = context_pos_embs[-1:].expand(t_lat_final - t_pos_final, -1, -1, -1)
                    context_pos_embs = torch.cat([context_pos_embs, pad], dim=0)
        
        return context_latents, context_pos_embs, \
                current_face_pos_embs, current_face_latents, \
                current_face_mask, fragment_future_context
    
    
    def get_top_left_latent_h_w_index_start(self, face: str, cube_map_latent_size_with_padding: int):
        """Get top left latent start index for a given face and padding width."""
        if len(self.active_faces) == 6:
            if self.use_vanilla_pos_embs:
                if face == 'F':
                    return 0, 0
                elif face == 'R':
                    return 0, cube_map_latent_size_with_padding
                elif face == 'B':
                    return 0, 2 * cube_map_latent_size_with_padding
                elif face == 'L':
                    return 0, 3 * cube_map_latent_size_with_padding
                elif face == 'U':
                    return 0, 4 * cube_map_latent_size_with_padding
                elif face == 'D':
                    return 0, 5 * cube_map_latent_size_with_padding
                else:
                    raise ValueError(f"Face {face} not in active faces: {self.active_faces}")
            else:
                if face == 'F':
                    return cube_map_latent_size_with_padding, 0
                elif face == 'R':
                    return cube_map_latent_size_with_padding, cube_map_latent_size_with_padding
                elif face == 'B':
                    return cube_map_latent_size_with_padding, 2 * cube_map_latent_size_with_padding
                elif face == 'L':
                    return cube_map_latent_size_with_padding, 3 * cube_map_latent_size_with_padding
                elif face == 'U':
                    return 0, 0
                elif face == 'D':
                    return 2 * cube_map_latent_size_with_padding, 0
                else:
                    raise ValueError(f"Face {face} not in active faces: {self.active_faces}")
        elif len(self.active_faces) == 4:
            if face == 'F':
                return 0, 0
            elif face == 'R':
                return 0, cube_map_latent_size_with_padding
            elif face == 'B':
                return 0, 2 * cube_map_latent_size_with_padding
            elif face == 'L':
                return 0, 3 * cube_map_latent_size_with_padding
            else:
                raise ValueError(f"Face {face} not in active faces: {self.active_faces}")
        else:
            raise ValueError(f"Active faces length {len(self.active_faces)} not supported")
    
    def compute_absolute_latent_pos_embs(self, current_face: str,
                                         current_start: int, current_end: int,
                                         with_padding: bool = False,
                                         actual_resolution: int = None):
        if with_padding:
            import warnings
            warnings.warn(
                "Using with_padding=True in compute_absolute_latent_pos_embs is deprecated. "
                "Use the new _compute_padded_current_face_pos_embs method instead, which applies "
                "proper cube map padding logic matching CubeMapPadder.",
                DeprecationWarning,
                stacklevel=2
            )
        current_t_interval, current_h_interval, current_w_interval = self.compute_absolute_latent_positions(
            current_face, current_start, current_end, with_padding=with_padding, actual_resolution=actual_resolution)
        _t = current_t_interval[1] - current_t_interval[0]
        _h = current_h_interval[1] - current_h_interval[0]
        _w = current_w_interval[1] - current_w_interval[0]
        current_freqs = torch.cat([
            self.dit_precompted_freqs[0][current_t_interval[0]:current_t_interval[1]].view(_t, 1, 1, -1).expand(_t, _h, _w, -1),
            self.dit_precompted_freqs[1][current_h_interval[0]:current_h_interval[1]].view(1, _h, 1, -1).expand(_t, _h, _w, -1),
            self.dit_precompted_freqs[2][current_w_interval[0]:current_w_interval[1]].view(1, 1, _w, -1).expand(_t, _h, _w, -1)
        ], dim=-1)  # (T', H', W', dim)
        return current_freqs
    
    def compute_absolute_latent_positions(
        self,
        current_face: str,
        current_start: int,
        current_end: int,
        with_padding: bool = False,
        actual_resolution: int = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute absolute positional indices for proper embedding assignment in latent space.
        
        Args:
            current_face: Current face being generated
            current_start: Start frame of current generation window (absolute)
            current_end: End frame of current generation window (absolute)
            with_padding: Whether to include padding in position calculation
            actual_resolution: Actual resolution of the content (for multi-resolution training)
            
        Returns:
            Tuple of (t_interval, h_interval, w_interval) for indexing into positional embeddings
        """
        # Use actual resolution if provided, otherwise fall back to self.cube_map_size
        if actual_resolution is None:
            actual_resolution = self.cube_map_size
        
        # Current generation window positions (these are the target positions)
        # Convert frame indices to latent indices
        current_t_interval = [current_start//4, (current_end-1)//4+1]
       
        cube_map_latent_size_with_padding = (actual_resolution + 2 * self.padding_width) // self.spatial_compression_factor

        latent_h_start, latent_w_start = self.get_top_left_latent_h_w_index_start(current_face, cube_map_latent_size_with_padding)

        # Convert spatial coordinates to latent space
        latent_h_size = cube_map_latent_size_with_padding
        latent_w_size = cube_map_latent_size_with_padding

        if with_padding:
            current_h_interval = [latent_h_start, latent_h_start + latent_h_size]
            current_w_interval = [latent_w_start, latent_w_start + latent_w_size]
        else:
            padding_width_latent = self.padding_width // self.spatial_compression_factor
            current_h_interval = [latent_h_start + padding_width_latent, latent_h_start + latent_h_size - padding_width_latent]
            current_w_interval = [latent_w_start + padding_width_latent, latent_w_start + latent_w_size - padding_width_latent]
        
        return current_t_interval, current_h_interval, current_w_interval
