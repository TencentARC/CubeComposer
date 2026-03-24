import torch
from typing import List, Dict, Tuple, Optional
import logging
from .context_pool import PositionalEmbeddingPadder, _save_context_video


logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(funcName)s - %(levelname)s - %(message)s")


class ContextLatentPool:
    """
    Manages context latent pool for efficient autoregressive cube map video generation.
    
    Handles context latents for efficient autoregressive cube map video generation.
    """
    
    def __init__(self, cube_map_size: int = 512, padding_width: int = 32, max_history_windows: int = 2,
                 dit_patch_size: List[int] = [1, 2, 2], dit_precompted_freqs: torch.Tensor = None,
                 vae_spatial_compression: int = 16, vae_temporal_compression: int = 4, active_faces: List[str] = None,
                 use_vanilla_pos_embs: bool = False, window_length: int = 9,
                 fragment_future_context: bool = False,
                 coverage_threshold: float = 0.1,
                 always_full_context: bool = False,
                 use_global_sink_token: bool = False,
                 use_tiled_vae: bool = False):
        self.cubemap_latent_size = cube_map_size
        self.padding_width = padding_width
        self.max_history_windows = max_history_windows
        self.always_full_context = always_full_context
        self.use_global_sink_token = use_global_sink_token
        self.use_tiled_vae = use_tiled_vae
        self.vae_spatial_compression = vae_spatial_compression
        self.vae_temporal_compression = vae_temporal_compression
        
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
        
        # History storage: {face: List[(start_frame, end_frame, tensor)]} (latent tensors)
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

    def detect_fragments(self, mask_cubemap: Dict[str, torch.Tensor]):
        """
        Detect continuous fragments across all cube faces based on coverage threshold.
        
        Args:
            mask_cubemap: Dict mapping face names to mask tensors [T, C, H, W] where 1=content, 0=empty
        """
        if not self.fragment_future_context:
            logging.warning(f"fragment_future_context is disabled. Detecting fragments is disabled.")
            return
        
        if not mask_cubemap:
            logging.warning(f"mask_cubemap is empty!")
            return
            
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
            assert T % 4 == 1, f"num_frame does not satisfy 4N+1 constraint, T: {T}"
            # Debug: check mask value range
            # mask_min = mask.min().item()
            # mask_max = mask.max().item()
            # mask_mean = mask.mean().item()
            # logging.info(f"Face {face}: shape={mask.shape}, min={mask_min:.4f}, max={mask_max:.4f}, mean={mask_mean:.4f}")
            
            # Calculate per-latent coverage
            # Handle both binary (0/1) and continuous (0-1) masks
            per_latent_coverage = []
            coverage_first_frame = mask[0].mean().item()
            per_latent_coverage.append(coverage_first_frame)
            
            for t in range(1, T, self.vae_temporal_compression):
                # VAE temporal downsampling factor is usually 4, but can be configured
                frame_mask = mask[t:t+self.vae_temporal_compression]  # [vae_temporal_compression, C, H, W]
                # Use mean of mask values as coverage (works for both binary and continuous masks)
                coverage = frame_mask.mean().item()
                per_latent_coverage.append(coverage)
            # logging.info(f"Face {face} per-latent coverage: {[f'{c:.3f}' for c in per_latent_coverage]}")
            # Detect continuous fragments where coverage stays above threshold
            fragments = []
            start_frame = None
            
            for t, coverage in enumerate(per_latent_coverage):
                if coverage >= self.coverage_threshold:
                    if start_frame is None:
                        start_frame = t
                        # logging.info(f"Face {face}: Fragment starts at frame {t}, coverage={coverage:.3f}")
                else:
                    if start_frame is not None:
                        # End of a fragment
                        end_frame = t
                        avg_coverage = sum(per_latent_coverage[start_frame:end_frame]) / (end_frame - start_frame)
                        fragments.append((start_frame, end_frame, avg_coverage))
                        # logging.info(f"Face {face}: Fragment detected [{start_frame}, {end_frame}), avg_coverage={avg_coverage:.3f}")
                        start_frame = None
            
            # Handle case where last fragment extends to end of video
            if start_frame is not None:
                end_frame = len(per_latent_coverage)
                avg_coverage = sum(per_latent_coverage[start_frame:end_frame]) / (end_frame - start_frame)
                fragments.append((start_frame, end_frame, avg_coverage))
                # logging.info(f"Face {face}: Final fragment detected [{start_frame}, {end_frame}), avg_coverage={avg_coverage:.3f}")
            
            self.fragment_cache[face] = fragments
            # logging.info(f"Face {face}: Total {len(fragments)} fragments detected")

    def set_generation_order(self, order: List[Tuple[str, int, int]], order_idx: int = 0):
        """Set the generation order for current inference/training step."""
        self.current_order = order
        self.current_order_idx = order_idx
    
    def advance_order(self):
        """Advance to next item in generation order."""
        self.current_order_idx += 1

    def add_to_history(self, face: str, start_frame: int, end_frame: int, content: torch.Tensor):
        """
        Add generated content to history.

        Args:
            face: Face name ('F', 'R', 'B', 'L', 'U', 'D')
            start_frame: Starting frame index (absolute in full video)
            end_frame: Ending frame index (absolute in full video)
            content: Generated content latent tensor [T, C', H', W'] (without padding)
        """
        self.history[face].append((start_frame, end_frame, content.clone()))

    def set_context_from_generated(self, face: str, start_frame: int, end_frame: int,
                                   latent_tensor: torch.Tensor):
        """
        Inject a student-generated latent face into history, replacing any
        overlapping entry.  Used during distillation rollout training so that
        subsequent faces receive the student's own output as context rather than
        ground-truth, thereby correcting exposure bias.

        In latent mode the history stores latent-space tensors [T, C, H, W].
        build_context_latents() will use them directly (no VAE re-encode needed).

        Args:
            face:          Face name ('F', 'R', 'B', 'L', 'U', 'D')
            start_frame:   Starting latent-frame index (absolute, inclusive)
            end_frame:     Ending   latent-frame index (absolute, exclusive)
            latent_tensor: Student-generated latent [T, C, H, W] in latent space,
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
            (start_frame, end_frame, latent_tensor.detach().cpu())
        )
        logging.debug(
            f"[set_context_from_generated] Injected generated latent face '{face}' "
            f"[{start_frame}, {end_frame}) shape={tuple(latent_tensor.shape)} "
            f"into context history."
        )

    def build_context_latents(
        self,
        current_face: str,
        current_latent_start: int, 
        current_latent_end: int,
        cond_latent_cubemap: Dict[str, torch.Tensor],
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
        if cond_latent_cubemap:
            sample_tensor = next(iter(cond_latent_cubemap.values()))
        
        if sample_tensor is None:
            return None, None, None, None, None, None
            
        # latent sample_tensor: [T, C, H, W]
        # logging.info(f"sample_tensor shape: {sample_tensor.shape}")
        C_latent = sample_tensor.shape[1]
        # Use actual resolution from input data (for multi-resolution training support)
        actual_latent_resolution = sample_tensor.shape[2]  # H dimension
        H_latent = actual_latent_resolution
        W_latent = actual_latent_resolution * len(self.active_faces)
        cubemap_latent_size = actual_latent_resolution

        # Calculate context time range with length limit
        if self.always_full_context:
            # Always full context mode: context spans entire video length
            # No limitation from max_history_windows
            # All future perspective video content is included
            min_latent_time = 0
            max_latent_time = sample_tensor.shape[0]  # Total latent video length
            T_total_context_latent = max_latent_time - min_latent_time
        else:
            # Standard mode: limited context window
            # Context includes: current window + history windows (up to max_history_windows)
            # Future content beyond current_end is handled by fragment_future_context
            current_latent_window_length = current_latent_end - current_latent_start

            max_context_latent_length = self.max_history_windows * current_latent_window_length
            # Context spans from min_time to current_end (not beyond)
            # Go back at most max_context_length frames from current_end
            min_latent_time = max(0, current_latent_end - max_context_latent_length)
            max_latent_time = current_latent_end
            T_total_context_latent = max_latent_time - min_latent_time
        
        # Initialize context video
        context_latent_video = torch.zeros((T_total_context_latent, C_latent, H_latent, W_latent), device=device, dtype=torch_dtype)
        # logging.info(f"context_latent_video shape: {context_latent_video.shape}, time range: [{min_latent_time}, {max_latent_time})")
        
        if self.always_full_context:
            # Always full context mode: match historical full context implementation
            # 1. Fill all historical content from self.history
            for face in self.active_faces:
                if face in self.history:
                    slot_idx = slot_map[face]
                    w_start = slot_idx * H_latent
                    w_end = w_start + H_latent
                    
                    for start_frame, end_frame, content in self.history[face]:
                        t_start_rel = start_frame - min_latent_time
                        t_end_rel = end_frame - min_latent_time
                        # Only place if within our context video bounds
                        if t_start_rel >= 0 and t_end_rel <= T_total_context_latent:
                            context_latent_video[t_start_rel:t_end_rel, :, :, w_start:w_end] = content.to(device=device, dtype=torch_dtype)
            
            # 2. Fill with cond_cubemap for all faces (perspective video)
            # For each face: check if current window is already filled by history
            for _face in self.active_faces:
                if _face not in cond_latent_cubemap:
                    continue
                    
                slot_idx = slot_map[_face]
                w_start = slot_idx * H_latent
                w_end = w_start + H_latent
                t_start_rel = current_latent_start - min_latent_time
                t_end_rel = current_latent_end - min_latent_time
                
                # Check if current window [current_start, current_end) is already filled by history
                if context_latent_video[t_start_rel:t_end_rel, :, :, w_start:w_end].sum() == 0:
                    # Not filled by history, fill from current_start onwards
                    cond_content = cond_latent_cubemap[_face][current_latent_start:]
                    context_latent_video[t_start_rel:, :, :, w_start:w_end] = cond_content.to(device=device, dtype=torch_dtype)
                else:
                    # Already filled by history, only fill future windows from current_end onwards
                    cond_content = cond_latent_cubemap[_face][current_latent_end:]
                    context_latent_video[t_end_rel:, :, :, w_start:w_end] = cond_content.to(device=device, dtype=torch_dtype)
        else:
            # Standard mode: fill history first, then fill gaps with cond_cubemap
            # 1. Fill historical content from self.history (only within [min_time, max_time))
            for face in self.active_faces:
                if face in self.history:
                    slot_idx = slot_map[face]
                    w_start = slot_idx * H_latent
                    w_end = w_start + H_latent
                    
                    for start_frame, end_frame, content in self.history[face]:
                        # Only include history that overlaps with [min_time, max_time)
                        if end_frame <= min_latent_time or start_frame >= max_latent_time:
                            continue  # Outside context window, skip
                        
                        # Clip to context window bounds
                        clipped_start = max(start_frame, min_latent_time)
                        clipped_end = min(end_frame, max_latent_time)
                        
                        # Calculate relative indices in context video
                        t_start_rel = clipped_start - min_latent_time
                        t_end_rel = clipped_end - min_latent_time
                        
                        # Calculate corresponding slice in content tensor
                        content_start = clipped_start - start_frame
                        content_end = content_start + (clipped_end - clipped_start)
                        
                        # logging.info(f"Adding history content for face {face} from {clipped_start} (rel{t_start_rel}) to {clipped_end} (rel{t_end_rel})")
                        # logging.info(f"content shape: {content.shape}, content_start: {content_start}, content_end: {content_end}")
                        context_latent_video[t_start_rel:t_end_rel, :, :, w_start:w_end] = content[content_start:content_end].to(device=device, dtype=torch_dtype)
            
            # 2. Fill with cond_cubemap for current generation window [current_start, current_end) where not already filled by history
            # This includes both the period before current window (if min_time < current_start) and the current window
            for _face in self.active_faces:
                if _face not in cond_latent_cubemap:
                    continue
                    
                slot_idx = slot_map[_face]
                w_start = slot_idx * H_latent
                w_end = w_start + H_latent
                
                # Extract the relevant slice from cond_cubemap for this context window
                t_start_rel = current_latent_start - min_latent_time
                t_end_rel = current_latent_end - min_latent_time
                # logging.info(f"Adding cond content for face {_face} from {current_latent_start} (rel{t_start_rel}) to {current_latent_end} (rel{t_end_rel})")
                # Only add if not already filled by history (only fill in current face and other ungenerated faces in current window)
                if context_latent_video[t_start_rel:t_end_rel, :, :, w_start:w_end].sum() == 0:
                    # Have not been filled by history, put cond content at current time window and future windows
                    cond_content = cond_latent_cubemap[_face][current_latent_start:current_latent_end]
                    context_latent_video[t_start_rel:t_end_rel, :, :, w_start:w_end] = cond_content.to(device=device, dtype=torch_dtype)
        
        context_latent_bcthw = context_latent_video.permute(1, 0, 2, 3).unsqueeze(0)   
        context_latents = context_latent_bcthw.to(torch_dtype)  # (B, C, T', H', W')
        
        # Build positional embeddings along the width dimension
        # Note: context_pos_embs must match the dimensions after patch embedding
        # patch_size is [1, 2, 2], so H and W need to be divided by 2
        context_pos_embs = []
        all_faces_pos_embs = {}  # Store individual face pos embs for reuse
        for _face in self.active_faces:
            current_freqs = self.compute_absolute_latent_pos_embs(_face, min_latent_time, max_latent_time, cubemap_latent_size=cubemap_latent_size)
            context_pos_embs.append(current_freqs)
            all_faces_pos_embs[_face] = current_freqs

        context_pos_embs = torch.cat(context_pos_embs, dim=2)  # (T', H'', 6*W'', dim)
        
        # Build padded current face pos embs by reusing data from all_faces_pos_embs
        current_latent_start_rel = current_latent_start - min_latent_time
        current_latent_end_rel = current_latent_end - min_latent_time
        current_face_pos_embs = self._compute_padded_current_face_pos_embs(
            current_face, current_latent_start_rel, current_latent_end_rel, all_faces_pos_embs, cubemap_latent_size=cubemap_latent_size
        )

        # Encode current face conditional latents for fuse_vae_embedding_in_latents
        current_face_latents = None
        current_face_mask = None
        if current_face in cond_latent_cubemap:
            current_face_latents = cond_latent_cubemap[current_face][current_latent_start:current_latent_end]  # [T, C, H, W]
            # Convert to [B, C, T, H, W] format expected by training_loss
            current_face_latents = current_face_latents.permute(1, 0, 2, 3).unsqueeze(0)  # [B, C, T, H, W]
        
        # Get fragment future context if enabled
        # Note: When always_full_context=True, we don't use fragment_future_context
        # because all future content is already included in the main context
        fragment_future_context = None
        if self.fragment_future_context and not self.always_full_context:
            fragment_future_context = self.get_fragment_future_context(
                current_face, current_latent_end, cond_latent_cubemap,
                save_fragments=save_context,  # Use same flag as context saving
                save_dir=save_dir,
                save_prefix=save_prefix + "_fragment",
                step=step,
                rank=rank,
                order_idx=order_idx,
                cubemap_latent_size=cubemap_latent_size
            )
        elif self.always_full_context:
            logging.info(f"[always_full_context=True] Skipping fragment_future_context since all future content is already included in the main context")
        
        return context_latents, context_pos_embs, \
                current_face_pos_embs, current_face_latents, \
                current_face_mask, fragment_future_context

    def _compute_padded_current_face_pos_embs(
        self, 
        current_face: str, 
        current_latent_start_rel: int, 
        current_latent_end_rel: int, 
        all_faces_pos_embs: Dict[str, torch.Tensor],
        cubemap_latent_size: int = None
    ) -> torch.Tensor:
        # Get cube map size in latent space (without padding)
        # Note: all_faces_pos_embs is already patched (H and W divided by patch_size[1] and patch_size[2])
        if cubemap_latent_size is None:
            cubemap_latent_size = self.cubemap_latent_size
        original_height_latent = cubemap_latent_size
        original_width_latent = cubemap_latent_size
        
        # Extract time slices for all faces  
        current_window_faces_pos_embs = {}
        for face_name, face_pos_embs in all_faces_pos_embs.items():
            # Extract current time window: [T_current, H, W, dim]
            current_window_faces_pos_embs[face_name] = face_pos_embs[current_latent_start_rel:current_latent_end_rel, :, :, :]
        
        # Apply padding using the padder
        padded_current_face_pos_embs = self.padder_pos_emb.pad_face_pos_embs(
            current_window_faces_pos_embs, 
            current_face, 
            original_height_latent,
            original_width_latent
        )
        
        return padded_current_face_pos_embs

    def compute_absolute_latent_pos_embs(self, current_face: str,
                                         current_latent_start: int, current_latent_end: int,
                                         cubemap_latent_size: int = None):
        current_t_interval, current_h_interval, current_w_interval = self.compute_absolute_latent_positions(
            current_face, current_latent_start, current_latent_end, cubemap_latent_size=cubemap_latent_size)
        _t = current_t_interval[1] - current_t_interval[0]
        _h = current_h_interval[1] - current_h_interval[0]
        _w = current_w_interval[1] - current_w_interval[0]
        current_freqs = torch.cat([
            self.dit_precompted_freqs[0][current_t_interval[0]:current_t_interval[1]].view(_t, 1, 1, -1).expand(_t, _h, _w, -1),
            self.dit_precompted_freqs[1][current_h_interval[0]:current_h_interval[1]].view(1, _h, 1, -1).expand(_t, _h, _w, -1),
            self.dit_precompted_freqs[2][current_w_interval[0]:current_w_interval[1]].view(1, 1, _w, -1).expand(_t, _h, _w, -1)
        ], dim=-1)  # (T', H', W', dim)
        return current_freqs
    
    def get_top_left_latent_h_w_index_start(self, face: str, cube_map_latent_size: int):
        """Get top left latent start index for a given face and padding width."""
        # For horizontal layout, all faces are placed horizontally side by side
        # Use index in active_faces list to determine horizontal position
        if len(self.active_faces) == 6:
            if self.use_vanilla_pos_embs:
                if face == 'F':
                    return 0, 0
                elif face == 'R':
                    return 0, cube_map_latent_size
                elif face == 'B':
                    return 0, 2 * cube_map_latent_size
                elif face == 'L':
                    return 0, 3 * cube_map_latent_size
                elif face == 'U':
                    return 0, 4 * cube_map_latent_size
                elif face == 'D':
                    return 0, 5 * cube_map_latent_size
                else:
                    raise ValueError(f"Face {face} not in active faces: {self.active_faces}")
            else:
                if face == 'F':
                    return cube_map_latent_size, 0
                elif face == 'R':
                    return cube_map_latent_size, cube_map_latent_size
                elif face == 'B':
                    return cube_map_latent_size, 2 * cube_map_latent_size
                elif face == 'L':
                    return cube_map_latent_size, 3 * cube_map_latent_size
                elif face == 'U':
                    return 0, 0
                elif face == 'D':
                    return 2 * cube_map_latent_size, 0
                else:
                    raise ValueError(f"Face {face} not in active faces: {self.active_faces}")
        elif len(self.active_faces) == 4:
            if face == 'F':
                return 0, 0
            elif face == 'R':
                return 0, cube_map_latent_size
            elif face == 'B':
                return 0, 2 * cube_map_latent_size
            elif face == 'L':
                return 0, 3 * cube_map_latent_size
            else:
                raise ValueError(f"Face {face} not in active faces: {self.active_faces}")
        else:
            raise ValueError(f"Active faces length {len(self.active_faces)} not supported")

    def compute_absolute_latent_positions(
        self,
        current_face: str,
        current_latent_start: int,
        current_latent_end: int,
        cubemap_latent_size: int = None
    ) -> Dict[str, torch.Tensor]:
        # Use actual resolution if provided, otherwise fall back to self.cube_map_size
        if cubemap_latent_size is None:
            cubemap_latent_size = self.cubemap_latent_size
        cubemap_latent_size = cubemap_latent_size // self.dit_patch_size[1]
        latent_h_start, latent_w_start = self.get_top_left_latent_h_w_index_start(current_face, cubemap_latent_size)
        current_h_interval = [latent_h_start, latent_h_start + cubemap_latent_size]
        current_w_interval = [latent_w_start, latent_w_start + cubemap_latent_size]
        return [current_latent_start, current_latent_end], current_h_interval, current_w_interval

    def get_fragment_future_context(
        self,
        current_face: str,
        current_latent_end: int,
        cond_latent_cubemap: Dict[str, torch.Tensor],
        save_fragments: bool = False,
        save_dir: Optional[str] = None,
        save_prefix: str = "fragment",
        step: Optional[int] = None,
        rank: Optional[int] = None,
        order_idx: Optional[int] = None,
        cubemap_latent_size: int = None
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
            step: Training/validation step number
            rank: Process rank (for multi-GPU)
            order_idx: Current order index
            
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
            if face_name not in cond_latent_cubemap:
                continue
            
            face_cond = cond_latent_cubemap[face_name]  # [T, C, H, W]
            # Debug: check if face_cond is in latent space (should have 48 channels) or pixel space (3 channels)
            if face_cond.shape[1] == 3:
                logging.warning(f"[get_fragment_future_context] face_cond for {face_name} has 3 channels (RGB), expected 48 (latent). "
                              f"This suggests cond_latent_cubemap contains pixel-space tensors, not latent-space tensors.")
            face_fragments = self.fragment_cache[face_name]
            
            # Find the nearest future fragment for this face
            nearest_fragment = None
            for start_frame, end_frame, avg_coverage in face_fragments:
                # Skip fragments that are completely in the past or current window
                if end_frame <= current_latent_end:
                    continue
                # This is a future fragment, record it
                nearest_fragment = (start_frame, end_frame, avg_coverage)
                break  # Take only the first (nearest) future fragment
            
            if nearest_fragment is None:
                continue  # No future fragment found for this face
            
            start_frame, end_frame, avg_coverage = nearest_fragment
            
            # logging.info(f"[get_fragment_future_context] face_fragments for {face_name}: [{start_frame}, {end_frame}] coverage: {avg_coverage}")
            # Calculate the future portion of this fragment
            frag_start = max(start_frame, current_latent_end)
            actual_length = end_frame - frag_start
            
            if actual_length == 0:
                continue

            # Set the fragment range
            frag_end = frag_start + actual_length
            
            # Extract fragment content
            fragment_content = face_cond[frag_start:frag_end]  # [T_frag, C, H, W]
            # logging.info(f"[get_fragment_future_context] fragment_content for {face_name}: {fragment_content.shape}")
            if fragment_content.shape[0] == 0:
                logging.warning(f"[get_fragment_future_context] fragment_content for {face_name} is empty")
                continue
            # Convert to [B, C, T, H, W] format expected by process_context_embs
            # Note: fragment_content is already in latent space (from cond_latent_cubemap)
            fragment_bcthw = fragment_content.permute(1, 0, 2, 3).unsqueeze(0)  # [B, C, T_frag, H, W]
            
            # Compute positional embeddings for this fragment (using face_name, not current_face)
            fragment_pos_embs = self.compute_absolute_latent_pos_embs(
                face_name, frag_start, frag_end, cubemap_latent_size=cubemap_latent_size
            )
            
            future_fragments.append((fragment_bcthw, fragment_pos_embs))
            
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
        
        # Summary info
        if save_fragments and save_dir:
            if len(future_fragments) > 0:
                logging.info(f"Total {len(future_fragments)} future fragments extracted "
                           f"(current face: {current_face}, processed faces: {faces_to_process}), "
                           f"current_end={current_latent_end}")
            else:
                logging.info(f"No future fragments found for current face {current_face} and adjacent faces, "
                           f"current_end={current_latent_end}")
        
        return future_fragments

