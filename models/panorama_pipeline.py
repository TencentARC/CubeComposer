from contextlib import nullcontext
import torch
from typing import Optional, Union, Tuple, Dict, List
import json
from safetensors.torch import load_file

from diffsynth.pipelines.wan_video_new import WanVideoPipeline, TemporalTiler_BCTHW
from diffsynth.utils import ModelConfig
from diffsynth.models.wan_video_dit import WanModel, modulate, rope_apply
from diffsynth.models.wan_video_motion_controller import WanMotionControllerModel
from diffsynth.models.wan_video_vace import VaceWanModel
from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
from models.context_pool import ContextPool
from models.context_latent_pool import ContextLatentPool
from einops import rearrange
from equilib import Equi2Cube
from .panorama import CubeMapPadder, GenerationOrderPlanner
from tqdm import tqdm
import os
import logging
from diffsynth.pipelines.wan_video_new import PipelineUnit, WanVideoUnit_NoiseInitializer

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(funcName)s - %(levelname)s - %(message)s")



class WanVideoUnit_NoiseInitializer_LatentMode(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames", "seed", "rand_device", "vace_reference_image"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device, vace_reference_image):
        length = num_frames
        if vace_reference_image is not None:
            length += 1
        shape = (1, pipe.vae.model.z_dim, length, height, width)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -1:], noise[:, :, :-1]), dim=2)
        return {"noise": noise}


def get_model_configs(model_paths=None, model_id_with_origin_paths=None, local_model_path=None):
    model_configs = []
    if model_paths is not None:
        model_paths = json.loads(model_paths)
        model_configs += [ModelConfig(
            path=path,
            local_model_path=local_model_path
        ) for path in model_paths]
    if model_id_with_origin_paths is not None:
        model_id_with_origin_paths = model_id_with_origin_paths.split(",")
        model_configs += [
            ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1],
            local_model_path=local_model_path) for i in model_id_with_origin_paths]
    for model_config in model_configs:
        model_config.local_model_path = local_model_path
        print(f"Settings model_config (model_id: {model_config.model_id}) local_model_path: {model_config.local_model_path}")
    return model_configs

def model_fn_panorama_video(**kwargs):
    # Forward to base model_fn
    return model_fn_wan_video(**kwargs)

##### Diagonal Context #####
def _diagonal_context_self_attn_clear_cache(self):
    """Clear cached K, V for context and fragment parts."""
    # Only store flat versions to save GPU memory
    # Spatial versions can be reconstructed via rearrange when needed
    self.cached_k_context = None
    self.cached_v_context = None
    self.cached_k_fragment = None
    self.cached_v_fragment = None
    self.cached_context_grid_size = None
    self.cached_fragment_grid_size = None

def _diagonal_context_self_attn_has_cache(self):
    """Check if cache exists."""
    return self.cached_k_context is not None

def _diagonal_context_self_attn_forward(self, x, freqs, x_grid_size, context_grid_size, last_fragment_grid_size=None, use_cache=False):
    """
    Args:
        x: torch.Tensor - input including origin, context, and optional fragment parts
        freqs: torch.Tensor - RoPE frequencies for all positions
        x_grid_size: tuple (f, h, w) for origin/denoising area
        context_grid_size: tuple (f, h, w) for context area
        last_fragment_grid_size: optional tuple (f, h, w) for fragment area
        use_cache: bool, whether to use KV cache for context and fragment parts
    
    Note: We always compute Q from the full x (including context) because x's context
    part changes across denoising steps due to residual connections and cross-attention.
    We only cache K and V for context/fragment since they depend on the fixed context_embs.
    """
    seq_len = x.shape[1]
    origin_len = x_grid_size[0] * x_grid_size[1] * x_grid_size[2]
    context_len = context_grid_size[0] * context_grid_size[1] * context_grid_size[2]
    fragment_len = seq_len - context_len - origin_len
    
    # Check if we can use cache
    has_cache = self.has_cache() if use_cache else False
    # Always compute full Q from current x (including context, which changes across steps)
    q = self.norm_q(self.q(x))
    q = rope_apply(q, freqs, self.num_heads)
    q_origin = q[:, :origin_len]
    
    if has_cache:
        # Fast path: compute origin K,V and use cached context/fragment K,V
        x_origin_input = x[:, :origin_len]
        k_origin = self.norm_k(self.k(x_origin_input))
        v_origin = self.v(x_origin_input)
        freqs_origin = freqs[:origin_len]
        k_origin = rope_apply(k_origin, freqs_origin, self.num_heads)
        
        # Concatenate with cached K, V (flat version) for full attention on origin
        k_full = torch.cat([k_origin, self.cached_k_context], dim=1)
        if self.cached_k_fragment is not None:
            k_full = torch.cat([k_full, self.cached_k_fragment], dim=1)
        
        v_full = torch.cat([v_origin, self.cached_v_context], dim=1)
        if self.cached_v_fragment is not None:
            v_full = torch.cat([v_full, self.cached_v_fragment], dim=1)
        
        x_origin = self.attn(q_origin, k_full, v_full)
        
        # Compute context attention with current Q but cached K, V
        # Rearrange cached flat K,V to spatial on-the-fly (cheap operation)
        q_context = q[:, origin_len:origin_len+context_len]
        q_context = rearrange(q_context, 'b (f h w) d -> (b f) (h w) d', 
                             f=self.cached_context_grid_size[0], 
                             h=self.cached_context_grid_size[1], 
                             w=self.cached_context_grid_size[2])
        k_context_spatial = rearrange(self.cached_k_context, 'b (f h w) d -> (b f) (h w) d',
                                     f=self.cached_context_grid_size[0],
                                     h=self.cached_context_grid_size[1],
                                     w=self.cached_context_grid_size[2])
        v_context_spatial = rearrange(self.cached_v_context, 'b (f h w) d -> (b f) (h w) d',
                                     f=self.cached_context_grid_size[0],
                                     h=self.cached_context_grid_size[1],
                                     w=self.cached_context_grid_size[2])
        x_context = self.attn(q_context, k_context_spatial, v_context_spatial)
        x_context = rearrange(x_context, '(b f) (h w) d -> b (f h w) d', 
                             f=self.cached_context_grid_size[0],
                             h=self.cached_context_grid_size[1], 
                             w=self.cached_context_grid_size[2])
        x = torch.cat([x_origin, x_context], dim=1)
        
        # Handle fragment if exists
        if self.cached_k_fragment is not None:
            q_fragment = q[:, -fragment_len:]
            fragment_batch_size = q_fragment.shape[0]
            q_fragment = rearrange(q_fragment, 'b (f h w) d -> (b f) (h w) d', 
                                  h=self.cached_fragment_grid_size[1], 
                                  w=self.cached_fragment_grid_size[2])
            k_fragment_spatial = rearrange(self.cached_k_fragment, 'b (f h w) d -> (b f) (h w) d',
                                          h=self.cached_fragment_grid_size[1],
                                          w=self.cached_fragment_grid_size[2])
            v_fragment_spatial = rearrange(self.cached_v_fragment, 'b (f h w) d -> (b f) (h w) d',
                                          h=self.cached_fragment_grid_size[1],
                                          w=self.cached_fragment_grid_size[2])
            x_fragment = self.attn(q_fragment, k_fragment_spatial, v_fragment_spatial)
            x_fragment = rearrange(x_fragment, '(b f) (h w) d -> b (f h w) d', 
                                  b=fragment_batch_size,
                                  h=self.cached_fragment_grid_size[1], 
                                  w=self.cached_fragment_grid_size[2])
            x = torch.cat([x, x_fragment], dim=1)
        
    else:

        # q is already computed above (always from full x)
        k = self.norm_k(self.k(x))
        v = self.v(x)
        k = rope_apply(k, freqs, self.num_heads)

        q_origin = q[:, :origin_len]
        x_origin = self.attn(q_origin, k, v) # [B, origin_len, num_heads*head_dim]

        q_context = q[:, origin_len:origin_len+context_len]
        k_context = k[:, origin_len:origin_len+context_len]
        v_context = v[:, origin_len:origin_len+context_len]
        q_context = rearrange(q_context, 'b (f h w) d -> (b f) (h w) d', f=context_grid_size[0], h=context_grid_size[1], w=context_grid_size[2])
        k_context = rearrange(k_context, 'b (f h w) d -> (b f) (h w) d', f=context_grid_size[0], h=context_grid_size[1], w=context_grid_size[2])
        v_context = rearrange(v_context, 'b (f h w) d -> (b f) (h w) d', f=context_grid_size[0], h=context_grid_size[1], w=context_grid_size[2])
        x_context = self.attn(q_context, k_context, v_context) # [B*(F_context), context_embs_seq_len, num_heads*head_dim]
        x_context = rearrange(x_context, '(b f) (h w) d -> b (f h w) d', f=context_grid_size[0], h=context_grid_size[1], w=context_grid_size[2])
        x = torch.cat([x_origin, x_context], dim=1)

        # Handle fragment if exists
        k_fragment_rearranged = None
        v_fragment_rearranged = None
        if last_fragment_grid_size is not None and fragment_len > 0:
            q_fragment = q[:, -fragment_len:]
            k_fragment = k[:, -fragment_len:]
            v_fragment = v[:, -fragment_len:]
            fragment_batch_size = q_fragment.shape[0]
            q_fragment = rearrange(q_fragment, 'b (f h w) d -> (b f) (h w) d', h=last_fragment_grid_size[1], w=last_fragment_grid_size[2])
            k_fragment_rearranged = rearrange(k_fragment, 'b (f h w) d -> (b f) (h w) d', h=last_fragment_grid_size[1], w=last_fragment_grid_size[2])
            v_fragment_rearranged = rearrange(v_fragment, 'b (f h w) d -> (b f) (h w) d', h=last_fragment_grid_size[1], w=last_fragment_grid_size[2])
            x_fragment = self.attn(q_fragment, k_fragment_rearranged, v_fragment_rearranged)
            x_fragment = rearrange(x_fragment, '(b f) (h w) d -> b (f h w) d', b=fragment_batch_size, h=last_fragment_grid_size[1], w=last_fragment_grid_size[2])
            x = torch.cat([x, x_fragment], dim=1)
        
        # Cache only flat K, V to save GPU memory (spatial can be reconstructed via rearrange)
        if use_cache:
            # Cache flat versions only
            self.cached_k_context = k[:, origin_len:origin_len+context_len]
            self.cached_v_context = v[:, origin_len:origin_len+context_len]
            self.cached_context_grid_size = context_grid_size
            
            if k_fragment_rearranged is not None:
                self.cached_k_fragment = k[:, -fragment_len:]
                self.cached_v_fragment = v[:, -fragment_len:]
                self.cached_fragment_grid_size = last_fragment_grid_size
            else:
                self.cached_k_fragment = None
                self.cached_v_fragment = None
                self.cached_fragment_grid_size = None
        
    return self.o(x)

def _modulate_generation_part_only(x, shift_msa, scale_msa, seq_generation_len):
    x_generation = x[:, :seq_generation_len]
    x_context = x[:, seq_generation_len:]
    return torch.concat([modulate(x_generation, shift_msa, scale_msa), x_context], dim=1)

def _diagonal_context_block_forward_separated_timestep_modulation(self, x, context, t_mod, freqs, x_grid_size, context_grid_size, last_fragment_grid_size=None, use_cache=False):
    has_seq = len(t_mod.shape) == 4
    chunk_dim = 2 if has_seq else 1
    # msa: multi-head self-attention  mlp: multi-layer perceptron
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
    if has_seq:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
            shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
        )
    seq_generation_len = x_grid_size[0] * x_grid_size[1] * x_grid_size[2]
    input_x = _modulate_generation_part_only(self.norm1(x), shift_msa, scale_msa, seq_generation_len)
    x = self.gate(x, gate_msa, self.self_attn(input_x, freqs, x_grid_size, context_grid_size, last_fragment_grid_size, use_cache))
    x = x + self.cross_attn(self.norm3(x), context)
    input_x = _modulate_generation_part_only(self.norm2(x), shift_mlp, scale_mlp, seq_generation_len)
    x = self.gate(x, gate_mlp, self.ffn(input_x))
    return x

def _diagonal_context_block_forward(self, x, context, t_mod, freqs, x_grid_size, context_grid_size, last_fragment_grid_size=None, use_cache=False):
    has_seq = len(t_mod.shape) == 4
    chunk_dim = 2 if has_seq else 1
    # msa: multi-head self-attention  mlp: multi-layer perceptron
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
    if has_seq:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
            shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
        )
    input_x = modulate(self.norm1(x), shift_msa, scale_msa)
    x = self.gate(x, gate_msa, self.self_attn(input_x, freqs, x_grid_size, context_grid_size, last_fragment_grid_size, use_cache))
    x = x + self.cross_attn(self.norm3(x), context)
    input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
    x = self.gate(x, gate_mlp, self.ffn(input_x))
    return x


class PanoramaWanPipeline(WanVideoPipeline):
    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, padding_width=16, window_length=8, cube_map_size=512, active_faces=None,
                use_vanilla_pos_embs=False, condition_mode=None, max_history_windows=2, fragment_future_context=None, inference_boundary_padding=True,
                inference_boundary_pixel_blending=False, use_diagonal_kv_cache=False, always_full_context=False, seperated_timestep_modulation=False,
                use_global_sink_token=False, disable_order_planning=False, use_tiled_vae=False, use_latent_mode=False,
                face_order_mode=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.condition_mode = condition_mode
        self.use_diagonal_kv_cache = use_diagonal_kv_cache  # Control whether to use KV cache for diagonal context inference

        if self.use_diagonal_kv_cache:
            logging.warning("Enabling KV cache is still experimental, which may cause unexpected issues. Use with caution.")

        self.inference_boundary_padding = inference_boundary_padding  # Control whether to use boundary padding for noise initialization
        self.inference_boundary_pixel_blending = inference_boundary_pixel_blending  # Control whether to blend padding pixels into adjacent faces
        # Set active faces
        if active_faces is None:
            self.active_faces = ['F', 'R', 'B', 'L', 'U', 'D']
        else:
            self.active_faces = active_faces
        
        self.padder = CubeMapPadder(padding_width=padding_width, is_latent_mode=use_latent_mode)
        # Create pixel-space padder for blending in latent mode (padding_width is in pixel space)
        if use_latent_mode:
            self.padder_pixel = CubeMapPadder(padding_width=padding_width, is_latent_mode=False)
        else:
            self.padder_pixel = self.padder  # In pixel mode, use the same padder
        self.planner = GenerationOrderPlanner(window_length=window_length, cube_map_size=cube_map_size, 
                                              active_faces=self.active_faces, disable_order_planning=disable_order_planning,
                                              is_latent_mode=use_latent_mode, face_order_mode=face_order_mode)
        self.equi2cube = Equi2Cube(w_face=cube_map_size, cube_format='dict')
        if use_latent_mode:
            self.context_pool = ContextLatentPool(cube_map_size=cube_map_size, padding_width=padding_width, active_faces=self.active_faces, 
            use_vanilla_pos_embs=use_vanilla_pos_embs, max_history_windows=max_history_windows, fragment_future_context=fragment_future_context,
            always_full_context=always_full_context, use_global_sink_token=use_global_sink_token, use_tiled_vae=use_tiled_vae)
            noise_initializer_idx = None
            for idx, unit in enumerate(self.units):
                if isinstance(unit, WanVideoUnit_NoiseInitializer):
                    noise_initializer_idx = idx
            assert noise_initializer_idx is not None, "WanVideoUnit_NoiseInitializer not found in units"
            self.units.remove(self.units[noise_initializer_idx])
            self.units.insert(noise_initializer_idx, WanVideoUnit_NoiseInitializer_LatentMode())
        else:
            self.context_pool = ContextPool(cube_map_size=cube_map_size, padding_width=padding_width, active_faces=self.active_faces, 
            use_vanilla_pos_embs=use_vanilla_pos_embs, max_history_windows=max_history_windows, fragment_future_context=fragment_future_context,
            always_full_context=always_full_context, use_global_sink_token=use_global_sink_token, use_tiled_vae=use_tiled_vae)
        self.fragment_future_context = fragment_future_context
        self.always_full_context = always_full_context
        self.seperated_timestep_modulation = seperated_timestep_modulation
        
        self.use_latent_mode = use_latent_mode
        self.model_fn = model_fn_panorama_video
        self.fuse_vae_embedding_in_latents = False


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"),
        redirect_common_files: bool = True, use_usp=False,
        padding_width=16, window_length=8, cube_map_size=512, 
        active_faces=None, fuse_vae_embedding_in_latents=False, use_vanilla_pos_embs=False, dit_checkpoint_path=None,
        condition_mode=None, max_history_windows=2, fragment_future_context=None, inference_boundary_padding=True,
        inference_boundary_pixel_blending=False, use_diagonal_kv_cache=False, always_full_context=False, seperated_timestep_modulation=False,
        use_global_sink_token=False,
        disable_order_planning=False, use_tiled_vae=False, use_latent_mode=False,
        face_order_mode=None
    ):
        logging.info(f"use_diagonal_kv_cache={use_diagonal_kv_cache}; always_full_context={always_full_context}; use_global_sink_token={use_global_sink_token}.")
        # Propagate local_model_path from model_configs to tokenizer_config
        # so the tokenizer is loaded from the same local cache instead of
        # falling back to downloading into ./models.
        if tokenizer_config.local_model_path is None and model_configs:
            for mc in model_configs:
                if mc.local_model_path is not None:
                    tokenizer_config.local_model_path = mc.local_model_path
                    break

        # Load using base pipeline API
        base_pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch_dtype, device=device,
            model_configs=model_configs, tokenizer_config=tokenizer_config,
            redirect_common_files=redirect_common_files, use_usp=use_usp
        )
        # Wrap into our subclass and copy models
        pipe = PanoramaWanPipeline(device=device, torch_dtype=torch_dtype, padding_width=padding_width, 
                                    window_length=window_length, cube_map_size=cube_map_size, active_faces=active_faces, 
                                    use_vanilla_pos_embs=use_vanilla_pos_embs, condition_mode=condition_mode, max_history_windows=max_history_windows,
                                    fragment_future_context=fragment_future_context, inference_boundary_padding=inference_boundary_padding,
                                    inference_boundary_pixel_blending=inference_boundary_pixel_blending, use_diagonal_kv_cache=use_diagonal_kv_cache,
                                    always_full_context=always_full_context, use_global_sink_token=use_global_sink_token,
                                    disable_order_planning=disable_order_planning, use_tiled_vae=use_tiled_vae, use_latent_mode=use_latent_mode,
                                    face_order_mode=face_order_mode)
        for name in ["text_encoder", "dit", "dit2", "vae", "image_encoder", "motion_controller", "vace", "prompter"]:
            setattr(pipe, name, getattr(base_pipe, name))
        pipe.height_division_factor = base_pipe.height_division_factor
        pipe.width_division_factor = base_pipe.width_division_factor
        pipe.context_pool.dit_patch_size = pipe.dit.patch_size
        pipe.context_pool.dit_precompted_freqs = pipe.dit.freqs
        
        # Context patch embedding for latent
        # Extra context patch embedding for latent and mask
        pipe.dit.context_latent_patch_embedding = torch.nn.Conv3d(
            pipe.dit.in_dim, pipe.dit.dim, 
            kernel_size=pipe.dit.patch_size, stride=pipe.dit.patch_size)
        # Initialize from base model
        pipe.dit.context_latent_patch_embedding.weight.data = pipe.dit.patch_embedding.weight.data.clone()  
        pipe.dit.context_latent_patch_embedding.bias.data = pipe.dit.patch_embedding.bias.data.clone()

        # Fragment patch embedding for latent
        pipe.dit.fragment_latent_patch_embedding = torch.nn.Conv3d(
            pipe.dit.in_dim, pipe.dit.dim, 
            kernel_size=pipe.dit.patch_size, stride=pipe.dit.patch_size)
        # Initialize from base model
        pipe.dit.fragment_latent_patch_embedding.weight.data = pipe.dit.patch_embedding.weight.data.clone()  
        pipe.dit.fragment_latent_patch_embedding.bias.data = pipe.dit.patch_embedding.bias.data.clone()

        pipe.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        if fuse_vae_embedding_in_latents:
            pipe.dit.context_mask_patch_embedding = torch.nn.Conv3d(
                pipe.dit.in_dim, pipe.dit.dim, kernel_size=pipe.dit.patch_size, stride=pipe.dit.patch_size)
            # Initialize as zeros
            pipe.dit.context_mask_patch_embedding.weight.data.zero_()
            pipe.dit.context_mask_patch_embedding.bias.data.zero_()

        if dit_checkpoint_path is not None:
            logging.info(f"Loading DIT checkpoint from {dit_checkpoint_path} to {device}...")
            if dit_checkpoint_path.endswith(".safetensors"):
                logging.info(f"Loading DIT checkpoint from {dit_checkpoint_path} to {device} as safetensors...")
                pipe.dit.load_state_dict(load_file(dit_checkpoint_path, device=device), strict=False)
            else:
                logging.info(f"Loading DIT checkpoint from {dit_checkpoint_path} to {device} as bin...")
                pipe.dit.load_state_dict(torch.load(dit_checkpoint_path, map_location=device), strict=False)
            logging.info(f"DIT checkpoint loaded successfully!")

        if condition_mode == "diagonal-context":
            import types
            for _block in pipe.dit.blocks:
                if seperated_timestep_modulation:
                    setattr(_block, "forward", types.MethodType(_diagonal_context_block_forward_separated_timestep_modulation, _block))
                else:
                    setattr(_block, "forward", types.MethodType(_diagonal_context_block_forward, _block))
                setattr(_block.self_attn, "forward", types.MethodType(_diagonal_context_self_attn_forward, _block.self_attn))
                setattr(_block.self_attn, "clear_cache", types.MethodType(_diagonal_context_self_attn_clear_cache, _block.self_attn))
                setattr(_block.self_attn, "has_cache", types.MethodType(_diagonal_context_self_attn_has_cache, _block.self_attn))
                
                _block.self_attn.cached_k_context = None
                _block.self_attn.cached_v_context = None
                _block.self_attn.cached_k_fragment = None
                _block.self_attn.cached_v_fragment = None
                _block.self_attn.cached_context_grid_size = None
                _block.self_attn.cached_fragment_grid_size = None
        elif condition_mode == "in-context":
            logging.info(f"In-context condition mode is enabled.")
        else:
            raise ValueError(f"Invalid condition mode: {condition_mode}")

        return pipe
    
    def clear_diagonal_kv_cache(self):
        """Clear diagonal context KV cache for all blocks in DiT model."""
        if self.condition_mode == "diagonal-context" and hasattr(self, 'dit'):
            for block in self.dit.blocks:
                if hasattr(block.self_attn, 'clear_cache'):
                    block.self_attn.clear_cache()
                else:
                    logging.warning(f"block.self_attn does not have clear_cache method. This is unexpected.")
            logging.info(f"Diagonal context KV cache cleared for all blocks in DiT model.")
        else:
            logging.warning(f"`clear_diagonal_kv_cache()` is only supported for diagonal-context mode. Current condition mode: {self.condition_mode}.")
    
    def training_loss(self, **inputs):
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)
        # logging.info(f"input_latents.shape={inputs['input_latents'].shape}, noise.shape={inputs['noise'].shape}")
        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)
        
        noise_pred = self.model_fn(**inputs, timestep=timestep)
        
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.scheduler.training_weight(timestep)
        return loss
    
    @torch.no_grad()
    def _call_latent_mode(
        self,
        prompt,
        negative_prompt,
        gt_cubemap: Dict[str, torch.Tensor],        # {face: [T,C,H,W]}
        cond_cubemap: Dict[str, torch.Tensor],      # {face: [T,C,H,W]}
        mask_cubemap: Dict[str, torch.Tensor],      # {face: [T,1|C,H,W]}
        num_inference_steps: int = 50,
        denoising_strength: float = 1.0,
        sigma_shift: float = 5.0,
        cfg_scale: float = 5.0,
        cfg_merge: bool = False,
        tiled: bool = False,
        tile_size: Tuple[int, int] = (30, 52),
        tile_stride: Tuple[int, int] = (15, 26),
        progress_bar_cmd=None,
        face_captions: Dict[str, str] = None,
        use_face_prompts: bool = None,
        debug_output_path: str = None,
        debug_current_step: int = None,
        debug_current_rank: int = None,
        enable_profiling: bool = False,
        skip_equirectangular_conversion: bool = False,
        **kwargs,
    ):
        # Plan windows directly from per-face masks
        sample_face = next(iter(cond_cubemap.values()))
        num_frames = sample_face.shape[0]
        order = self.planner.plan_order_from_cubemap_masks(mask_cubemap, num_frames)
        logging.info(f"Generated order: {order}")
        # Collect context snapshots for each generation step during this call
        self.context_step_snapshots = []

        self.load_models_to_device(["vae"])
        cond_latent_cubemap = {}
        for face in cond_cubemap:
            cond_latent_cubemap[face] = self.vae.encode(
                cond_cubemap[face].permute(1, 0, 2, 3).unsqueeze(0).to(device=self.device, dtype=self.torch_dtype),
                device=self.device
            ).to(self.torch_dtype)[0].permute(1, 0, 2, 3) # [T, C, H, W]
            # logging.info(f"Encoded cond face {face} (original shape {cond_cubemap[face].shape}) into latents with shape {cond_latent_cubemap[face].shape}")

        self.load_models_to_device([])

        # Get dimensions
        sample_latent_face = next(iter(cond_latent_cubemap.values()))
        temporal_frame_latent, channel_num_latent, height_latent, width_latent = sample_latent_face.shape
        
        # Optional: store padded faces for debugging
        debug_save_padded = debug_output_path is not None
        if debug_save_padded:
            self.generated_faces_with_padding = {f: [] for f in self.active_faces}

        # Store padded latents for blending in latent mode
        # Format: {face: List[(start_frame, end_frame, latent_tensor)]}
        self.generated_latents_with_padding = {f: [] for f in self.active_faces}

        # Initialize context pool
        self.context_pool.clear_history()
        # Set generation order for proper context handling
        self.context_pool.set_generation_order(order)
        if self.fragment_future_context:
            self.context_pool.detect_fragments(mask_cubemap)

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Iterate windows/faces
        order_idx = 0 # For debugging
        pbar = tqdm(order, desc="Generating faces")
        # Decide whether to use face-wise prompts
        if use_face_prompts is None:
            use_face_prompts = getattr(self, 'use_face_prompts_in_val', False)


        for face, start, end in order:
            # Use cond_cubemap as the input (already masked input condition)
            pbar.set_description(f"Generation #{order_idx}: Face {face} [{start}:{end}]")
            
            # Set current position in generation order
            self.context_pool.current_order_idx = order_idx
            
            # Build context latents using new strategy
            context_latents, context_pos_embs, current_face_pos_embs, current_face_cond_latents, current_face_cond_mask, \
                fragment_future_context = self.context_pool.build_context_latents(
                    face, start, end, cond_latent_cubemap, 
                    self.device, self.torch_dtype,
                    save_context=True, 
                    save_dir=os.path.join(debug_output_path, "inference_debug") if debug_output_path else None,
                    step=getattr(self, '_training_step', None),
                    rank=getattr(self, '_training_rank', None),
                    order_idx=order_idx
                )
            
            # clean_latents_with_padding = None
            # if self.inference_boundary_padding:
            #     clean_latents_with_padding = self.context_pool.get_current_face_clean_latents_with_padding(
            #         face, start, end, cond_cubemap,
            #         self.vae, self.device, self.torch_dtype
            #     )
            # self.load_models_to_device([])
            # Collect snapshot for validation processing (store the actual context that was built)
            # Note: context is saved during build_context_latents if requested
            self.context_step_snapshots.append({
                'order_idx': int(order_idx),
                'face': face,
                'start': int(start),
                'end': int(end),
                'context_bcthw': None  # Will be filled by validation code if needed
            })
            order_idx += 1
            # Build pipeline inputs
            # Choose per-face prompt if enabled and available
            face_prompt = None
            if use_face_prompts and isinstance(face_captions, dict):
                face_prompt = face_captions.get(face, None)
            inputs_posi = {
                "prompt": face_prompt if (face_prompt is not None and isinstance(face_prompt, str) and len(face_prompt) > 0) else prompt,
            }
            inputs_nega = {
                "negative_prompt": negative_prompt,
            }

            height_latent_padded = height_latent + 2 * self.padder.padding_width
            width_latent_padded = width_latent + 2 * self.padder.padding_width
            inputs_shared = {
                **kwargs,
                'height': height_latent_padded,
                'width': width_latent_padded,
                'num_frames': end - start,
                'cfg_scale': cfg_scale,
                'tiled': tiled,
                'tile_size': tile_size,
                'tile_stride': tile_stride,
                'rand_device': self.device,
                'cfg_merge': cfg_merge,
                'context_latents': context_latents,
                'context_pos_embs': context_pos_embs,
                'current_face_pos_embs': current_face_pos_embs,
                'current_face_cond_latents': current_face_cond_latents,
                'current_face_cond_mask': current_face_cond_mask,
                'fuse_vae_embedding_in_latents': self.fuse_vae_embedding_in_latents and current_face_cond_latents is not None,
                'condition_mode': self.condition_mode,
                'training': False,
                'fragment_future_context': fragment_future_context,
                'use_diagonal_kv_cache': self.use_diagonal_kv_cache
            }

            # Run units (default + padding)
            for unit in self.units:
                inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
            # Denoise
            self.load_models_to_device(self.in_iteration_models)
            models = {name: getattr(self, name) for name in self.in_iteration_models}
            
            # Clear diagonal KV cache before starting denoising for this face
            if self.use_diagonal_kv_cache:
                self.clear_diagonal_kv_cache()
            
            
            timesteps = self.scheduler.timesteps
            rng = enumerate(timesteps) if progress_bar_cmd is None else enumerate(progress_bar_cmd(timesteps))
            for progress_id, timestep in rng:
                if isinstance(timestep, torch.Tensor):
                    t = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
                else:
                    t = torch.tensor([timestep], dtype=self.torch_dtype, device=self.device)
                noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=t)
                if cfg_scale != 1.0:
                    if cfg_merge:
                        noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                    else:
                        noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=t)
                    noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                else:
                    noise_pred = noise_pred_posi
                inputs_shared['latents'] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared['latents'])

            denoised_latents_with_padding = inputs_shared['latents'][0].permute(1, 0, 2, 3)
            self.generated_latents_with_padding[face].append((start, end, denoised_latents_with_padding.clone().cpu()))
            denoised_latents_tchw = self.padder.crop_padding(denoised_latents_with_padding)
            self.context_pool.add_to_history(face, start, end, denoised_latents_tchw)
            self.context_pool.advance_order()
            
            pbar.update(1)

        p_latent = self.padder.padding_width
        height_latent_padded = height_latent + 2 * p_latent
        width_latent_padded = width_latent + 2 * p_latent
        
        generated_faces_padded = {}
        self.load_models_to_device([])
        self.load_models_to_device(['vae'])
        for f in self.active_faces:
            if f not in self.generated_latents_with_padding or len(self.generated_latents_with_padding[f]) == 0:
                logging.warning(f"No padded latents found for face {f} in final assembly")
                continue
            
            # Create empty tensor for this face with padding
            latent_face_tensor_padded = torch.zeros((temporal_frame_latent, channel_num_latent, height_latent_padded, width_latent_padded), 
                                                     dtype=sample_latent_face.dtype, device=self.device)
            
            # Fill in content from all padded latent windows for this face
            for start_frame, end_frame, content in self.generated_latents_with_padding[f]:
                latent_face_tensor_padded[start_frame:end_frame] = content.to(device=self.device)

            pixel_face_tensor_padded = self.vae.decode(latent_face_tensor_padded.permute(1, 0, 2, 3).unsqueeze(0), device=self.device)
            generated_faces_padded[f] = pixel_face_tensor_padded[0].permute(1, 0, 2, 3)
        
        # Apply boundary pixel blending if enabled
        if self.inference_boundary_pixel_blending:
            logging.info("Applying boundary pixel blending for all faces")
            generated_faces_cropped = {}
            for f in self.active_faces:
                if f in generated_faces_padded:
                    generated_faces_cropped[f] = self.padder_pixel.crop_padding(generated_faces_padded[f])
            
            # Process each face to blend its padding into adjacent faces
            for f in self.active_faces:
                if f not in generated_faces_padded:
                    continue
                
                # Get adjacent faces that have been decoded (without padding for blending)
                adjacent_faces_dict = {}
                for adj_face_id in self.active_faces:
                    if adj_face_id == f:
                        continue
                    if adj_face_id in generated_faces_cropped:
                        adjacent_faces_dict[adj_face_id] = generated_faces_cropped[adj_face_id]
                
                # Blend padding regions from current face into adjacent faces
                # Use padder_pixel for pixel-space blending (correct padding width)
                if len(adjacent_faces_dict) > 0:
                    updated_faces = self.padder_pixel.blend_padding_into_adjacent_faces(
                        f, generated_faces_padded[f], adjacent_faces_dict
                    )
                    # Update the cropped faces with blended results
                    for adj_face_id, updated_content in updated_faces.items():
                        generated_faces_cropped[adj_face_id] = updated_content
                    logging.info(f"Applied blending for face {f} and updated adjacent faces")
            
            # Use the blended cropped faces as final result
            generated_faces = generated_faces_cropped
        else:
            # Crop padding to get final face content (no blending)
            # Use padder_pixel for pixel-space operations (correct padding width)
            generated_faces = {}
            for f in self.active_faces:
                if f in generated_faces_padded:
                    generated_faces[f] = self.padder_pixel.crop_padding(generated_faces_padded[f])
        
        self.load_models_to_device([])
        # Verify we have all expected faces
        if len(generated_faces) != len(self.active_faces):
            logging.warning(f"Expected {len(self.active_faces)} faces but only got {len(generated_faces)} from history")
        
        cropped_faces_frames = []
        all_faces = ['F', 'R', 'B', 'L', 'U', 'D']
        for _t in range(generated_faces[all_faces[0]].shape[0]):
            frame_dict = {}
            for f in self.active_faces:
                if f in generated_faces:
                    frame_dict[f] = generated_faces[f][_t].float()
            # Fill missing faces with black for complete equirectangular conversion
            for f in all_faces:
                if f not in frame_dict:
                    # Create black face tensor with same dimensions as active faces
                    if self.active_faces:
                        sample_face = frame_dict[self.active_faces[0]]
                        frame_dict[f] = torch.zeros_like(sample_face).float()
            cropped_faces_frames.append(frame_dict)
    
        if not skip_equirectangular_conversion:
            from equilib import Cube2Equi
            cube2equi = Cube2Equi(height=height_latent * 16 * 2, width=height_latent * 16 * 4, cube_format='dict', mode='bilinear')
            equi_video = cube2equi(cropped_faces_frames, backend='native')  # [T,C,H,W]
        else:
            equi_video = generated_faces
        
        return equi_video, generated_faces, order

    @torch.no_grad()
    def _call_non_latent_mode(
        self,
        prompt,
        negative_prompt,
        gt_cubemap: Dict[str, torch.Tensor],        # {face: [T,C,H,W]}
        cond_cubemap: Dict[str, torch.Tensor],      # {face: [T,C,H,W]}
        mask_cubemap: Dict[str, torch.Tensor],      # {face: [T,1|C,H,W]}
        num_inference_steps: int = 50,
        denoising_strength: float = 1.0,
        sigma_shift: float = 5.0,
        cfg_scale: float = 5.0,
        cfg_merge: bool = False,
        tiled: bool = False,
        tile_size: Tuple[int, int] = (30, 52),
        tile_stride: Tuple[int, int] = (15, 26),
        progress_bar_cmd=None,
        face_captions: Dict[str, str] = None,
        use_face_prompts: bool = None,
        debug_output_path: str = None,
        debug_current_step: int = None,
        debug_current_rank: int = None,
        enable_profiling: bool = False,
        skip_equirectangular_conversion: bool = False,
        **kwargs,
    ):
        # Plan windows directly from per-face masks
        num_frames = next(iter(gt_cubemap.values())).shape[0]
        order = self.planner.plan_order_from_cubemap_masks(mask_cubemap, num_frames)
        # Collect context snapshots for each generation step during this call
        self.context_step_snapshots = []

        # Get dimensions
        p = self.padder.padding_width
        sample_face = next(iter(gt_cubemap.values()))
        T, C, H, W = sample_face.shape
        
        # Optional: store padded faces for debugging
        debug_save_padded = debug_output_path is not None
        if debug_save_padded:
            self.generated_faces_with_padding = {f: [] for f in self.active_faces}

        # Initialize context pool
        self.context_pool.clear_history()
        # Set generation order for proper context handling
        self.context_pool.set_generation_order(order)
        if self.fragment_future_context:
            self.context_pool.detect_fragments(mask_cubemap)

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Iterate windows/faces
        order_idx = 0   # For debugging
        pbar = tqdm(order, desc="Generating faces")
        # Decide whether to use face-wise prompts
        if use_face_prompts is None:
            use_face_prompts = getattr(self, 'use_face_prompts_in_val', False)

        for face, start, end in order:
            # Use cond_cubemap as the input (already masked input condition)
            pbar.set_description(f"Generation #{order_idx}: Face {face} [{start}:{end}]")
            
            # Set current position in generation order
            self.context_pool.current_order_idx = order_idx
            
            # Build context latents using new strategy
            self.load_models_to_device(["vae"])
            context_latents, context_pos_embs, current_face_pos_embs, \
                current_face_cond_latents, current_face_cond_mask, \
                fragment_future_context = self.context_pool.build_context_latents(
                    face, start, end, cond_cubemap, mask_cubemap,
                    self.vae, self.device, self.torch_dtype, order_idx=order_idx,
                    save_context=True, 
                    save_dir=os.path.join(debug_output_path, "inference_debug") if debug_output_path else None,
                    save_prefix="context_inference",
                    step=debug_current_step,
                    rank=debug_current_rank,
                )
            
            clean_latents_with_padding = None
            if self.inference_boundary_padding:
                clean_latents_with_padding = self.context_pool.get_current_face_clean_latents_with_padding(
                    face, start, end, cond_cubemap,
                    self.vae, self.device, self.torch_dtype
                )
            self.load_models_to_device([])
            self.context_step_snapshots.append({
                'order_idx': int(order_idx),
                'face': face,
                'start': int(start),
                'end': int(end),
                'context_bcthw': None
            })
            order_idx += 1
            face_prompt = None
            if use_face_prompts and isinstance(face_captions, dict):
                face_prompt = face_captions.get(face, None)
            inputs_posi = {
                "prompt": face_prompt if (face_prompt is not None and isinstance(face_prompt, str) and len(face_prompt) > 0) else prompt,
            }
            inputs_nega = {
                "negative_prompt": negative_prompt,
            }
            inputs_shared = {
                **kwargs,
                'height': H + 2 * p,
                'width': W + 2 * p,
                'num_frames': end - start,
                'cfg_scale': cfg_scale,
                'tiled': tiled,
                'tile_size': tile_size,
                'tile_stride': tile_stride,
                'rand_device': self.device,
                'cfg_merge': cfg_merge,
                'context_latents': context_latents,
                'context_pos_embs': context_pos_embs,
                'current_face_pos_embs': current_face_pos_embs,
                'current_face_cond_latents': current_face_cond_latents,
                'current_face_cond_mask': current_face_cond_mask,
                'fuse_vae_embedding_in_latents': self.fuse_vae_embedding_in_latents and current_face_cond_latents is not None,
                'condition_mode': self.condition_mode,
                'training': False,
                'fragment_future_context': fragment_future_context,
                'use_diagonal_kv_cache': self.use_diagonal_kv_cache
            }

            # Run units (default + padding)
            for unit in self.units:
                inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

            if 'latents' in inputs_shared and clean_latents_with_padding is not None:
                # Get the noise from current latents (for inference, this is pure noise)
                noise = inputs_shared['latents']
                # Verify shape compatibility
                if noise.shape != clean_latents_with_padding.shape:
                    logging.warning(f"noise shape {noise.shape} != clean_latents shape {clean_latents_with_padding.shape}")
                # Initialize latents by adding noise to clean latents with proper scheduling
                inputs_shared['latents'] = self.scheduler.add_noise(
                    clean_latents_with_padding.to(device=self.device, dtype=self.torch_dtype),
                    noise,
                    timestep=self.scheduler.timesteps[0]
                )
            self.load_models_to_device(self.in_iteration_models)
            models = {name: getattr(self, name) for name in self.in_iteration_models}
            if self.use_diagonal_kv_cache:
                self.clear_diagonal_kv_cache()
            
            timesteps = self.scheduler.timesteps
            rng = enumerate(timesteps) if progress_bar_cmd is None else enumerate(progress_bar_cmd(timesteps))
            for progress_id, timestep in rng:
                if isinstance(timestep, torch.Tensor):
                    t = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
                else:
                    t = torch.tensor([timestep], dtype=self.torch_dtype, device=self.device)
                noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=t)
                if cfg_scale != 1.0:
                    if cfg_merge:
                        noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                    else:
                        noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=t)
                    noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                else:
                    noise_pred = noise_pred_posi
                inputs_shared['latents'] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared['latents'])

            # Decode and merge into face buffer (BCTHW → TCHW padded)
            self.load_models_to_device(['vae'])
            decoded = self.vae.decode(inputs_shared['latents'], device=self.device,
            tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            decoded_tchw_padded = decoded[0].permute(1, 0, 2, 3) # (T, C, H+2p, W+2p)
            
            # Apply boundary pixel blending if enabled
            if self.inference_boundary_pixel_blending:
                # Get adjacent faces from context pool history for the same time window
                logging.info(f"Applying boundary pixel blending for face {face} [{start}:{end}]")
                adjacent_faces_dict = {}
                for adj_face_id in self.active_faces:
                    if adj_face_id == face:
                        continue  # Skip current face
                    # Try to get adjacent face content from context pool history for the same time window
                    adj_content = self.context_pool.get_face_content_from_history(adj_face_id, start, end)
                    if adj_content is not None:
                        adjacent_faces_dict[adj_face_id] = adj_content
                        logging.info(f"Found adjacent face {adj_face_id} content from context pool history")
                # Blend padding regions into adjacent faces
                if len(adjacent_faces_dict) > 0:
                    updated_faces = self.padder.blend_padding_into_adjacent_faces(
                        face, decoded_tchw_padded, adjacent_faces_dict
                    )
                    # Update context pool history with blended adjacent faces
                    for adj_face_id, updated_content in updated_faces.items():
                        self.context_pool.update_history(adj_face_id, start, end, updated_content)
                        logging.info(f"Updated context pool history for adjacent face {adj_face_id} [{start}:{end}]")
                        
            if debug_save_padded:
                self.generated_faces_with_padding[face].append({
                    'start': start,
                    'end': end,
                    'content': decoded_tchw_padded.clone().cpu()
                })
            
            decoded_tchw = self.padder.crop_padding(decoded_tchw_padded)
            self.load_models_to_device([])
            
            # Add generated content to context pool history (this is our only storage)
            self.context_pool.add_to_history(face, start, end, decoded_tchw)
            # Advance the order index for next iteration
            self.context_pool.advance_order()
            
            pbar.update(1)

        generated_faces = {}
        for f in self.active_faces:
            if f not in self.context_pool.history or len(self.context_pool.history[f]) == 0:
                logging.warning(f"No history found for face {f} in final assembly")
                continue
            
            # Create empty tensor for this face
            face_tensor = torch.zeros((T, C, H, W), dtype=sample_face.dtype, device=sample_face.device)
            
            # Fill in content from all history windows for this face
            for start_frame, end_frame, content in self.context_pool.history[f]:
                face_tensor[start_frame:end_frame] = content.to(device=sample_face.device)
            
            generated_faces[f] = face_tensor.float()
        
        # Verify we have all expected faces
        if len(generated_faces) != len(self.active_faces):
            logging.warning(f"Expected {len(self.active_faces)} faces but only got {len(generated_faces)} from history")
        
        cropped_faces_frames = []
        for _t in range(T):
            frame_dict = {}
            for f in self.active_faces:
                if f in generated_faces:
                    frame_dict[f] = generated_faces[f][_t]
            # Fill missing faces with black for complete equirectangular conversion
            all_faces = ['F', 'R', 'B', 'L', 'U', 'D']
            for f in all_faces:
                if f not in frame_dict:
                    # Create black face tensor with same dimensions as active faces
                    if self.active_faces:
                        sample_face = frame_dict[self.active_faces[0]]
                        frame_dict[f] = torch.zeros_like(sample_face)
            cropped_faces_frames.append(frame_dict)
    
        if not skip_equirectangular_conversion:
            from equilib import Cube2Equi
            cube2equi = Cube2Equi(height=H * 2, width=H * 4, cube_format='dict', mode='bilinear')
            
            # Process in chunks if video is longer than 54 frames to avoid OOM
            if T > 54:
                logging.info(f"[PanoramaPipeline] Video has {T} frames, using chunked processing for Cube2Equi conversion")
                equi_chunks = []
                max_frames = 54
                for start_idx in range(0, T, max_frames):
                    end_idx = min(start_idx + max_frames, T)
                    chunk_frames = cropped_faces_frames[start_idx:end_idx]
                    logging.debug(f"[PanoramaPipeline] Processing Cube2Equi chunk [{start_idx}:{end_idx}] ({end_idx - start_idx} frames)")
                    equi_chunk = cube2equi(chunk_frames, backend='native')  # [T_chunk,C,H,W]
                    equi_chunks.append(equi_chunk)
                equi_video = torch.cat(equi_chunks, dim=0)  # [T,C,H,W]
            else:
                equi_video = cube2equi(cropped_faces_frames, backend='native')  # [T,C,H,W]
        else:
            equi_video = generated_faces
        
        return equi_video, generated_faces, order
    
    def __call__(self, *args, **kwargs):
        if self.use_latent_mode:
            return self._call_latent_mode(*args, **kwargs)
        else:
            return self._call_non_latent_mode(*args, **kwargs)

    # ------------------------------------------------------------------ #
    #  Distillation helpers                                                #
    # ------------------------------------------------------------------ #

    def one_step_generate_face(
        self,
        face: str,
        start: int,
        end: int,
        cond_cubemap: Dict[str, torch.Tensor],
        mask_cubemap: Dict[str, torch.Tensor],
        prompt: str,
        negative_prompt: str,
        sigma_shift: float = 5.0,
        cfg_scale: float = 1.0,
        use_gradient_checkpointing: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        One-step denoising of a single cubemap face in **non-latent mode**.

        Runs the DiT once at the highest-noise timestep (sigma ≈ 1) and
        performs a single ODE step all the way to sigma = 0.  The generated
        latent is VAE-decoded and returned as pixel-space frames.

        This is the inner building block for `training_rollout()`.

        Args:
            face:     Which face to generate ('F', 'R', 'B', 'L', 'U', 'D').
            start, end: Pixel-frame range for the current temporal window.
            cond_cubemap:  {face: [T, C, H, W]} pixel-space conditional input.
            mask_cubemap:  {face: [T, 1, H, W]} coverage masks.
            prompt / negative_prompt: text conditioning.
            sigma_shift: passed to scheduler.set_timesteps.
            cfg_scale: classifier-free guidance scale (use 1.0 to skip).
            use_gradient_checkpointing: passed to model_fn.

        Returns:
            decoded_tchw: [T, C, H, W] pixel-space output (no padding, on CPU).
        """
        assert not self.use_latent_mode, \
            "one_step_generate_face currently supports non-latent mode only."

        p = self.padder.padding_width
        sample_face = next(iter(cond_cubemap.values()))
        _, C, H, W = sample_face.shape

        # Set scheduler to 1 step
        self.scheduler.set_timesteps(1, shift=sigma_shift)

        # Build context from the pool (VAE encode happens inside)
        self.load_models_to_device(["vae"])
        context_latents, context_pos_embs, current_face_pos_embs, \
            current_face_cond_latents, current_face_cond_mask, \
            fragment_future_context = self.context_pool.build_context_latents(
                face, start, end, cond_cubemap, mask_cubemap,
                self.vae, self.device, self.torch_dtype,
            )

        # Clean latents for noise initialisation (boundary padding)
        clean_latents_with_padding = None
        if self.inference_boundary_padding:
            clean_latents_with_padding = self.context_pool.get_current_face_clean_latents_with_padding(
                face, start, end, cond_cubemap,
                self.vae, self.device, self.torch_dtype,
            )
        self.load_models_to_device([])

        # Build inputs (mirrors _call_non_latent_mode)
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            **kwargs,
            "height": H + 2 * p,
            "width":  W + 2 * p,
            "num_frames": end - start,
            "cfg_scale": cfg_scale,
            "rand_device": self.device,
            "cfg_merge": False,
            "context_latents": context_latents,
            "context_pos_embs": context_pos_embs,
            "current_face_pos_embs": current_face_pos_embs,
            "current_face_cond_latents": current_face_cond_latents,
            "current_face_cond_mask": current_face_cond_mask,
            "fuse_vae_embedding_in_latents": (
                self.fuse_vae_embedding_in_latents
                and current_face_cond_latents is not None
            ),
            "condition_mode": self.condition_mode,
            "training": True,
            "fragment_future_context": fragment_future_context,
            "use_diagonal_kv_cache": False,
            "use_gradient_checkpointing": use_gradient_checkpointing,
        }

        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega
            )

        # Optionally start from noisy clean latents (boundary padding)
        if "latents" in inputs_shared and clean_latents_with_padding is not None:
            noise = inputs_shared["latents"]
            inputs_shared["latents"] = self.scheduler.add_noise(
                clean_latents_with_padding.to(device=self.device, dtype=self.torch_dtype),
                noise,
                timestep=self.scheduler.timesteps[0],
            )

        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}

        # Single denoising step
        timestep = self.scheduler.timesteps[0]
        t = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device) \
            if isinstance(timestep, torch.Tensor) \
            else torch.tensor([timestep], dtype=self.torch_dtype, device=self.device)

        noise_pred_posi = self.model_fn(
            **models, **inputs_shared, **inputs_posi, timestep=t
        )
        if cfg_scale != 1.0:
            noise_pred_nega = self.model_fn(
                **models, **inputs_shared, **inputs_nega, timestep=t
            )
            noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
        else:
            noise_pred = noise_pred_posi

        denoised_latents = self.scheduler.step(
            noise_pred, self.scheduler.timesteps[0], inputs_shared["latents"],
            to_final=True,
        )

        # VAE decode
        self.load_models_to_device(["vae"])
        decoded = self.vae.decode(denoised_latents, device=self.device)
        decoded_tchw_padded = decoded[0].permute(1, 0, 2, 3)  # [T, C, H+2p, W+2p]
        decoded_tchw = self.padder.crop_padding(decoded_tchw_padded)   # [T, C, H, W]
        self.load_models_to_device([])

        return decoded_tchw  # pixel-space, on CUDA

    def training_rollout(
        self,
        cond_cubemap: Dict[str, torch.Tensor],
        mask_cubemap: Dict[str, torch.Tensor],
        gt_cubemap: Dict[str, torch.Tensor],
        prompt: str,
        negative_prompt: str,
        order: List[Tuple[str, int, int]],
        sigma_shift: float = 5.0,
        cfg_scale: float = 1.0,
        use_gradient_checkpointing: bool = False,
        **kwargs,
    ) -> List[Dict]:
        """
        Face-by-face autoregressive rollout for distillation training.

        Implements the Self-Forcing / CausVid exposure-bias correction:
        after each face is generated with `one_step_generate_face()`, the
        student output is injected back into the context pool via
        `context_pool.set_context_from_generated()` so subsequent faces see
        the student's (imperfect) output rather than GT.

        Gradients are **truncated at face boundaries** (detach before storing)
        to prevent BPTT explosion across 6 faces.

        Operates in **non-latent mode** (pixel-space history).

        Args:
            cond_cubemap:  {face: [T, C, H, W]} perspective-video condition.
            mask_cubemap:  {face: [T, 1, H, W]} coverage masks.
            gt_cubemap:    {face: [T, C, H, W]} ground-truth (for regression loss).
            prompt / negative_prompt: text conditioning.
            order:  generation order list [(face, start, end), …]
                    (from planner.plan_order_from_cubemap_masks).
            sigma_shift:  scheduler shift parameter.
            cfg_scale:    guidance scale.
            use_gradient_checkpointing: memory-saving flag.

        Returns:
            rollout_results: list of dicts, one per (face, window):
                {
                  'face': str,
                  'start': int,
                  'end': int,
                  'generated_pixel': Tensor [T, C, H, W]  (on GPU, with grad),
                  'gt_pixel':        Tensor [T, C, H, W]  (on GPU, no grad),
                }
        """
        assert not self.use_latent_mode, \
            "training_rollout currently supports non-latent mode only."

        self.context_pool.clear_history()
        self.context_pool.set_generation_order(order)
        if self.fragment_future_context:
            self.context_pool.detect_fragments(mask_cubemap)

        rollout_results = []

        for face, start, end in order:
            # --- generate current face (student, one step) ---
            generated_pixel = self.one_step_generate_face(
                face=face,
                start=start,
                end=end,
                cond_cubemap=cond_cubemap,
                mask_cubemap=mask_cubemap,
                prompt=prompt,
                negative_prompt=negative_prompt,
                sigma_shift=sigma_shift,
                cfg_scale=cfg_scale,
                use_gradient_checkpointing=use_gradient_checkpointing,
                **kwargs,
            )  # [T, C, H, W]  on CUDA

            gt_pixel = gt_cubemap[face][start:end].to(
                device=self.device, dtype=self.torch_dtype
            ) if face in gt_cubemap else None

            rollout_results.append({
                "face": face,
                "start": start,
                "end": end,
                "generated_pixel": generated_pixel,
                "gt_pixel": gt_pixel,
            })

            # --- inject student output as context for subsequent faces ---
            # Gradient is TRUNCATED here (detach) to avoid BPTT across faces.
            self.context_pool.set_context_from_generated(
                face, start, end, generated_pixel.detach()
            )
            self.context_pool.advance_order()

        return rollout_results

def get_spatially_aware_timestep_emb(dit, latents, timestep, current_face_cond_mask):
    """
    Get spatially-aware timestep embedding for conditional areas.
    Args:
        dit: WanModel
        latents: torch.Tensor
        timestep: torch.Tensor
        current_face_cond_mask: torch.Tensor
    Returns:
        t: torch.Tensor (B, L, D)
    """
    # Create spatially-aware timestep based on mask
    T_latent, H_latent, W_latent = latents.shape[2], latents.shape[3], latents.shape[4]
    # Create timestep tensor for each spatial location
    timestep_first_channel = torch.ones((T_latent, H_latent, W_latent), dtype=latents.dtype, device=latents.device) * timestep
    # Set conditional areas to zero timestep based on mask
    # Take first channel: (B, 1, T', H', W')
    mask_first_channel = current_face_cond_mask[:, 0:1]
    # Apply zero timestep to masked conditional areas
    timestep_first_channel = torch.where(mask_first_channel == 1.0,
        torch.zeros_like(timestep_first_channel, device=latents.device), # Clean area for conditioning
        timestep_first_channel) # Noisy area for generation
    timestep_first_channel = torch.nn.functional.interpolate(
        timestep_first_channel, size=(T_latent, H_latent // 2, W_latent // 2), mode='nearest')
    # Reshape to token format and divide by 4 (patch size)
    timestep_first_channel = timestep_first_channel.contiguous().view(-1)
    # Original spatially-aware timestep (but with loss in frame-wise precision)
    t_original = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep_first_channel).unsqueeze(0)) # (B, L, D)
    # Additional spatially-aware timestep for other channels (compensate the loss in frame-wise precision)
    t_other_channels = dit.context_mask_patch_embedding(current_face_cond_mask) # (B, C, T', H', W')
    t_other_channels = rearrange(t_other_channels, 'b c f h w -> b (f h w) c').contiguous()
    # logging.info(f"[get_spatially_aware_timestep_emb] original timestep t_original.shape: {t_original.shape}, additional timestep t_other_channels.shape: {t_other_channels.shape}")
    return t_original + t_other_channels

def process_context_embs(dit, context_latents, context_pos_embs, is_fragment: bool = False):
    if is_fragment:
        context_embs = dit.fragment_latent_patch_embedding(context_latents)
    else:
        context_embs = dit.context_latent_patch_embedding(context_latents)
    f_ctx, h_ctx, w_ctx = context_embs.shape[2], context_embs.shape[3], context_embs.shape[4]
    # logging.info(f"[model_fn_wan_video] context_embs.shape: {context_embs.shape}, context_pos_embs.shape: {context_pos_embs.shape}")
    assert f_ctx == context_pos_embs.shape[0], f"[model_fn_wan_video] f_ctx({f_ctx}) should be equal to context_pos_embs.shape[0]({context_pos_embs.shape[0]})"
    assert h_ctx == context_pos_embs.shape[1], f"[model_fn_wan_video] h_ctx({h_ctx}) should be equal to context_pos_embs.shape[1]({context_pos_embs.shape[1]})"
    assert w_ctx == context_pos_embs.shape[2], f"[model_fn_wan_video] w_ctx({w_ctx}) should be equal to context_pos_embs.shape[2]({context_pos_embs.shape[2]})"
    context_embs = rearrange(context_embs, 'b c f h w -> b (f h w) c').contiguous()
    context_pos_embs = rearrange(context_pos_embs, 'f h w d -> (f h w) 1 d').contiguous().to(context_latents.device)
    return context_embs, context_pos_embs, (f_ctx, h_ctx, w_ctx)

def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    tea_cache = None,
    use_unified_sequence_parallel: bool = False,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
    fuse_vae_embedding_in_latents: bool = False,
    context_latents: torch.Tensor = None,
    context_pos_embs: torch.Tensor = None,
    current_face_pos_embs: torch.Tensor = None,
    current_face_cond_latents: torch.Tensor = None,
    current_face_cond_mask: torch.Tensor = None,
    condition_mode: str = None,
    training: bool = False,
    fragment_future_context: List[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_diagonal_kv_cache: bool = False,
    **kwargs,
):
    dit.freqs = dit.freqs[0].to(latents.device), dit.freqs[1].to(latents.device), dit.freqs[2].to(latents.device)
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )
    
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)
    # logging.info(f"[model_fn_wan_video] latents.shape: {latents.shape}, timestep.shape: {timestep.shape}, current_face_cond_latents.shape: {current_face_cond_latents.shape}, current_face_cond_mask.shape: {current_face_cond_mask.shape}, context_latents.shape: {context_latents.shape}, context_pos_embs.shape: {context_pos_embs.shape}, current_face_pos_embs.shape: {current_face_pos_embs.shape}")
    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents and current_face_cond_latents is not None and current_face_cond_mask is not None:
        t_latent = get_spatially_aware_timestep_emb(dit, latents, timestep, current_face_cond_mask)
        t_context = get_spatially_aware_timestep_emb(dit, context_latents, timestep, torch.ones_like(context_latents, device=latents.device))
        # t_latent: (B, L, D); t_context: (B, L, D)
        t = torch.cat([t_latent, t_context], dim=1)

        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in t_chunks]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    
    context = dit.text_embedding(context)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)
    
    # Apply fuse_vae_embedding_in_latents mechanism
    if fuse_vae_embedding_in_latents and current_face_cond_latents is not None and current_face_cond_mask is not None:
        # Apply masked fusion based on conditional coverage
        # current_face_cond_latents: (B, C, T', H', W')
        # current_face_cond_mask: (C, T', H', W') where C=4 for temporal-channel conversion
        # x (latents): (B, C, T', H', W')
        assert current_face_cond_latents.shape[2] == x.shape[2], f"Conditional latents temporal dimension {current_face_cond_latents.shape[2]} exceeds input latents {x.shape[2]}"
        # For batch dimension consistency
        assert current_face_cond_latents.shape[0] == x.shape[0], f"Conditional latents batch dimension {current_face_cond_latents.shape[0]} should be equal to input latents {x.shape[0]}"
        assert current_face_cond_mask.shape[1] == x.shape[1], f"Conditional mask channel dimension {current_face_cond_mask.shape[1]} should be equal to input latents {x.shape[1]}"
        assert current_face_cond_mask.shape[2] == x.shape[2], f"Conditional mask temporal dimension {current_face_cond_mask.shape[2]} should be equal to input latents {x.shape[2]}"
        # Apply masked fusion: use conditional latents where mask=1, keep original where mask=0
        x = torch.where(current_face_cond_mask == 1.0, current_face_cond_latents, x)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    
    x, (f, h, w) = dit.patchify(x, control_camera_latents_input)
    x_grid_size = (f, h, w)
    freqs = rearrange(current_face_pos_embs, 'f h w d -> (f h w) 1 d').contiguous().to(x.device)
    context_grid_size = None
    last_fragment_grid_size = None
    # In-context conditioning: concatenate context latents as extra tokens
    if context_latents is not None:
        assert context_pos_embs is not None, "[model_fn_wan_video] context_pos_embs is not provided"
        # Expect (B, C, T, H, W) in the same latent grid units as x before patching
        # Patch-embed context latents via the same conv3d to align channel dim
        context_embs, context_pos_embs, context_grid_size = process_context_embs(dit, context_latents, context_pos_embs)
        all_fragment_embs = []
        all_fragment_pos_embs = []
        if fragment_future_context is not None:
            _frag_idx = 0
            for fragment_latents, fragment_pos_embs in fragment_future_context:
                # logging.info(f"[model_fn_wan_video] Frag-#{_frag_idx} fragment_latents.shape: {fragment_latents.shape}, fragment_pos_embs.shape: {fragment_pos_embs.shape}")
                _frag_idx += 1
                fragment_embs, fragment_pos_embs, last_fragment_grid_size = process_context_embs(dit, fragment_latents, fragment_pos_embs, is_fragment=True)
                all_fragment_embs.append(fragment_embs)
                all_fragment_pos_embs.append(fragment_pos_embs)
                # logging.info(f"[model_fn_wan_video] fragment_embs.shape: {fragment_embs.shape}, fragment_pos_embs.shape: {fragment_pos_embs.shape}, last_fragment_grid_size: {last_fragment_grid_size}")
            # logging.info(f"[model_fn_wan_video] fragment_future_context: {len(all_fragment_embs)} fragments found")
        # Appending fragment context sequence to the main context sequence
        if len(all_fragment_embs) > 0:
            # logging.info(f"[model_fn_wan_video] [Before appending fragment context sequence to the main context sequence] context_embs.shape: {context_embs.shape}, context_pos_embs.shape: {context_pos_embs.shape}")
            context_embs = torch.cat([context_embs, torch.cat(all_fragment_embs, dim=1)], dim=1)
            context_pos_embs = torch.cat([context_pos_embs, torch.cat(all_fragment_pos_embs, dim=0)], dim=0)
            # logging.info(f"[model_fn_wan_video] [After appending fragment context sequence] context_embs.shape: {context_embs.shape}, context_pos_embs.shape: {context_pos_embs.shape}")
    
        if condition_mode == "in-context" or condition_mode == "segmented-context" or condition_mode == "diagonal-context":
            # logging.info(f"[In-context conditioning] x.shape: {x.shape}, context_embs.shape: {context_embs.shape}")
            x = torch.cat([x, context_embs], dim=1)
            # logging.info(f"[In-context conditioning] freqs.shape: {freqs.shape}, context_pos_embs.shape: {context_pos_embs.shape}")
            freqs = torch.cat([freqs, context_pos_embs], dim=0)    
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False

    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in chunks]
            x = chunks[get_sequence_parallel_rank()]
            
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        for _, block in enumerate(dit.blocks):
            if use_gradient_checkpointing_offload or use_gradient_checkpointing:
                with torch.autograd.graph.save_on_cpu() if use_gradient_checkpointing_offload else nullcontext():
                    if condition_mode == "kv-context":
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                                x, context, t_mod, freqs, context_embs, context_pos_embs, training,
                                use_reentrant=False,
                            )
                    elif condition_mode == "segmented-context":
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs, context_grid_size, last_fragment_grid_size, 
                            use_reentrant=False,
                        )
                    elif condition_mode == "diagonal-context":
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs, x_grid_size, context_grid_size, last_fragment_grid_size, use_diagonal_kv_cache,
                            use_reentrant=False,
                        )
                    else:
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
            else:
                if condition_mode == "kv-context":
                    x = block(x, context, t_mod, freqs, context_embs=context_embs, context_pos_embs=context_pos_embs, training=training)
                elif condition_mode == "segmented-context":
                    x = block(x, context, t_mod, freqs, context_grid_size=context_grid_size, last_fragment_grid_size=last_fragment_grid_size)
                elif condition_mode == "diagonal-context":
                    x = block(x, context, t_mod, freqs, x_grid_size=x_grid_size, context_grid_size=context_grid_size, last_fragment_grid_size=last_fragment_grid_size, use_cache=use_diagonal_kv_cache)
                else:
                    x = block(x, context, t_mod, freqs)
                    
        if tea_cache is not None:
            tea_cache.store(x)
            
    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x
           
    x = dit.unpatchify(x, (f, h, w))
    return x
