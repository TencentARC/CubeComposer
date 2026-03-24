import torch
import numpy as np
from typing import Optional, Union, Tuple, Dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, model_fn_wan_video
from models.panorama import PanoVideoProcessor, CubeMapPadder, GenerationOrderPlanner
from equilib import Equi2Cube, Equi2Equi

from diffsynth.utils import ModelConfig
from diffsynth.models import ModelManager, load_state_dict

class CubeMapPaddingUnit:
    def __init__(self):
        self.input_params = ("current_face", "adj_faces", "latents", "vace_video")

    def process(self, pipe, current_face, adj_faces, latents, vace_video):
        # Pad pixel-level face before encoding to latents
        face_tensors = {current_face: vace_video, **adj_faces}
        padded_face = pipe.padder.pad_face(face_tensors, current_face)
        pipe.load_models_to_device(["vae"])
        # Convert TCHW -> BCTHW for VAE
        padded_face_bcthw = padded_face.permute(1, 0, 2, 3).unsqueeze(0)
        padded_latents = pipe.vae.encode(padded_face_bcthw, device=pipe.device).to(pipe.torch_dtype)
        return {"latents": padded_latents}


def model_fn_panorama_video(**kwargs):
    # forward to base model_fn with optional perspective_latents for in-context conditioning
    return model_fn_wan_video(**kwargs)


class PanoramaWanPipeline(WanVideoPipeline):
    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, padding_width=16, window_length=8, stride=4):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.padder = CubeMapPadder(padding_width=padding_width)
        self.equi2cube = Equi2Cube(w_face=512, cube_format='dict')
        self.equi2equi = Equi2Equi(height=512, width=1024, mode='bilinear')
        self.planner = GenerationOrderPlanner(window_length=window_length, stride=stride, cube_map_size=512)
        self.units.append(CubeMapPaddingUnit())
        self.model_fn = model_fn_panorama_video

    @torch.no_grad()
    def __call__(self, perspective_video: torch.Tensor, perspective_mask: torch.Tensor, rotations: np.ndarray, caption, **kwargs):
        # Sliding-window plan based on mask coverage
        num_frames = perspective_video.shape[0]
        mask_2d = perspective_mask.squeeze(1) if perspective_mask.ndim == 4 and perspective_mask.shape[1] == 1 else perspective_mask.mean(dim=1) if perspective_mask.ndim == 4 else perspective_mask
        order = self.planner.plan_order(mask_2d, num_frames)

        # Buffers for outputs
        p = self.padder.padding_width
        T, C, H, W = perspective_video.shape
        generated_faces: Dict[str, torch.Tensor] = {f: torch.zeros((T, C, H + 2 * p, W + 2 * p), dtype=perspective_video.dtype, device=perspective_video.device) for f in ['F','R','B','L','U','D']}

        # Iterate windows and faces
        for face, start, end in order:
            window_video = perspective_video[start:end]
            window_mask = perspective_mask[start:end]
            adj = {adj: generated_faces[adj][start:end] for adj in ['R','L','U','D']}
            print(f"window_video.shape: {window_video.shape}")
            print(f"window_mask.shape: {window_mask.shape}")
            print(f"adj: {adj}")
            print(f"face: {face}, start: {start}, end: {end}")
            # Prepare initial latents from the window (unpadded) if needed
            window_bcthw = window_video.permute(1, 0, 2, 3).unsqueeze(0)
            inputs_shared = {
                **kwargs,
                'latents': self.vae.encode(window_bcthw, device=self.device).to(self.torch_dtype),
                'vace_video': window_video,
                'vace_video_mask': window_mask,
                'current_face': face,
                'adj_faces': adj,
                'height': window_video.shape[-2] + 2 * self.padder.padding_width,
                'width': window_video.shape[-1] + 2 * self.padder.padding_width,
                'num_frames': end - start,
                'cfg_scale': 1,
                'tiled': False,
                'rand_device': self.device,
                'cfg_merge': False,
                'vace_scale': 1,
            }
            for unit in self.units:
                inputs_shared, inputs_posi, _ = self.unit_runner(unit, self, inputs_shared, {'prompt': caption}, {})
                
                
            # Diffusion Denoising Loop
            # self.load_models_to_device(self.in_iteration_models)
            # models = {name: getattr(self, name) for name in self.in_iteration_models}
            # for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            #     # Switch DiT if necessary
            #     if timestep.item() < switch_DiT_boundary * self.scheduler.num_train_timesteps and self.dit2 is not None and not models["dit"] is self.dit2:
            #         self.load_models_to_device(self.in_iteration_models_2)
            #         models["dit"] = self.dit2
                    
            #     # Timestep
            #     timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
                
            #     # Inference
            #     noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            #     if cfg_scale != 1.0:
            #         if cfg_merge:
            #             noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
            #         else:
            #             noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
            #         noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            #     else:
            #         noise_pred = noise_pred_posi

            #     # Scheduler
            #     inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
            #     if "first_frame_latents" in inputs_shared:
            #         inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]   
                
            # Please also merge the sub window and faces together. If padding is applied, we need to crop the padding part before merging into the final output.
                
               
            self.load_models_to_device(['vae'])
            decoded = self.vae.decode(inputs_shared['latents'], device=self.device)
            # Convert BCTHW -> TCHW
            decoded_tchw = decoded[0].permute(1, 0, 2, 3)
            generated_faces[face][start:end] = decoded_tchw

        # Crop and warp to equirectangular
        cropped_faces = {face: self.padder.crop_padding(generated_faces[face]) for face in generated_faces}
        from equilib import Cube2Equi
        cube2equi = Cube2Equi(height=512, width=1024, cube_format='dict', mode='bilinear')
        equi_video = cube2equi(cropped_faces, backend='native')
        return equi_video, generated_faces, order
    
    
    
    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"),
        redirect_common_files: bool = True,
        use_usp=False,
        padding_width=16, window_length=8, stride=4
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern]
        
        # Initialize pipeline
        pipe = PanoramaWanPipeline(device=device, torch_dtype=torch_dtype, padding_width=padding_width, window_length=window_length, stride=stride)
        if use_usp: pipe.initialize_usp()
        
        # Download and load models
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary(use_usp=use_usp)
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )
        
        # Load models
        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
        dit = model_manager.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_manager.fetch_model("wan_video_vae")
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        pipe.motion_controller = model_manager.fetch_model("wan_video_motion_controller")
        pipe.vace = model_manager.fetch_model("wan_video_vace")
        
        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer
        tokenizer_config.download_if_necessary(use_usp=use_usp)
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_config.path)
        
        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()
        return pipe


