#!/usr/bin/env python3
import torch
import os
import json
import argparse
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
import random

from huggingface_hub import snapshot_download

from models.panorama_pipeline import PanoramaWanPipeline, get_model_configs
from dataset.odv360 import ODV360Dataset
from dataset.base_dataset import CudaPreprocessor


HF_REPO_ID = "TencentARC/CubeComposer"
VARIANT_TO_SUBDIR = {
    "2k": "cubecomposer-3k",
    "3k": "cubecomposer-3k",
    "4k": "cubecomposer-4k",
}
DEFAULT_TEST_MODE = "3k"


def load_args_from_json(json_path):
    """Load training arguments from JSON file."""
    with open(json_path, 'r') as f:
        args_dict = json.load(f)
    
    # Convert dict to argparse.Namespace for compatibility
    args = argparse.Namespace(**args_dict)
    print(f"Loaded args from {json_path}")
    print("Key arguments:")
    print(f"  base_model: {getattr(args, 'base_model', 'wan2.2')}")
    print(f"  cube_map_size: {args.cube_map_size}")
    print(f"  window_length: {args.window_length}")
    print(f"  active_faces: {args.active_faces}")
    print(f"  condition_mode: {getattr(args, 'condition_mode', None)}")
    print(f"  max_history_windows: {getattr(args, 'max_history_windows', 2)}")
    print(f"  fragment_future_context: {getattr(args, 'fragment_future_context', None)}")
    print(f"  use_global_sink_token: {getattr(args, 'use_global_sink_token', False)}")
    
    return args


def _download_variant_from_hf(test_mode: str, cache_dir: str = "./hf_models_cache"):
    """
    Download CubeComposer weights + args for a given test_mode from Hugging Face.
    Falls back to DEFAULT_TEST_MODE (3k) if test_mode is invalid.
    """
    mode = test_mode or DEFAULT_TEST_MODE
    if mode not in VARIANT_TO_SUBDIR:
        print(f"[WARN] Unknown test_mode='{mode}', falling back to '{DEFAULT_TEST_MODE}'.")
        mode = DEFAULT_TEST_MODE

    subdir = VARIANT_TO_SUBDIR[mode]
    print(f"Using CubeComposer variant '{subdir}' for test_mode='{mode}'.")

    local_root = snapshot_download(
        repo_id=HF_REPO_ID,
        cache_dir=cache_dir,
        allow_patterns=[f"{subdir}/*"],
    )

    args_json = os.path.join(local_root, subdir, "args.json")
    checkpoint_path = os.path.join(local_root, subdir, "model.safetensors")

    if not os.path.exists(args_json):
        raise FileNotFoundError(f"Downloaded args.json not found at {args_json}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Downloaded checkpoint not found at {checkpoint_path}")

    print(f"Resolved args.json from HF: {args_json}")
    print(f"Resolved checkpoint from HF: {checkpoint_path}")
    return args_json, checkpoint_path


def resolve_args_and_checkpoint(
    args_json: str | None,
    checkpoint_path: str | None,
    test_mode: str,
):
    """
    Resolve args.json path and checkpoint path for testing.

    Priority:
      1. If both args_json and checkpoint_path are provided and exist locally, use them.
      2. Otherwise, automatically download the appropriate variant from HF based on test_mode.
         If test_mode is invalid, fall back to 3k (cubecomposer-3k).
    """
    local_args_ok = args_json is not None and os.path.exists(args_json)
    local_ckpt_ok = checkpoint_path is not None and os.path.exists(checkpoint_path)

    if local_args_ok and local_ckpt_ok:
        print("Using locally provided args.json and checkpoint.")
        return args_json, checkpoint_path

    if args_json and not local_args_ok:
        print(f"[WARN] Provided args_json does not exist: {args_json}")
    if checkpoint_path and not local_ckpt_ok:
        print(f"[WARN] Provided checkpoint_path does not exist: {checkpoint_path}")

    print("Falling back to Hugging Face weights (defaulting to 3k if test_mode is invalid).")
    return _download_variant_from_hf(test_mode)


def create_test_dataset(args, num_samples=None, sample_indices=None, start_idx=0):
    """Create test dataset for validation."""
    
    # ODV360 validation dataset
    if not hasattr(args, 'odv_root_dir') or not args.odv_root_dir or not os.path.exists(args.odv_root_dir):
        raise ValueError(f"ODV360 root directory not found or not specified: {getattr(args, 'odv_root_dir', 'N/A')}")
    
    odv_val = ODV360Dataset(
        root_dir=args.odv_root_dir,
        division='test/HR',
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        cube_map_size=args.cube_map_size,
        window_length=args.window_length,
        active_faces=args.active_faces.split(',') if isinstance(args.active_faces, str) else args.active_faces,
        use_random_fov=False,
        use_random_num_waypoints=False,
        trajectory_mode=getattr(args, "trajectory_mode", "rotation"),  # Default to rotation mode for test
        keep_original_resolution=getattr(args, 'keep_original_resolution', False),
    )
    
    print(f"Total validation samples available: {len(odv_val)}")
    
    # Select samples
    if sample_indices is not None:
        # Use specific indices
        indices = sample_indices
        print(f"Using specific sample indices: {indices}")
    elif num_samples is not None:
        # Use first num_samples from start_idx
        end_idx = min(start_idx + num_samples, len(odv_val))
        indices = list(range(start_idx, end_idx))
        print(f"Using samples from index {start_idx} to {end_idx-1}")
    else:
        # Use all samples
        indices = list(range(len(odv_val)))
        print(f"Using all {len(odv_val)} samples")
    
    subset = torch.utils.data.Subset(odv_val, indices)
    return subset


def create_pipeline_from_args(args, checkpoint_path=None):
    """
    Create pipeline from training arguments and load checkpoint if provided.
    
    Args:
        args: Training arguments loaded from args.json
        checkpoint_path: Path to model checkpoint
    """
    
    # Get model configs, optionally using local base_model_path for offline loading
    model_configs = get_model_configs(
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        local_model_path=getattr(args, 'base_model_path', None),
    )
    
    fragment_future_context = getattr(args, 'fragment_future_context', None)
    use_global_sink_token = getattr(args, 'use_global_sink_token', False)
    
    # Common pipeline arguments (aligned with training/validation configuration)
    pipeline_args = {
        'torch_dtype': torch.bfloat16,
        'device': "cuda",
        'model_configs': model_configs,
        'padding_width': getattr(args, 'padding_width', 16),
        'window_length': args.window_length,
        'cube_map_size': args.cube_map_size,
        'active_faces': args.active_faces.split(',') if isinstance(args.active_faces, str) else args.active_faces,
        'use_vanilla_pos_embs': getattr(args, 'use_vanilla_pos_embs', False),
        'dit_checkpoint_path': checkpoint_path,
        'condition_mode': getattr(args, 'condition_mode', None),
        'max_history_windows': getattr(args, 'max_history_windows', 2),
        'fragment_future_context': fragment_future_context,
        'use_global_sink_token': use_global_sink_token,
        'inference_boundary_padding': getattr(args, 'inference_boundary_padding', True),
        'inference_boundary_pixel_blending': getattr(args, 'inference_boundary_pixel_blending', True),
        'fuse_vae_embedding_in_latents': getattr(args, 'fuse_vae_embedding_in_latents', False),
        'seperated_timestep_modulation': getattr(args, 'seperated_timestep_modulation', False),
        'use_tiled_vae': getattr(args, 'use_tiled_vae', False),
        'use_latent_mode': getattr(args, 'use_latent_mode', False),
        'use_diagonal_kv_cache': getattr(args, 'use_diagonal_kv_cache', False),
        'always_full_context': getattr(args, 'always_full_context', False),
    }
    
    # For open-source usage we only support Wan2.2-style pipeline
    pipe = PanoramaWanPipeline.from_pretrained(**pipeline_args)
    
    # Propagate validation-time face prompt behaviour from training config if available
    setattr(pipe, 'use_face_prompts_in_val', getattr(args, 'use_face_prompts_in_val', False))
    
    return pipe


def run_test(pipe, test_dataset, output_dir,
                                num_inference_steps=20, cfg_scale=5.0, use_face_prompts=False,
                                save_video_format='mp4', trajectories=None):
    """Run test inference with optional attention visualization."""
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    video_output_dir = os.path.join(output_dir, "generated_videos")
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Create data loader
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0], num_workers=0)
    
    # Create preprocessor (follow pipeline device)
    preprocessor_device = getattr(pipe, "device", "cuda")
    preprocessor = CudaPreprocessor(device=preprocessor_device)
    
    results = []
    
    for idx, data in enumerate(tqdm(dataloader, desc="Running test inference")):
        # If fixed trajectory definitions are provided, use per-sample seed (if any)
        # so that random trajectory generation becomes reproducible for each sample.
        current_traj = None
        if trajectories is not None:
            if idx < len(trajectories):
                current_traj = trajectories[idx]
            else:
                current_traj = None
        if current_traj is not None and "seed" in current_traj:
            seed = int(current_traj["seed"])
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        print(f"\n--- Processing sample {idx} ---")

        # Inject per-frame rotations from the trajectory file so that
        # CudaPreprocessor uses the exact exported trajectory instead of
        # re-generating a random one (even with the same seed the generation
        # logic may differ across code versions).
        if current_traj is not None and "rotations" in current_traj:
            fixed_rots = np.asarray(
                [[float(r["roll"]), float(r["pitch"]), float(r["yaw"])]
                 for r in current_traj["rotations"]],
                dtype=np.float64,
            )
            data["perspective_params"] = data["perspective_params"].copy()
            data["perspective_params"]["fixed_rotations"] = fixed_rots
            data["perspective_params"]["simulate_camera_shake"] = False

        # Apply CUDA preprocessing
        data = preprocessor.preprocess(data)
        
        gt_cubemap = data['gt_cubemap']
        cond_cubemap = data['cond_cubemap']
        mask_cubemap = data['mask_cubemap']
        perspective_video = data.get('perspective_video', None)
        caption = data.get('caption', '')
        face_captions = data.get('face_captions', None)
        # Get metadata from original data (not in preprocessed data)
        video_path = data.get('video_path', '')
        metadata = {'id': Path(video_path).stem if video_path else f'sample_{idx}'}
        
        # Print sample info
        num_frames = next(iter(gt_cubemap.values())).shape[0]
        video_id = metadata.get('id', f'sample_{idx}')
        print(f"Sample info: {video_id}, {num_frames} frames, faces: {list(gt_cubemap.keys())}")
        print(f"Caption: {caption}")
        
        sample_output_dir = os.path.join(video_output_dir, f"sample_{idx:03d}")
        os.makedirs(sample_output_dir, exist_ok=True)
        
        # Save input perspective video (cropped from dataset) into sample directory
        if perspective_video is not None:
            try:
                input_perspective_path = os.path.join(sample_output_dir, "input_perspective.mp4")
                pers_thwc = (perspective_video.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu()
                torchvision.io.write_video(input_perspective_path, pers_thwc, fps=8)
            except Exception as e:
                print(f"Warning: failed to save input perspective video for sample {idx}: {e}")
        
        # Run generation
        try:
            with torch.no_grad():
                equi_video, generated_faces, order = pipe(
                    prompt=caption,
                    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                    gt_cubemap=gt_cubemap,
                    cond_cubemap=cond_cubemap,
                    mask_cubemap=mask_cubemap,
                    num_inference_steps=num_inference_steps,
                    denoising_strength=1.0,
                    sigma_shift=5.0,
                    cfg_scale=cfg_scale,
                    face_captions=face_captions,
                    use_face_prompts=use_face_prompts,
                    tiled=False,
                )

            # Save output equirectangular video
            if save_video_format == 'mp4':
                equi_video_path = os.path.join(sample_output_dir, "generated_equirectangular.mp4")
                video_thwc = (equi_video.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu()
                torchvision.io.write_video(equi_video_path, video_thwc, fps=8)
            else:
                # Save as frames
                for t in range(equi_video.shape[0]):
                    frame = equi_video[t].permute(1, 2, 0).cpu().numpy()  # H, W, C
                    frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
                    frame_path = os.path.join(sample_output_dir, f"frame_{t:03d}.jpg")
                    
                    from PIL import Image
                    Image.fromarray(frame).save(frame_path)
            
            # Save individual cube faces
            faces_dir = os.path.join(sample_output_dir, "cube_faces")
            os.makedirs(faces_dir, exist_ok=True)
            for face_name, face_video in generated_faces.items():
                face_path = os.path.join(faces_dir, f"face_{face_name}.mp4")
                face_thwc = (face_video.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu()
                torchvision.io.write_video(face_path, face_thwc, fps=8)
            
            # Save generation order info
            order_info = {
                'video_id': video_id,
                'caption': caption,
                'order': [(face, int(start), int(end)) for face, start, end in order],
                'num_frames': int(num_frames),
                'generated_faces': list(generated_faces.keys()),
                'num_inference_steps': num_inference_steps,
                'cfg_scale': cfg_scale,
            }

            # Attach per-sample camera trajectory definition if provided
            if current_traj is not None:
                order_info['camera_trajectory'] = current_traj
            
            with open(os.path.join(sample_output_dir, "generation_info.json"), 'w') as f:
                json.dump(order_info, f, indent=2)
            
            results.append({
                'sample_idx': idx,
                'video_id': video_id,
                'num_frames': num_frames,
                'generation_order': order_info['order'],
                'success': True,
                'error': None
            })
            
            print(f"✓ Sample {idx} completed successfully")
            print(f"  Generated {equi_video.shape[0]} frames")
            print(f"  Output saved to: {sample_output_dir}")
            
        except Exception as e:
            print(f"✗ Sample {idx} failed with error: {e}")
            import traceback
            traceback.print_exc()
            
            results.append({
                'sample_idx': idx,
                'video_id': video_id,
                'num_frames': num_frames,
                'success': False,
                'error': str(e)
            })
    
    # Save summary results
    with open(os.path.join(output_dir, "test_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    successful = sum(1 for r in results if r['success'])
    print(f"\n{'='*80}")
    print(f"=== Test Summary ===")
    print(f"{'='*80}")
    print(f"Total samples: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="CubeComposer Inference Script")
    parser.add_argument(
        "--args_json",
        type=str,
        default=None,
        help=(
            "Path to args.json file from training (contains model and dataset arguments). "
            "If omitted or not found, will be resolved automatically from the CubeComposer "
            "Hugging Face repo based on --test_mode."
        ),
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help=(
            "Path to model checkpoint (.safetensors file). If omitted or not found, will "
            "be resolved automatically from the CubeComposer Hugging Face repo based on "
            "--test_mode."
        ),
    )
    parser.add_argument("--output_dir", type=str, default="./test_outputs",
                       help="Directory to save test outputs")
    parser.add_argument("--base_model_path", type=str, default=None,
                       help="Path to base model weights")
    parser.add_argument("--odv_root_dir", type=str, default=None,
                       help="Root directory of ODV360 dataset (overrides value in args.json)")
    
    # Sample selection
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of test samples to process (None = all)")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="Starting sample index")
    parser.add_argument("--sample_indices", type=int, nargs='+',
                       help="Specific sample indices to evaluate")
    
    # Inference settings
    parser.add_argument("--num_inference_steps", type=int, default=20,
                       help="Number of denoising steps for generation")
    parser.add_argument("--cfg_scale", type=float, default=5.0,
                       help="Classifier-free guidance scale")
    parser.add_argument("--use_face_prompts", action="store_true",
                       help="Use face-wise prompts if available")

    # Output format
    parser.add_argument("--save_video_format", type=str, default="mp4", choices=["mp4", "frames"],
                       help="Video save format (mp4 or frames)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run inference on")
    parser.add_argument(
        "--test_mode",
        type=str,
        default="3k",
        choices=["2k", "3k", "4k"],
        help=(
            "Output resolution mode. 2k/3k use CubeComposer 3k weights (cubecomposer-3k), "
            "4k uses CubeComposer 4k weights (cubecomposer-4k)."
        ),
    )
    parser.add_argument(
        "--trajectory_file",
        type=str,
        default=None,
        help=(
            "Optional path to a fixed camera trajectory JSON exported by "
            "export_trajectory.py. It can contain either a single trajectory "
            "dict or a list/wrapped dict of multiple trajectories. If provided, "
            "per-sample trajectories will be recorded into generation_info.json "
            "and their seeds will be used to make the trajectory generation "
            "reproducible."
        ),
    )
    
    args = parser.parse_args()

    # Resolve args.json and checkpoint, potentially downloading from HF.
    resolved_args_json, resolved_checkpoint_path = resolve_args_and_checkpoint(
        args_json=args.args_json,
        checkpoint_path=args.checkpoint_path,
        test_mode=args.test_mode,
    )
    
    print("=" * 80)
    print("=== Panorama Video Generation Test ===")
    print("=" * 80)
    print(f"Args JSON: {resolved_args_json}")
    print(f"Checkpoint: {resolved_checkpoint_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Num inference steps: {args.num_inference_steps}")
    print(f"CFG scale: {args.cfg_scale}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Load optional fixed trajectory definitions
    trajectories = None
    if args.trajectory_file is not None:
        if not os.path.exists(args.trajectory_file):
            raise FileNotFoundError(f"Trajectory file not found: {args.trajectory_file}")
        with open(args.trajectory_file, "r") as f:
            raw = json.load(f)

        # Support multiple formats for convenience / backward compatibility:
        # 1) Single trajectory dict (has 'rotations')
        # 2) List of trajectory dicts
        # 3) Wrapped dict: {"version": 1, "num_samples": N, "trajectories": [...]}
        if isinstance(raw, dict) and "trajectories" in raw:
            trajectories = raw["trajectories"]
        elif isinstance(raw, list):
            trajectories = raw
        elif isinstance(raw, dict) and "rotations" in raw:
            trajectories = [raw]
        else:
            raise ValueError(
                f"Unrecognized trajectory file format in {args.trajectory_file}. "
                f"Expected a dict with 'rotations', a list of dicts, or a dict "
                f"with a 'trajectories' field."
            )

        print(f"Loaded fixed trajectories from: {args.trajectory_file}")
        print(f"  num_trajectories={len(trajectories)}")
        first = trajectories[0]
        print(
            f"  [0] mode={first.get('trajectory_mode')}, "
            f"FoV={first.get('fov_x')}, "
            f"num_waypoints={first.get('num_waypoints')}, "
            f"seed={first.get('seed')}"
        )

    # Load training arguments
    training_args = load_args_from_json(resolved_args_json)
    training_args.base_model_path = args.base_model_path
    if args.odv_root_dir is not None:
        training_args.odv_root_dir = args.odv_root_dir
    
    # Create test dataset
    print(f"\nCreating test dataset...")
    test_dataset = create_test_dataset(
        training_args, 
        num_samples=args.num_samples,
        sample_indices=args.sample_indices,
        start_idx=args.start_idx
    )
    print(f"Test dataset created with {len(test_dataset)} samples")

    # If we have trajectory definitions, validate count consistency
    if trajectories is not None:
        if len(trajectories) != len(test_dataset):
            raise ValueError(
                f"Trajectory count ({len(trajectories)}) does not match test dataset size "
                f"({len(test_dataset)}). Please ensure export_trajectory.py --num_samples "
                f"and test.py --num_samples / selection are aligned."
            )
    
    # Create pipeline and load checkpoint
    print("\nCreating pipeline and loading checkpoint...")
    pipe = create_pipeline_from_args(training_args, checkpoint_path=resolved_checkpoint_path)
    pipe.to(args.device)
    print("Pipeline created and checkpoint loaded successfully")
    
    print(f"\nRunning test inference...")
    run_test(
        pipe=pipe,
        test_dataset=test_dataset,
        output_dir=args.output_dir,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        use_face_prompts=args.use_face_prompts,
        save_video_format=args.save_video_format,
        trajectories=trajectories,
    )
    
    print(f"\n{'='*80}")
    print(f"=== Test completed ===")
    print(f"{'='*80}")
    print(f"Check outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
