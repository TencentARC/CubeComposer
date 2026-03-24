#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from dataset.odv360 import ODV360Dataset
from models.panorama import PanoVideoProcessor


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_args_from_json(json_path: str) -> argparse.Namespace:
    """Load training args.json into an argparse.Namespace (minimal subset)."""
    with open(json_path, "r") as f:
        args_dict = json.load(f)
    return argparse.Namespace(**args_dict)


def _export_single_trajectory(
    odv_val: ODV360Dataset,
    sample_index: int,
    seed: int,
    train_args: argparse.Namespace,
    device: str,
) -> dict:
    """
    Export trajectory for a single sample index with a specific seed.
    Returns a dict that can be stored in the output JSON.
    """
    set_global_seed(seed)

    sample = odv_val[sample_index]
    equi_video = sample["equi_video"].to(device)
    meta = sample.get("video_path") or sample.get("id") or f"idx_{sample_index}"

    # Use the same panorama processor used in training / testing.
    # This returns:
    # - pers_video: perspective video (not saved here)
    # - pers_mask: coverage mask (not saved here)
    # - rotations: np.ndarray [T, 3] of (roll, pitch, yaw) in degrees
    pers_height = sample["height"]
    pers_width = sample["width"]

    # Build trajectory_config equivalent to rotation mode
    trajectory_config = {
        "trajectory_type": "multi_waypoint",
        "num_waypoints": 2,
        "start_from_center": False,
        "require_high_coverage": True,
        "distant_start_range": {
            "pitch_range": (-40, 40),
            "yaw_range": (-150, 150),
            "roll_range": (-30, 30),
        },
        "segment_length": 27,
    }

    print(
        f"[Trajectory] Generating sample_index={sample_index}, seed={seed}, "
        f"FoV=90, num_waypoints=2, rotation mode..."
    )
    pers_video, pers_mask, rotations = PanoVideoProcessor.synthesize_perspective_video(
        equi_video,
        pers_height=pers_height,
        pers_width=pers_width,
        fov_x=90.0,
        num_waypoints=trajectory_config["num_waypoints"],
        pitch_range=sample["perspective_params"]["pitch_range"],
        yaw_range=sample["perspective_params"]["yaw_range"],
        roll_range=sample["perspective_params"]["roll_range"],
        simulate_camera_shake=sample["perspective_params"].get("simulate_camera_shake", True),
        shake_magnitude=sample["perspective_params"].get("shake_magnitude", 0.2),
        use_diverse_trajectories=sample["perspective_params"].get("use_diverse_trajectories", True),
        trajectory_config=trajectory_config,
    )

    num_frames = int(rotations.shape[0])
    rotations_list = []
    for t in range(num_frames):
        roll_deg, pitch_deg, yaw_deg = [float(x) for x in rotations[t]]
        rotations_list.append(
            {
                "frame": t,
                "roll": roll_deg,
                "pitch": pitch_deg,
                "yaw": yaw_deg,
            }
        )

    output = {
        "trajectory_mode": "rotation",
        "fov_x": 90.0,
        "num_waypoints": 2,
        "start_from_center": False,
        "require_high_coverage": True,
        "segment_length": trajectory_config["segment_length"],
        "num_frames": num_frames,
        "seed": int(seed),
        "dataset": "ODV360",
        "dataset_index": int(sample_index),
        "video_id": Path(meta).stem if isinstance(meta, str) else str(meta),
        "height": int(pers_height),
        "width": int(pers_width),
        "rotations": rotations_list,
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fixed camera trajectories for ODV360.")
    parser.add_argument(
        "--args_json",
        type=str,
        required=True,
        help="Path to args.json file from training (contains dataset arguments).",
    )
    parser.add_argument(
        "--odv_root_dir",
        type=str,
        default=None,
        help="Root directory of ODV360 dataset (overrides value in args.json).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./fixed_trajectory_rotation_fov90_2wp.json",
        help="Path to save the exported trajectory JSON (single or multiple samples).",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Index of the ODV360 validation sample to use (only when num_samples=1).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of validation samples to export trajectories for.",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Starting sample index when exporting multiple trajectories (used when num_samples>1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed used for trajectory generation (recorded into the JSON).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for trajectory generation (typically 'cuda').",
    )

    args = parser.parse_args()

    if not os.path.exists(args.args_json):
        raise FileNotFoundError(f"Args JSON not found: {args.args_json}")

    train_args = load_args_from_json(args.args_json)
    if args.odv_root_dir is not None:
        train_args.odv_root_dir = args.odv_root_dir

    if not hasattr(train_args, "odv_root_dir") or not train_args.odv_root_dir:
        raise ValueError(
            f"ODV360 root directory not specified. "
            f"Pass --odv_root_dir or make sure args_json contains 'odv_root_dir'."
        )
    if not os.path.isdir(train_args.odv_root_dir):
        raise FileNotFoundError(f"ODV360 root directory not found: {train_args.odv_root_dir}")
    
    perspective_params = {
        "fov_x": 90.0,
        "num_waypoints": 2,
        "simulate_camera_shake": True,
    }

    active_faces = (
        train_args.active_faces.split(",") if isinstance(getattr(train_args, "active_faces", None), str) else getattr(train_args, "active_faces", None)
    )

    odv_val = ODV360Dataset(
        root_dir=train_args.odv_root_dir,
        division="val/HR",
        num_frames=train_args.num_frames,
        height=train_args.height,
        width=train_args.width,
        cube_map_size=train_args.cube_map_size,
        window_length=train_args.window_length,
        active_faces=active_faces,
        perspective_params=perspective_params,
        use_random_fov=False,
        use_random_num_waypoints=False,
        trajectory_mode="rotation",
        keep_original_resolution=getattr(train_args, "keep_original_resolution", False),
    )

    total_samples = len(odv_val)
    print(f"Total samples: {total_samples}")

    # Decide which indices to export
    if args.num_samples <= 1:
        # Single-sample mode (backward compatible)
        if args.sample_index < 0 or args.sample_index >= total_samples:
            raise IndexError(f"sample_index {args.sample_index} out of range [0, {total_samples - 1}]")
        indices = [args.sample_index]
    else:
        start = max(0, args.start_idx)
        end = min(start + args.num_samples, total_samples)
        if end - start != args.num_samples:
            raise ValueError(
                f"Requested num_samples={args.num_samples} from start_idx={start}, "
                f"but only {end - start} samples available (total={total_samples})."
            )
        indices = list(range(start, end))

    print(f"Exporting trajectories for indices: {indices}")

    trajectories: list[dict] = []
    for i, sample_idx in enumerate(indices):
        # Derive per-sample seed to avoid accidental collisions
        per_sample_seed = int(args.seed)
        traj = _export_single_trajectory(
            odv_val=odv_val,
            sample_index=sample_idx,
            seed=per_sample_seed,
            train_args=train_args,
            device=args.device,
        )
        trajectories.append(traj)

    # Output format:
    # - If only one trajectory, write it as a single dict (backward compatible)
    # - If multiple, write as {"version": 1, "num_samples": N, "trajectories": [...]}
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or ".", exist_ok=True)
    if len(trajectories) == 1:
        to_save = trajectories[0]
    else:
        to_save = {
            "version": 1,
            "num_samples": len(trajectories),
            "trajectories": trajectories,
        }

    with open(args.output_path, "w") as f:
        json.dump(to_save, f, indent=2)

    print(f"Saved {len(trajectories)} fixed trajectory sample(s) to: {args.output_path}")


if __name__ == "__main__":
    main()
