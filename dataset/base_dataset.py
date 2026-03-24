
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import random
from typing import List
import logging
import torch
import numpy as np

from models.panorama import PanoVideoProcessor
logging.basicConfig(level=logging.WARNING)

# Trajectory generation strategy configuration
# This dict controls the diversity of trajectory types during training
TRAJECTORY_GENERATION_STRATEGY = {
    # Probability distribution for different trajectory types
    # Format: {trajectory_type: probability_weight}
    'type_weights': {
        'fixed_point': 0.1,  # Fixed point [0,0,0] throughout
        'multi_waypoint': 0.9,  # Multi-waypoint trajectories (2, 3, 4, 5 points)
    },
    
    # Multi-waypoint trajectory configuration
    'multi_waypoint': {
        # Number of waypoints and their probabilities
        'num_waypoints_weights': {
            2: 0.25,  # 25% chance for 2 waypoints
            3: 0.25,  # 25% chance for 3 waypoints
            4: 0.25,  # 25% chance for 4 waypoints
            5: 0.25,  # 25% chance for 5 waypoints
        },
        
        # Starting point configuration
        # 50% from center [0,0,0], 50% from random distant point
        'start_from_center_prob': 0.5,
        
        # Coverage requirement
        # 50% require high coverage (80%+), 50% don't require full coverage
        'require_high_coverage_prob': 0.5,
        
        # Range for random distant starting points (degrees)
        'distant_start_range': {
            'pitch_range': (-40, 40),
            'yaw_range': (-150, 150),
            'roll_range': (-30, 30),
        },
    },
    
    # Default angle ranges (can be overridden by perspective_params)
    'default_ranges': {
        'pitch_range': (-45, 45),
        'yaw_range': (-120, 120),
        'roll_range': (-45, 45),
    },
}

class Base360VideoDataset(Dataset, ABC):
    def __init__(self,
                 num_frames=16,
                 height=512,
                 width=1024,
                 stride=1,
                 use_random_stride=False,
                 *,
                 cube_map_size=512,
                 window_length=9,
                 perspective_params: dict | None = None,
                 active_faces: List[str] = None,
                 use_random_fov: bool = False,
                 use_random_num_waypoints: bool = False,
                 trajectory_mode: str = "diverse",  # "diverse", "rotation", "fixed"
                 ):
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.stride = stride
        self.use_random_stride = use_random_stride
        # Panorama-related
        self.cube_map_size = cube_map_size
        self.window_length = window_length
        self.active_faces = active_faces if active_faces is not None else ['F', 'R', 'B', 'L', 'U', 'D']
        default_perspective = {
            'fov_x': 120,
            'num_waypoints': 2,
            'pitch_range': (-45, 45),
            'yaw_range': (-120, 120),
            'roll_range': (-45, 45),
            'simulate_camera_shake': True,
            'shake_magnitude': 0.2,
        }
        self.perspective_params = {**default_perspective, **(perspective_params or {})}
        self.use_random_fov = use_random_fov
        self.use_random_num_waypoints = use_random_num_waypoints
        # Trajectory mode: "diverse" (mixed strategies), "rotation" (simple sweep), "fixed" (static camera)
        # if a boolean is passed, map True→"diverse", False→"rotation"
        if isinstance(trajectory_mode, bool):
            trajectory_mode = "diverse" if trajectory_mode else "rotation"
        if trajectory_mode not in ("diverse", "rotation", "fixed"):
            trajectory_mode = "diverse"
        self.trajectory_mode = trajectory_mode

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def _load_video_tensor(self, idx, stride):
        """
        Loads a video tensor of shape (T, C, H, W).
        The implementation should handle frame sampling, resizing, and padding.
        """
        pass
    
    @abstractmethod
    def get_metadata(self, idx):
        """
        Returns a dictionary with 'caption' and 'video_path'.
        """
        pass

    def _select_trajectory_type(self):
        """
        Select trajectory type based on TRAJECTORY_GENERATION_STRATEGY configuration.
        Returns a dict with trajectory configuration.
        """
        strategy = TRAJECTORY_GENERATION_STRATEGY
        type_weights = strategy['type_weights']
        
        # Normalize weights to probabilities
        total_weight = sum(type_weights.values())
        rand_val = random.random() * total_weight
        
        cumulative = 0
        for traj_type, weight in type_weights.items():
            cumulative += weight
            if rand_val <= cumulative:
                selected_type = traj_type
                break
        else:
            # Fallback to multi_waypoint if something goes wrong
            selected_type = 'multi_waypoint'
        
        if selected_type == 'fixed_point':
            # Fixed point trajectory: always [0, 0, 0]
            return {
                'trajectory_type': 'fixed_point',
                'num_waypoints': 1,
                'start_from_center': True,
                'require_high_coverage': False,
            }
        else:
            # Multi-waypoint trajectory
            multi_config = strategy['multi_waypoint']
            
            # Select number of waypoints
            num_waypoints_weights = multi_config['num_waypoints_weights']
            total_wp_weight = sum(num_waypoints_weights.values())
            rand_wp = random.random() * total_wp_weight
            cumulative_wp = 0
            selected_num_waypoints = 2  # default
            for num_wp, weight in num_waypoints_weights.items():
                cumulative_wp += weight
                if rand_wp <= cumulative_wp:
                    selected_num_waypoints = num_wp
                    break
            
            # Select starting point
            start_from_center = random.random() < multi_config['start_from_center_prob']
            
            # Select coverage requirement
            require_high_coverage = random.random() < multi_config['require_high_coverage_prob']
            
            # Include distant_start_range in config for use in panorama.py
            config = {
                'trajectory_type': 'multi_waypoint',
                'num_waypoints': selected_num_waypoints,
                'start_from_center': start_from_center,
                'require_high_coverage': require_high_coverage,
                'distant_start_range': multi_config['distant_start_range'],
            }
            return config

    def __getitem__(self, idx):
        # Load base equirectangular video and metadata (CPU only)
        stride = random.randint(1, self.stride) if self.use_random_stride and self.stride > 1 else self.stride
        equi_video = self._load_video_tensor(idx, stride)
        metadata = self.get_metadata(idx)

        # Prepare perspective params for this sample
        perspective_params = self.perspective_params.copy()
        if self.use_random_fov:
            perspective_params['fov_x'] = random.randint(60, 120)
        
        # Select trajectory configuration based on trajectory_mode
        mode = self.trajectory_mode
        if mode == "diverse":
            # Diverse mode: use mixed trajectory strategies
            trajectory_config = self._select_trajectory_type()
            perspective_params['trajectory_config'] = trajectory_config
            # Optional: randomize num_waypoints if enabled
            if self.use_random_num_waypoints:
                perspective_params['num_waypoints'] = random.randint(2, 5)
                if trajectory_config['trajectory_type'] == 'multi_waypoint':
                    trajectory_config['num_waypoints'] = perspective_params['num_waypoints']
        elif mode == "rotation":
            # Rotation mode: simple multi-waypoint trajectory that sweeps the scene
            if self.use_random_num_waypoints:
                perspective_params['num_waypoints'] = random.randint(2, 5)
            num_waypoints = perspective_params.get('num_waypoints', 2)
            trajectory_config = {
                'trajectory_type': 'multi_waypoint',
                'num_waypoints': num_waypoints,
                # Start from center for stability (only for first segment)
                'start_from_center': False,
                # Ensure high coverage (sweep across pitch/yaw ranges) within each segment
                'require_high_coverage': True,
                # Use the default distant range config for completeness
                'distant_start_range': TRAJECTORY_GENERATION_STRATEGY['multi_waypoint']['distant_start_range'],
                'segment_length': 27,
            }
            perspective_params['trajectory_config'] = trajectory_config
        else:
            # Fallback: behave like simple rotation mode without explicit coverage enforcement
            if self.use_random_num_waypoints:
                perspective_params['num_waypoints'] = random.randint(2, 5)
        return {
            'equi_video': equi_video,
            'perspective_params': perspective_params,
            'height': self.height,
            'width': self.width,
            'cube_map_size': self.cube_map_size,
            'window_length': self.window_length,
            'active_faces': self.active_faces,
            'caption': metadata['caption'],
            'video_path': metadata['video_path'],
            'face_captions': metadata.get('face_captions', None),
            'idx': idx,
        }

# Backward compatibility alias
BasePanoramaEquirectDataset = Base360VideoDataset


def _chunked_panorama_operation(video_tensor, operation_func, max_frames=54, **operation_kwargs):
    """
    Helper function to perform panorama operations on video in chunks to avoid OOM.
    
    Args:
        video_tensor: torch.Tensor of shape (T, C, H, W) - input video
        operation_func: Function that processes video tensor and returns result
        max_frames: Maximum number of frames to process at once (default: 54)
        **operation_kwargs: Additional keyword arguments to pass to operation_func
    
    Returns:
        Result from operation_func, concatenated if chunked
    """
    num_frames = video_tensor.shape[0]
    
    # If video is short enough, process directly
    if num_frames <= max_frames:
        return operation_func(video_tensor, **operation_kwargs)
    
    # Otherwise, process in chunks
    logging.info(f"[ChunkedOperation] Processing {num_frames} frames in chunks (max {max_frames} frames per chunk)")
    chunk_results = []
    
    for start_idx in range(0, num_frames, max_frames):
        end_idx = min(start_idx + max_frames, num_frames)
        chunk = video_tensor[start_idx:end_idx]
        logging.debug(f"[ChunkedOperation] Processing chunk [{start_idx}:{end_idx}] ({end_idx - start_idx} frames)")
        
        chunk_result = operation_func(chunk, **operation_kwargs)
        chunk_results.append(chunk_result)
    
    # Concatenate results along time dimension
    if isinstance(chunk_results[0], torch.Tensor):
        # Simple tensor concatenation
        return torch.cat(chunk_results, dim=0)
    elif isinstance(chunk_results[0], dict):
        # Dictionary of tensors (e.g., cubemap faces)
        result_dict = {}
        for key in chunk_results[0].keys():
            result_dict[key] = torch.cat([chunk[key] for chunk in chunk_results], dim=0)
        return result_dict
    elif isinstance(chunk_results[0], tuple):
        # Tuple of results (e.g., (video, mask, rotations))
        num_outputs = len(chunk_results[0])
        result_tuple = []
        for i in range(num_outputs):
            if isinstance(chunk_results[0][i], torch.Tensor):
                result_tuple.append(torch.cat([chunk[i] for chunk in chunk_results], dim=0))
            elif isinstance(chunk_results[0][i], np.ndarray):
                result_tuple.append(np.concatenate([chunk[i] for chunk in chunk_results], axis=0))
            else:
                result_tuple.append(chunk_results[0][i])  # For non-tensor/array types, just use first chunk
        return tuple(result_tuple)
    else:
        # Fallback: return first chunk result (shouldn't happen in practice)
        logging.warning(f"[ChunkedOperation] Unexpected result type: {type(chunk_results[0])}, returning first chunk")
        return chunk_results[0]


class CudaPreprocessor:
    """
    Preprocessor to handle CUDA-based operations that were previously in dataset __getitem__.
    This allows DataLoader to work with num_workers > 0.
    """
    def __init__(self, device="cuda", is_latent_mode=False):
        self.device = device
        self.planner_cache = {}
        self.is_latent_mode = is_latent_mode

    
    def get_planner(self, window_length, cube_map_size, active_faces):
        """Get or create a cached planner."""
        from models.panorama import GenerationOrderPlanner
        active_faces_tuple = tuple(active_faces)
        key = (window_length, cube_map_size, active_faces_tuple)
        if key not in self.planner_cache:
            self.planner_cache[key] = GenerationOrderPlanner(
                window_length=window_length,
                cube_map_size=cube_map_size,
                active_faces=list(active_faces),
                is_latent_mode=self.is_latent_mode
            )
        return self.planner_cache[key]
    
    def preprocess(self, data):
        """
        Apply CUDA preprocessing to dataset sample.
        
        Args:
            data: Dictionary from dataset __getitem__ (CPU tensors)
            
        Returns:
            Dictionary with preprocessed data (CUDA tensors)
        """
        
        # Move equi_video to CUDA
        equi_video = data['equi_video'].to(self.device)
        
        # Get perspective params
        perspective_params = data['perspective_params']
        height = data['height']
        width = data['width']
        cube_map_size = data['cube_map_size']
        window_length = data['window_length']
        active_faces = data['active_faces']
        
        
        # Get processor and synthesize perspective video
        # Use chunked processing if video is longer than 54 frames to avoid OOM
        num_frames = equi_video.shape[0]
        if num_frames > 54:
            logging.info(f"[CudaPreprocessor] Video has {num_frames} frames, using chunked processing for panorama operations")
        
        # Synthesize perspective video with chunked processing if needed
        def _synthesize_perspective(video_chunk):
            return PanoVideoProcessor.synthesize_perspective_video(
                video_chunk,
                pers_height=height,
                pers_width=width,
                fov_x=perspective_params['fov_x'],
                num_waypoints=perspective_params.get('num_waypoints', 2),
                pitch_range=perspective_params['pitch_range'],
                yaw_range=perspective_params['yaw_range'],
                roll_range=perspective_params['roll_range'],
                simulate_camera_shake=perspective_params.get('simulate_camera_shake', True),
                shake_magnitude=perspective_params.get('shake_magnitude', 0.2),
                use_diverse_trajectories=perspective_params.get('use_diverse_trajectories', True),
                trajectory_config=perspective_params.get('trajectory_config', None),
                fixed_rotations=perspective_params.get('fixed_rotations', None),
            )
        
        pers_video, pers_mask, rotations = _chunked_panorama_operation(
            equi_video,
            _synthesize_perspective,
            max_frames=54
        )
        pers_mask = pers_mask.to(self.device)
        
        # Extract cube maps with full precision to avoid quality degradation
        # Save original dtypes and convert to float32 for high-quality cubemap extraction
        equi_video_dtype = equi_video.dtype
        pers_mask_dtype = pers_mask.dtype
        
        # Convert to float32 for cubemap extraction to maintain quality
        # This is especially important when preprocess is called within autocast context
        equi_video_f32 = equi_video.to(torch.float32) if equi_video.dtype != torch.float32 else equi_video
        pers_mask_f32 = pers_mask.to(torch.float32) if pers_mask.dtype != torch.float32 else pers_mask
        
        # Extract cube maps with chunked processing if needed
        def _extract_cubemap(video_chunk):
            return PanoVideoProcessor.extract_cube_maps(video_chunk, cube_map_size=cube_map_size)
        
        gt_cubemap = _chunked_panorama_operation(
            equi_video_f32,
            _extract_cubemap,
            max_frames=54
        )
        mask_cubemap = _chunked_panorama_operation(
            pers_mask_f32,
            _extract_cubemap,
            max_frames=54
        )
        
        # Convert cubemaps back to original dtype if needed
        if equi_video_dtype != torch.float32:
            gt_cubemap = {face: face_video.to(equi_video_dtype) for face, face_video in gt_cubemap.items()}
        if pers_mask_dtype != torch.float32:
            mask_cubemap = {face: face_video.to(pers_mask_dtype) for face, face_video in mask_cubemap.items()}
        
        # Conditional cubemap = GT masked by mask cubemap
        cond_cubemap = {}
        for face in gt_cubemap:
            cond_cubemap[face] = gt_cubemap[face] * mask_cubemap[face]
        
        # Plan generation order
        planner = self.get_planner(window_length, cube_map_size, active_faces)
        order = planner.plan_order_from_cubemap_masks(mask_cubemap, equi_video.shape[0])
        
        # Return preprocessed data
        return {
            'equi_video': equi_video,
            'perspective_video': pers_video,
            'mask': pers_mask,
            'rotations': rotations,
            'cond_cubemap': cond_cubemap,
            'gt_cubemap': gt_cubemap,
            'mask_cubemap': mask_cubemap,
            'planned_order': order,
            'caption': data['caption'],
            'video_path': data['video_path'],
            'face_captions': data.get('face_captions', None),
            'idx': data['idx'],
        }
