from equilib import Equi2Pers, Equi2Cube, Pers2Equi
import torch
import numpy as np
from scipy.interpolate import CubicSpline
import math
from typing import Dict, List, Tuple
import torch.nn.functional as F
import torchvision

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(funcName)s - %(levelname)s - %(message)s")


class PanoVideoProcessor:
    @staticmethod
    def synthesize_perspective_video(video, fov_x=90.0, pers_height=480, pers_width=832, 
                                     num_waypoints=5, random_num_waypoints=False,
                                     pitch_range=(-30, 30), yaw_range=(-90, 90), roll_range=(-10, 10),
                                     simulate_camera_shake=True, shake_magnitude=0.2, use_diverse_trajectories=True,
                                     trajectory_config=None, fixed_rotations=None):
        """
        Synthesizes a perspective video with continuous moving camera rotations from a panoramic video.

        This simulates a real human-captured video by generating a smooth but diverse camera trajectory,
        with optional camera shake.

        Args:
            video (torch.Tensor): Input panoramic video (num_frames, C, H, W).
            fov_x (float): Horizontal field of view for the perspective camera.
            pers_height (int): Height of the perspective video.
            pers_width (int): Width of the perspective video.
            num_waypoints (int): Number of waypoints to define the camera path. More waypoints create a more complex path.
            pitch_range (tuple): Min and max pitch angles in degrees for waypoints.
            yaw_range (tuple): Min and max yaw angles in degrees for waypoints.
            roll_range (tuple): Min and max roll angles in degrees for waypoints.
            simulate_camera_shake (bool): If True, simulates camera shake.
            shake_magnitude (float): The magnitude of the camera shake for pitch, yaw, and roll in degrees.
            use_diverse_trajectories (bool): If True, uses spline interpolation for diverse trajectories.
                                             Otherwise, uses linear interpolation.
            trajectory_config (dict): Optional trajectory configuration dict with keys:
                - 'trajectory_type': 'fixed_point' or 'multi_waypoint'
                - 'num_waypoints': int (for multi_waypoint)
                - 'start_from_center': bool (for multi_waypoint)
                - 'require_high_coverage': bool (for multi_waypoint)
        Returns:
            pers_video (torch.Tensor): The synthesized perspective video.
            rotations (torch.Tensor): The rotation for each frame (num_frames, 3) for (roll, pitch, yaw) in degrees.
        """
        num_frames = video.shape[0]
        
        # Handle trajectory_config
        if trajectory_config is None:
            # Default behavior: use old logic
            if random_num_waypoints:
                num_waypoints = np.random.randint(2, num_waypoints)
            trajectory_config = {
                'trajectory_type': 'multi_waypoint',
                'num_waypoints': num_waypoints,
                'start_from_center': False,
                'require_high_coverage': True,
            }
        else:
            # Use trajectory_config, override num_waypoints if provided
            if trajectory_config.get('trajectory_type') == 'multi_waypoint':
                num_waypoints = trajectory_config.get('num_waypoints', num_waypoints)
        
        # Check if segment-based trajectory is requested (for rotation mode)
        segment_length = trajectory_config.get('segment_length', None)
        use_segmented_trajectory = (segment_length is not None and 
                                   segment_length > 0 and 
                                   num_frames > segment_length and
                                   trajectory_config.get('trajectory_type') != 'fixed_point')
        
        # Handle pre-computed per-frame rotations (e.g. from exported trajectory JSON)
        if fixed_rotations is not None:
            _fr = np.asarray(fixed_rotations, dtype=np.float64)
            if _fr.shape[0] < num_frames:
                raise ValueError(
                    f"fixed_rotations has {_fr.shape[0]} frames but video has {num_frames} frames"
                )
            roll_traj  = _fr[:num_frames, 0].copy()
            pitch_traj = _fr[:num_frames, 1].copy()
            yaw_traj   = _fr[:num_frames, 2].copy()
            # Pre-computed rotations already include any shake; skip random generation and shake.
            simulate_camera_shake = False

        # Handle fixed point trajectory
        elif trajectory_config.get('trajectory_type') == 'fixed_point':
            # Fixed point: all frames at [0, 0, 0]
            roll_traj = np.zeros(num_frames)
            pitch_traj = np.zeros(num_frames)
            yaw_traj = np.zeros(num_frames)
            # Camera shake will be applied later in the common section
        elif use_segmented_trajectory:
            # Segmented trajectory: generate independent sweep for each segment
            # This is used in rotation mode to ensure each 27-frame segment completes a full sweep
            roll_traj = np.zeros(num_frames)
            pitch_traj = np.zeros(num_frames)
            yaw_traj = np.zeros(num_frames)
            
            roll_min, roll_max = roll_range
            pitch_min, pitch_max = pitch_range
            yaw_min, yaw_max = yaw_range
            
            pitch_range_size = pitch_max - pitch_min
            yaw_range_size = yaw_max - yaw_min
            
            # Generate trajectory for each segment independently
            num_segments = (num_frames + segment_length - 1) // segment_length  # Ceiling division
            
            for seg_idx in range(num_segments):
                seg_start = seg_idx * segment_length
                seg_end = min((seg_idx + 1) * segment_length, num_frames)
                seg_length = seg_end - seg_start
                
                if seg_length < 2:
                    # Too short segment, use fixed point
                    continue
                
                # Generate waypoints for this segment
                seg_num_waypoints = min(num_waypoints, seg_length)
                if seg_num_waypoints > 1:
                    seg_waypoint_frames = np.linspace(0, seg_length - 1, seg_num_waypoints, dtype=int)
                else:
                    seg_waypoint_frames = np.array([0])
                
                # Generate waypoints for this segment
                seg_waypoint_roll = np.random.uniform(roll_min, roll_max, seg_num_waypoints)
                seg_waypoint_pitch = np.random.uniform(pitch_min, pitch_max, seg_num_waypoints)
                seg_waypoint_yaw = np.random.uniform(yaw_min, yaw_max, seg_num_waypoints)
                
                # Handle starting point for segment
                start_from_center = trajectory_config.get('start_from_center', False)
                if start_from_center and seg_idx == 0:
                    # Only first segment starts from center
                    seg_waypoint_roll[0] = 0.0
                    seg_waypoint_pitch[0] = 0.0
                    seg_waypoint_yaw[0] = 0.0
                else:
                    # Start from random distant point
                    distant_range = trajectory_config.get('distant_start_range', {
                        'pitch_range': (-40, 40),
                        'yaw_range': (-150, 150),
                        'roll_range': (-30, 30),
                    })
                    seg_waypoint_roll[0] = np.random.uniform(distant_range['roll_range'][0], distant_range['roll_range'][1])
                    seg_waypoint_pitch[0] = np.random.uniform(distant_range['pitch_range'][0], distant_range['pitch_range'][1])
                    seg_waypoint_yaw[0] = np.random.uniform(distant_range['yaw_range'][0], distant_range['yaw_range'][1])
                
                # Handle coverage requirement for segment
                require_high_coverage = trajectory_config.get('require_high_coverage', True)
                if require_high_coverage and seg_num_waypoints >= 2:
                    # Ensure 80% coverage for yaw and pitch within this segment
                    yaw_80_range = yaw_range_size * 0.8
                    pitch_80_range = pitch_range_size * 0.8
                    
                    if not (start_from_center and seg_idx == 0):
                        seg_waypoint_yaw[0] = yaw_min + (yaw_range_size - yaw_80_range) / 2
                        seg_waypoint_pitch[0] = pitch_min + (pitch_range_size - pitch_80_range) / 2
                    # Always adjust end point to ensure coverage
                    seg_waypoint_yaw[-1] = yaw_max - (yaw_range_size - yaw_80_range) / 2
                    seg_waypoint_pitch[-1] = pitch_max - (pitch_range_size - pitch_80_range) / 2
                
                # Sequential organization: either ascending or descending
                sort_prob = np.random.uniform(0, 1)
                if sort_prob < 0.5:
                    if sort_prob < 0.25:
                        # All ascending
                        seg_waypoint_roll = np.sort(seg_waypoint_roll)
                        seg_waypoint_pitch = np.sort(seg_waypoint_pitch)
                        seg_waypoint_yaw = np.sort(seg_waypoint_yaw)
                    else:
                        # Ascending roll, descending pitch and yaw
                        seg_waypoint_roll = np.sort(seg_waypoint_roll)
                        seg_waypoint_pitch = np.sort(seg_waypoint_pitch)[::-1]
                        seg_waypoint_yaw = np.sort(seg_waypoint_yaw)[::-1]
                else:
                    if sort_prob < 0.75:
                        # All descending
                        seg_waypoint_roll = np.sort(seg_waypoint_roll)[::-1]
                        seg_waypoint_pitch = np.sort(seg_waypoint_pitch)[::-1]
                        seg_waypoint_yaw = np.sort(seg_waypoint_yaw)[::-1]
                    else:
                        # Descending roll, ascending pitch and yaw
                        seg_waypoint_roll = np.sort(seg_waypoint_roll)[::-1]
                        seg_waypoint_pitch = np.sort(seg_waypoint_pitch)
                        seg_waypoint_yaw = np.sort(seg_waypoint_yaw)
                
                # Preserve starting point if it was set to center
                if start_from_center and seg_idx == 0:
                    seg_waypoint_roll[0] = 0.0
                    seg_waypoint_pitch[0] = 0.0
                    seg_waypoint_yaw[0] = 0.0
                
                # Interpolate to create smooth trajectory for this segment
                seg_frame_indices = np.arange(seg_length)
                if use_diverse_trajectories and seg_num_waypoints > 2:
                    # Use cubic spline for non-linear, diverse paths
                    spline_roll = CubicSpline(seg_waypoint_frames, seg_waypoint_roll, bc_type='natural')
                    spline_pitch = CubicSpline(seg_waypoint_frames, seg_waypoint_pitch, bc_type='natural')
                    spline_yaw = CubicSpline(seg_waypoint_frames, seg_waypoint_yaw, bc_type='natural')
                    seg_roll_traj = spline_roll(seg_frame_indices)
                    seg_pitch_traj = spline_pitch(seg_frame_indices)
                    seg_yaw_traj = spline_yaw(seg_frame_indices)
                else:
                    # Use linear interpolation for simpler paths
                    seg_roll_traj = np.interp(seg_frame_indices, seg_waypoint_frames, seg_waypoint_roll)
                    seg_pitch_traj = np.interp(seg_frame_indices, seg_waypoint_frames, seg_waypoint_pitch)
                    seg_yaw_traj = np.interp(seg_frame_indices, seg_waypoint_frames, seg_waypoint_yaw)
                
                # Assign to full trajectory
                roll_traj[seg_start:seg_end] = seg_roll_traj
                pitch_traj[seg_start:seg_end] = seg_pitch_traj
                yaw_traj[seg_start:seg_end] = seg_yaw_traj
        else:
            # Multi-waypoint trajectory (original logic for non-segmented)
            # 1. Generate waypoints for the camera path
            if num_waypoints > 1:
                waypoint_frames = np.linspace(0, num_frames - 1, num_waypoints, dtype=int)
            else:
                waypoint_frames = np.array([0])

            roll_min, roll_max = roll_range
            pitch_min, pitch_max = pitch_range
            yaw_min, yaw_max = yaw_range
            
            pitch_range_size = pitch_max - pitch_min
            yaw_range_size = yaw_max - yaw_min
            roll_range_size = roll_max - roll_min
            
            # Generate waypoints
            waypoint_roll = np.random.uniform(roll_min, roll_max, num_waypoints)
            waypoint_pitch = np.random.uniform(pitch_min, pitch_max, num_waypoints)
            waypoint_yaw = np.random.uniform(yaw_min, yaw_max, num_waypoints)
            
            # Handle starting point
            start_from_center = trajectory_config.get('start_from_center', False)
            if start_from_center:
                # Start from center [0, 0, 0]
                waypoint_roll[0] = 0.0
                waypoint_pitch[0] = 0.0
                waypoint_yaw[0] = 0.0
            else:
                # Start from random distant point
                # Use a wider range for distant starting points
                # Default distant range (can be overridden via trajectory_config)
                distant_range = trajectory_config.get('distant_start_range', {
                    'pitch_range': (-40, 40),
                    'yaw_range': (-150, 150),
                    'roll_range': (-30, 30),
                })
                waypoint_roll[0] = np.random.uniform(distant_range['roll_range'][0], distant_range['roll_range'][1])
                waypoint_pitch[0] = np.random.uniform(distant_range['pitch_range'][0], distant_range['pitch_range'][1])
                waypoint_yaw[0] = np.random.uniform(distant_range['yaw_range'][0], distant_range['yaw_range'][1])
            
            # Handle coverage requirement
            require_high_coverage = trajectory_config.get('require_high_coverage', True)
            if require_high_coverage and num_waypoints >= 2:
                # Ensure 80% coverage for yaw and pitch
                yaw_80_range = yaw_range_size * 0.8
                pitch_80_range = pitch_range_size * 0.8
                
                # Set extreme values to ensure 80% coverage
                # If start_from_center, keep starting point at [0,0,0] and adjust end point
                # Otherwise, adjust both start and end points
                if not start_from_center:
                    waypoint_yaw[0] = yaw_min + (yaw_range_size - yaw_80_range) / 2
                    waypoint_pitch[0] = pitch_min + (pitch_range_size - pitch_80_range) / 2
                # Always adjust end point to ensure coverage
                waypoint_yaw[-1] = yaw_max - (yaw_range_size - yaw_80_range) / 2
                waypoint_pitch[-1] = pitch_max - (pitch_range_size - pitch_80_range) / 2
            # else: no coverage requirement, waypoints can be anywhere
            
            # Sequential organization: either ascending or descending
            sort_prob = np.random.uniform(0, 1)
            if sort_prob < 0.5:
                if sort_prob < 0.25:
                    # All ascending
                    waypoint_roll = np.sort(waypoint_roll)
                    waypoint_pitch = np.sort(waypoint_pitch)
                    waypoint_yaw = np.sort(waypoint_yaw)
                else:
                    # Ascending roll, descending pitch and yaw
                    waypoint_roll = np.sort(waypoint_roll)
                    waypoint_pitch = np.sort(waypoint_pitch)[::-1]
                    waypoint_yaw = np.sort(waypoint_yaw)[::-1]
            else:
                if sort_prob < 0.75:
                    # All descending
                    waypoint_roll = np.sort(waypoint_roll)[::-1]
                    waypoint_pitch = np.sort(waypoint_pitch)[::-1]
                    waypoint_yaw = np.sort(waypoint_yaw)[::-1]
                else:
                    # Descending roll, ascending pitch and yaw
                    waypoint_roll = np.sort(waypoint_roll)[::-1]
                    waypoint_pitch = np.sort(waypoint_pitch)
                    waypoint_yaw = np.sort(waypoint_yaw)
            
            # Preserve starting point if it was set to center
            # (This is already handled above, but we ensure it's preserved after sorting)
            if start_from_center:
                waypoint_roll[0] = 0.0
                waypoint_pitch[0] = 0.0
                waypoint_yaw[0] = 0.0

            # 2. Interpolate to create a smooth trajectory
            frame_indices = np.arange(num_frames)
            if use_diverse_trajectories and num_waypoints > 2:
                # Use cubic spline for non-linear, diverse paths
                spline_roll = CubicSpline(waypoint_frames, waypoint_roll, bc_type='natural')
                spline_pitch = CubicSpline(waypoint_frames, waypoint_pitch, bc_type='natural')
                spline_yaw = CubicSpline(waypoint_frames, waypoint_yaw, bc_type='natural')
                roll_traj = spline_roll(frame_indices)
                pitch_traj = spline_pitch(frame_indices)
                yaw_traj = spline_yaw(frame_indices)
            else:
                # Use linear interpolation for simpler paths
                roll_traj = np.interp(frame_indices, waypoint_frames, waypoint_roll)
                pitch_traj = np.interp(frame_indices, waypoint_frames, waypoint_pitch)
                yaw_traj = np.interp(frame_indices, waypoint_frames, waypoint_yaw)

        # 3. Simulate camera shake if enabled (applies to both fixed_point and multi_waypoint)
        if simulate_camera_shake:
            # Generate smooth random noise
            shake_kernel_size = 3
            shake_kernel = np.ones(shake_kernel_size) / shake_kernel_size
            
            roll_shake = np.random.randn(num_frames) * shake_magnitude 
            pitch_shake = np.random.randn(num_frames) * shake_magnitude
            yaw_shake = np.random.randn(num_frames) * shake_magnitude

            roll_traj += np.convolve(roll_shake, shake_kernel, mode='same')
            pitch_traj += np.convolve(pitch_shake, shake_kernel, mode='same')
            yaw_traj += np.convolve(yaw_shake, shake_kernel, mode='same')

        # Convert trajectories to torch tensors
        # rotations_deg = torch.from_numpy(np.stack([roll_traj, pitch_traj, yaw_traj], axis=-1)).float()

        # 4. Extract perspective frames
        equi2pers = Equi2Pers(
            height=pers_height,
            width=pers_width,
            fov_x=fov_x,
            mode="bilinear",
        )
        original_dtype = video.dtype
        pers_frames = []
        pers_masks = []
        for i in range(num_frames):
            roll_deg, pitch_deg, yaw_deg = roll_traj[i], pitch_traj[i], yaw_traj[i]
            rots = {
                'roll': math.radians(roll_deg),
                'pitch': math.radians(pitch_deg),
                'yaw': math.radians(yaw_deg),
            }
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                pers_img, mask = equi2pers(equi=video[i], rots=rots)
                # blur the mask and apply threshold to remove high-frequency pattern artifacts
                mask = torchvision.transforms.functional.gaussian_blur(mask, kernel_size=31, sigma=1.0)
                mask = torch.where(mask > 0.05, torch.ones_like(mask), torch.zeros_like(mask))
            pers_frames.append(pers_img)
            pers_masks.append(mask)

        pers_video = torch.stack(pers_frames, axis=0).to(original_dtype)
        pers_mask = torch.stack(pers_masks, axis=0).to(original_dtype)
        return pers_video, pers_mask, np.stack([roll_traj, pitch_traj, yaw_traj], axis=-1)
    
    @staticmethod
    def remap_perspective_to_equirectangular(
        pers_video,
        rotations,
        fov_x,
        equi_height,
        equi_width,
        mode="bilinear"
    ):
        """
        Remap perspective video back to equirectangular video using estimated rotations.
        
        Args:
            pers_video: torch.Tensor of shape (T, C, H, W) - perspective video
            rotations: np.ndarray of shape (T, 3) with [roll, pitch, yaw] in degrees
            fov_x: Horizontal field of view in degrees
            equi_height: Height of output equirectangular video
            equi_width: Width of output equirectangular video
            mode: Interpolation mode ('bilinear' or 'nearest')
        
        Returns:
            equi_video: torch.Tensor of shape (T, C, equi_height, equi_width)
            equi_mask: torch.Tensor of shape (T, 1, equi_height, equi_width) - coverage mask
        """
        num_frames = pers_video.shape[0]
        original_dtype = pers_video.dtype
        
        # Create Pers2Equi converter
        pers2equi = Pers2Equi(
            height=equi_height,
            width=equi_width,
            mode=mode,
            clip_output=True
        )
        
        equi_frames = []
        equi_masks = []
        
        for i in range(num_frames):
            roll_deg, pitch_deg, yaw_deg = rotations[i]
            rots = {
                'roll': math.radians(roll_deg),
                'pitch': math.radians(pitch_deg),
                'yaw': math.radians(yaw_deg),
            }
            
            # Convert perspective frame to equirectangular
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                equi_frame = pers2equi(pers=pers_video[i], rots=rots, fov_x=fov_x)
                pers_mask_frame = torch.ones(
                    (1, pers_video.shape[2], pers_video.shape[3]),
                    device=pers_video.device,
                    dtype=pers_video.dtype
                )
                equi_mask_frame = pers2equi(pers=pers_mask_frame, rots=rots, fov_x=fov_x)
                # Threshold mask to binary
                equi_mask_frame = torch.where(equi_mask_frame > 0.1, torch.ones_like(equi_mask_frame), torch.zeros_like(equi_mask_frame))
            
            equi_frames.append(equi_frame)
            equi_masks.append(equi_mask_frame)
        
        equi_video = torch.stack(equi_frames, dim=0).to(original_dtype)
        equi_mask = torch.stack(equi_masks, dim=0).to(original_dtype)
        
        return equi_video, equi_mask
    
    @staticmethod
    def extract_cube_maps(equi_video, cube_map_size=512, quality_assurance_upsampling_factor=1) -> Dict[str, torch.Tensor]:
        """
        Extract cube maps from a panoramic video and return a dict of faces.
        Args:
            equi_video (torch.Tensor): [T, C, H, W]
            cube_map_size (int): width/height for each face
            quality_assurance_upsampling_factor (int): upsampling factor before downsampling back
        Returns:
            Dict[str, torch.Tensor]: {face: [T, C, cube_map_size, cube_map_size]} for faces in ('F','R','B','L','U','D')
        """
        assert quality_assurance_upsampling_factor >= 1, "quality_assurance_upsampling_factor must be >= 1"
        num_frames = equi_video.shape[0]

        equi2cube = Equi2Cube(
            w_face=int(cube_map_size * quality_assurance_upsampling_factor),
            cube_format='dict',
            mode='bilinear',
        )
        rots = [
            {
                'roll': 0,
                'pitch': 0,
                'yaw': 0,
            } for _ in range(num_frames)
        ]
        original_dtype = equi_video.dtype

        # Avoid hardcoding autocast to CUDA to keep CPU compatibility
        cube_list = equi2cube(equi=equi_video, rots=rots, backend='native')  # List[Dict[str, Tensor]]

        faces = ['F', 'R', 'B', 'L', 'U', 'D']
        cube_maps: Dict[str, list[torch.Tensor]] = {f: [] for f in faces}
        for t in range(num_frames):
            for f in faces:
                cube_maps[f].append(cube_list[t][f])  # [C, H, W]

        out: Dict[str, torch.Tensor] = {}
        for f in faces:
            face_video = torch.stack(cube_maps[f], dim=0)  # [T, C, H, W]
            if quality_assurance_upsampling_factor != 1:
                face_video = F.interpolate(face_video, size=(cube_map_size, cube_map_size), mode='bilinear', align_corners=False)
            out[f] = face_video.to(dtype=original_dtype)
        return out


class CubeMapPadder:
    def __init__(self, padding_width: int = 16, is_latent_mode: bool = False):
        self.padding_width = padding_width
        self.is_latent_mode = is_latent_mode
        if is_latent_mode:
            self.padding_width = padding_width // 16
        if padding_width == 0:
            print(f"[CubeMapPadder] padding_width is 0, no padding will be applied")
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

    def _extract_edge_strip(self, face: torch.Tensor, edge: str) -> torch.Tensor:
        p = self.padding_width
        if edge == 'top':
            return face[:, :, :p, :]
        if edge == 'bottom':
            return face[:, :, -p:, :]
        if edge == 'left':
            return face[:, :, :, :p]
        if edge == 'right':
            return face[:, :, :, -p:]
        raise ValueError(f"Unknown edge {edge}")

    def pad_face(self, face_tensors: Dict[str, torch.Tensor], face_id: str, original_height: int, original_width: int) -> torch.Tensor:
        """
        Pad a single face with pixels from adjacent faces.
        Args:
            face_tensors: {face: [T, C, H, W]}
            face_id: one of 'F','R','B','L','U','D'
        Returns:
            [T, C, H+2p, W+2p]
        """
        p = self.padding_width
        if p == 0:
            return face_tensors[face_id]
        face = face_tensors[face_id]
        # print(f"[pad_face] Face: {face_id}, Shape: {face.shape}")
        
        T, C, H, W = face.shape
        if H > original_height or W > original_width:
            print(f"[pad_face] Face: {face_id}, Shape: {face.shape}, Original shape: {original_height}, {original_width}")
            print(f"Already padded, return original face")
            return face

        padded = torch.zeros(T, C, H + 2 * p, W + 2 * p, device=face.device, dtype=face.dtype)
        padded[:, :, p:-p, p:-p] = face

        for edge, (adj_face, adj_edge, rot, flip) in self.face_map[face_id].items():
            adj_pixels = face_tensors[adj_face]
            strip = self._extract_edge_strip(adj_pixels, adj_edge)
            if rot != 0:
                # dims [T,C,H,W], rotate over spatial dims (H,W)
                k = (rot // 90) % 4
                strip = torch.rot90(strip, k=k, dims=[2, 3])
            if flip:
                # rarely used in current mapping
                if adj_edge in ['top', 'bottom']:
                    strip = torch.flip(strip, dims=[3])
                else:
                    strip = torch.flip(strip, dims=[2])
            # print(f"[pad_face] Strip shape: {strip.shape}, Padded shape: {padded.shape}")
            
            # place strip into padded tensor
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

        return padded

    def crop_padding(self, padded_face: torch.Tensor) -> torch.Tensor:
        p = self.padding_width
        if p == 0:
            return padded_face
        if padded_face.ndim == 5:
            return padded_face[:, :, :, p:-p, p:-p]
        elif padded_face.ndim == 4:   
            return padded_face[:, :, p:-p, p:-p]
        elif padded_face.ndim == 3:
            return padded_face[:, p:-p, p:-p]
        else:
            raise ValueError(f"Unsupported padded face shape: {padded_face.shape}")
    
    def extract_padding_strips(self, padded_face: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract padding strips from a padded face.
        Args:
            padded_face: [T, C, H+2p, W+2p]
        Returns:
            Dict with keys 'top', 'bottom', 'left', 'right', each containing [T, C, p, W] or [T, C, H, p]
        """
        p = self.padding_width
        if p == 0:
            return {}
        
        strips = {
            'top': padded_face[:, :, :p, p:-p],          # [T, C, p, W]
            'bottom': padded_face[:, :, -p:, p:-p],      # [T, C, p, W]
            'left': padded_face[:, :, p:-p, :p],         # [T, C, H, p]
            'right': padded_face[:, :, p:-p, -p:]        # [T, C, H, p]
        }
        return strips
    
    def create_alpha_blending_weight(self, strip_shape: Tuple[int, ...], adj_edge: str, device: torch.device) -> torch.Tensor:
        """
        Create alpha blending weight map for the adjacent face's edge region.
        Alpha represents the weight of the adjacent face's original values.
        The weight increases from 0 (at the boundary) to 1 (towards the face interior).
        
        This implements gradual blending where:
        - At the boundary (far from face center): alpha = 0 → use newly generated padding values
        - Towards face interior: alpha increases → gradually transition to original face values
        
        Args:
            strip_shape: Shape of the strip [T, C, H_strip, W_strip]
            adj_edge: The edge of the adjacent face ('top', 'bottom', 'left', 'right')
            device: Device to create tensor on
        
        Returns:
            Alpha weight tensor of shape [1, 1, H_strip, W_strip] where:
            - alpha = 0 means use 100% newly generated padding value
            - alpha = 1 means use 100% original adjacent face value
        """
        p = self.padding_width
        if p == 0:
            return torch.ones(1, 1, 1, 1, device=device)
        
        T, C, H_strip, W_strip = strip_shape
        
        if adj_edge in ['top', 'bottom']:
            # For top/bottom edges: shape is [T, C, p, W]
            # Alpha increases from boundary (0) towards interior (1)
            alpha_profile = torch.linspace(0, 1, p, device=device)  # [p]
            
            if adj_edge == 'top':
                # For top edge: first row is at boundary (alpha=0), last row towards interior (alpha=1)
                pass  # Already correct orientation
            else:  # bottom
                # For bottom edge: first row is at boundary (alpha=0), last row towards interior (alpha=1)  
                alpha_profile = alpha_profile.flip(0)
            
            # Broadcast to [1, 1, p, W_strip]
            alpha = alpha_profile.view(1, 1, p, 1).expand(1, 1, p, W_strip)
        else:  # left or right
            # For left/right edges: shape is [T, C, H, p]
            # Alpha increases from boundary (0) towards interior (1)
            alpha_profile = torch.linspace(0, 1, p, device=device)  # [p]
            
            if adj_edge == 'left':
                # For left edge: first col is at boundary (alpha=0), last col towards interior (alpha=1)
                pass  # Already correct orientation
            else:  # right
                # For right edge: first col is at boundary (alpha=0), last col towards interior (alpha=1)
                alpha_profile = alpha_profile.flip(0)
            
            # Broadcast to [1, 1, H_strip, p]
            alpha = alpha_profile.view(1, 1, 1, p).expand(1, 1, H_strip, p)
        
        return alpha
    
    def blend_padding_into_adjacent_faces(
        self, 
        current_face_id: str, 
        current_face_padded: torch.Tensor,
        adjacent_faces_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Blend padding regions from current generated face into adjacent faces.
        
        Args:
            current_face_id: ID of the current face ('F', 'R', 'B', 'L', 'U', 'D')
            current_face_padded: Generated face with padding [T, C, H+2p, W+2p]
            adjacent_faces_dict: Dict of adjacent face contents {face_id: [T, C, H, W]}
        
        Returns:
            Dict of updated adjacent faces {face_id: [T, C, H, W]}
        """
        p = self.padding_width
        if p == 0:
            return adjacent_faces_dict
        
        # Extract padding strips from current face
        strips = self.extract_padding_strips(current_face_padded)
        
        # Updated faces dict
        updated_faces = {}
        
        # For each edge of current face, find the adjacent face and blend
        for edge, (adj_face_id, adj_edge, rot, flip) in self.face_map[current_face_id].items():
            if adj_face_id not in adjacent_faces_dict:
                # Adjacent face not available in context, skip
                continue
            
            # Get the strip from current face's padding
            strip = strips[edge].clone()  # [T, C, p, W] or [T, C, H, p]
            
            # Get adjacent face content
            adj_face = adjacent_faces_dict[adj_face_id].clone()  # [T, C, H, W]
            
            # Reverse the rotation and flip to map strip back to adjacent face's coordinate
            # Note: We need to reverse the transformation
            if flip:
                if adj_edge in ['top', 'bottom']:
                    strip = torch.flip(strip, dims=[3])  # flip width
                else:
                    strip = torch.flip(strip, dims=[2])  # flip height
            
            if rot != 0:
                # Reverse rotation: if original was rot degrees, reverse is -rot
                k = (-rot // 90) % 4
                strip = torch.rot90(strip, k=k, dims=[2, 3])
            
            # Create alpha blending weight for the adjacent face edge
            # Alpha represents the weight of adjacent face's original values
            # alpha = 0 at boundary → use newly generated padding values
            # alpha = 1 towards interior → use original adjacent face values
            alpha = self.create_alpha_blending_weight(strip.shape, adj_edge, strip.device)
            
            # Extract the corresponding edge region from adjacent face
            if adj_edge == 'top':
                adj_region = adj_face[:, :, :p, :]  # [T, C, p, W]
                # Blend: blended = alpha * adj_region + (1-alpha) * strip
                # At boundary (alpha=0): use strip; towards interior (alpha=1): use adj_region
                blended = alpha * adj_region + (1 - alpha) * strip
                adj_face[:, :, :p, :] = blended
            elif adj_edge == 'bottom':
                adj_region = adj_face[:, :, -p:, :]  # [T, C, p, W]
                blended = alpha * adj_region + (1 - alpha) * strip
                adj_face[:, :, -p:, :] = blended
            elif adj_edge == 'left':
                adj_region = adj_face[:, :, :, :p]  # [T, C, H, p]
                blended = alpha * adj_region + (1 - alpha) * strip
                adj_face[:, :, :, :p] = blended
            elif adj_edge == 'right':
                adj_region = adj_face[:, :, :, -p:]  # [T, C, H, p]
                blended = alpha * adj_region + (1 - alpha) * strip
                adj_face[:, :, :, -p:] = blended
            
            updated_faces[adj_face_id] = adj_face
        
        return updated_faces


class GenerationOrderPlanner:
    def __init__(
        self,
        window_length: int = 9,
        cube_map_size: int = 512,
        active_faces: List[str] = None,
        is_latent_mode: bool = False,
        **kwargs,
    ):
        """
        Initialize the generation order planner.
        
        Args:
            window_length: Number of frames per generation window
            cube_map_size: Size of cube map faces
            active_faces: List of active face IDs
            is_latent_mode: Whether operating in latent space
        """
        self.is_latent_mode = is_latent_mode
        
        if is_latent_mode:
            logging.info(f"[GenerationOrderPlanner] Latent mode is enabled")
        else:
            logging.info(f"[GenerationOrderPlanner] Latent mode is disabled")
        self.window_length = window_length
        self.cube_map_size = cube_map_size
        
        # Set active faces
        if active_faces is None:
            self.active_faces = ['F', 'R', 'B', 'L', 'U', 'D']
        else:
            self.active_faces = active_faces
            
        # Full face adjacency mapping (even inactive faces might be referenced)
        self.face_adjacency: Dict[str, List[str]] = {
            'F': ['R', 'L', 'U', 'D'],
            'R': ['F', 'B', 'U', 'D'],
            'B': ['R', 'L', 'U', 'D'],
            'L': ['F', 'B', 'U', 'D'],
            'U': ['F', 'R', 'B', 'L'],
            'D': ['F', 'R', 'B', 'L'],
        }
        
        # Filter adjacency to only include active faces
        self.active_face_adjacency: Dict[str, List[str]] = {}
        for face in self.active_faces:
            self.active_face_adjacency[face] = [adj for adj in self.face_adjacency.get(face, []) if adj in self.active_faces]

    def _project_mask_to_faces(self, perspective_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Project an equirectangular coverage mask to cube faces.
        Args:
            perspective_mask: [T, H_equi, W_equi] in [0,1]
        Returns:
            Dict[str, torch.Tensor]: {face: [T, 1, S, S]} where S=cube_map_size
        """
        T, H, W = perspective_mask.shape
        equi2cube = Equi2Cube(w_face=self.cube_map_size, cube_format='dict', mode='nearest')
        equi_mask = perspective_mask.unsqueeze(1)  # [T, 1, H, W]
        cube_list = equi2cube(equi=equi_mask, rots=[{'roll': 0, 'pitch': 0, 'yaw': 0}] * T, backend='native')
        faces = self.active_faces
        out: Dict[str, List[torch.Tensor]] = {f: [] for f in faces}
        for t in range(T):
            for f in faces:
                out[f].append(cube_list[t][f])  # [1, S, S]
        return {f: torch.stack(out[f], dim=0) for f in faces}

    def compute_coverage(self, perspective_mask: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Compute per-face coverage per frame.
        Returns dict of face -> [T] coverage fraction.
        """
        face_masks = self._project_mask_to_faces(perspective_mask)
        coverage: Dict[str, np.ndarray] = {}
        for f, m in face_masks.items():
            cov = m.mean(dim=[1, 2, 3]).cpu().numpy()  # [T]
            coverage[f] = cov
        return coverage

    def plan_order(self, perspective_mask: torch.Tensor, num_frames: int) -> List[Tuple[str, int, int]]:
        """
        Plan generation order in a causal way: iterate windows in ascending time; 
        within a window, schedule faces by descending coverage to maximize input condition usage.
        """
        # Convert perspective mask to cube face masks
        face_masks = self._project_mask_to_faces(perspective_mask)
        # Use the causal planning method
        if self.is_latent_mode:
            return self.plan_latent_order_causal_from_face_masks(face_masks, num_frames)
        else:
            return self.plan_order_causal_from_face_masks(face_masks, num_frames)
    
    def plan_order_from_cubemap_masks(self, mask_cubemap: Dict[str, torch.Tensor], num_frames: int) -> List[Tuple[str, int, int]]:
        """
        Plan generation order directly from cube map masks in a causal way.
        
        Args:
            mask_cubemap: Dict of {face: [T, C|1, H, W]} where values are 0-1 masks
            num_frames: Total number of frames
            
        Returns:
            List of (face, start, end) tuples in generation order
        """
        if self.is_latent_mode:
            return self.plan_latent_order_causal_from_face_masks(mask_cubemap, num_frames)
        else:
            return self.plan_order_causal_from_face_masks(mask_cubemap, num_frames)


    def compute_coverage_from_face_masks(self, face_masks: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """
        Compute per-face coverage per frame directly from a face mask dict.
        face_masks: {face: [T, C|1, H, W] or [T, H, W]}
        Returns dict of face -> [T] ndarray.
        """
        coverage: Dict[str, np.ndarray] = {}
        for f, m in face_masks.items():
            # print(f"[compute_coverage_from_face_masks] Face: {f}, Shape: {m.shape}")
            if m.ndim == 5:
                cov = m[0].mean(dim=[1, 2, 3]).float().cpu().numpy()
            elif m.ndim == 4:
                cov = m.mean(dim=[1, 2, 3]).float().cpu().numpy()
            elif m.ndim == 3:
                cov = m.mean(dim=[1, 2]).float().cpu().numpy()
            else:
                raise ValueError(f"Unsupported face mask shape for face {f}: {tuple(m.shape)}")
            coverage[f] = cov
        return coverage

    def plan_order_causal_from_face_masks(self, face_masks: Dict[str, torch.Tensor], num_frames: int) -> List[Tuple[str, int, int]]:
        """
        Time-causal planning: iterate windows in ascending time; within a window,
        schedule faces based on the configured order mode:
        - "planned": sort by descending coverage (mask mean)
        - "fixed": use fixed order from active_faces
        - "random": shuffle face order randomly
        Returns list of (face, start, end).
        """
        coverage = self.compute_coverage_from_face_masks(face_masks)  # face -> [T]
        order: List[Tuple[str, int, int]] = []
        for start in range(0, num_frames, self.window_length):
            end = min(start + self.window_length, num_frames)
            # Build (face, score) for this window
            scored: List[Tuple[str, float]] = []
            for face in self.active_faces:
                window_cov = float(coverage[face][start:end].mean()) if end > start else 0.0
                scored.append((face, window_cov))
            
            scored.sort(key=lambda x: x[1], reverse=True)
            
            for face, _ in scored:
                order.append((face, start, end))
        return order

    def plan_latent_order_causal_from_face_masks(self, face_masks: Dict[str, torch.Tensor], num_frames: int) -> List[Tuple[str, int, int]]:
        """
        Time-causal planning: iterate windows in ascending time; within a window,
        schedule faces based on the configured order mode:
        - "planned": sort by descending coverage (mask mean)
        - "fixed": use fixed order from active_faces
        - "random": shuffle face order randomly
        Returns list of (face, start, end).
        """
        coverage = self.compute_coverage_from_face_masks(face_masks)  # face -> [T]
        order: List[Tuple[str, int, int]] = []
        start = 0
        end = start + self.window_length
        while end <= num_frames:
            scored: List[Tuple[str, float]] = []
            for face in self.active_faces:
                window_cov = float(coverage[face][start:end].mean()) if end > start else 0.0
                scored.append((face, window_cov))
            
            scored.sort(key=lambda x: x[1], reverse=True)
            
            latent_start = self._frame_idx_to_latent_idx(start)
            latent_end = self._frame_idx_to_latent_idx(end)
            for face, _ in scored:
                order.append((face, latent_start, latent_end))

            start = end
            end = start + self.window_length - 1 # Later frames does not contain the first frame
        return order

    def _frame_idx_to_latent_idx(self, frame_idx: int) -> int:
        """
        Convert frame index to VAE latent index. The first frame occupies the first latent, and later every 4 frames occupies 1 latent.
        Mapping example:
            frame_idx   latent_idx
            0           0
            1           1
            2           1
            3           1
            4           1
            5           2
            6           2
            7           2
            8           2
        """
        if frame_idx == 0:
            return 0
        else:
            return 1 + (frame_idx - 1) // 4