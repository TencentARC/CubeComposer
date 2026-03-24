import os
import random
import torch
import torchvision.transforms as T
from PIL import Image
from .base_dataset import Base360VideoDataset
import json


class ODV360Dataset(Base360VideoDataset):
    def __init__(self, root_dir, division='train/HR', num_frames=16, height=256, width=512, stride=1, use_random_stride=False, sample_idx_range=None,
                 cube_map_size=512, window_length=8, active_faces=None, perspective_params=None,
                 use_random_fov=False, use_random_num_waypoints=False, trajectory_mode: str = "diverse", keep_original_resolution=False):
        super().__init__(num_frames=num_frames, height=height, width=width, stride=stride, use_random_stride=use_random_stride,
                         cube_map_size=cube_map_size, window_length=window_length, active_faces=active_faces,
                         perspective_params=perspective_params,
                         use_random_fov=use_random_fov, use_random_num_waypoints=use_random_num_waypoints,
                         trajectory_mode=trajectory_mode)
        self.root_dir = root_dir
        self.division = division
        self.division_path = os.path.join(self.root_dir, self.division)
        self.video_folders = [os.path.join(self.division_path, d) for d in os.listdir(self.division_path) if os.path.isdir(os.path.join(self.division_path, d))]
        # remove the "caption" folder from the video_folders
        self.video_folders = [d for d in self.video_folders if not d.endswith('caption')]
        if sample_idx_range:
            self.video_folders = self.video_folders[sample_idx_range[0]:sample_idx_range[1]]
        
        # If keep_original_resolution=True, don't resize (use original image resolution)
        # This preserves maximum quality but may require more memory
        self.keep_original_resolution = keep_original_resolution
        if keep_original_resolution:
            self.transform = T.Compose([
                T.ToTensor(),
            ])
            print(f"[ODV360Dataset] Using original resolution (no resize)")
        else:
            self.transform = T.Compose([
                T.Resize((self.height, self.width)),
                T.ToTensor(),
            ])
            print(f"[ODV360Dataset] Resizing to {self.height}x{self.width}")

    def __len__(self):
        return len(self.video_folders)

    def _load_video_tensor(self, idx, stride):
        video_folder = self.video_folders[idx]
        frame_files = sorted([os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.png')])
        total_frames = len(frame_files)

        if total_frames == 0:
            frames = [torch.zeros(3, self.height, self.width) for _ in range(self.num_frames)]
            return torch.stack(frames, dim=0)

        frames = []
        max_start = max(0, total_frames - (self.num_frames - 1) * stride)
        start_idx = random.randint(0, max_start) if max_start > 0 else 0

        for i in range(self.num_frames):
            frame_idx = start_idx + i * stride
            if frame_idx >= total_frames:
                break
            
            frame_path = frame_files[frame_idx]
            try:
                frame = Image.open(frame_path).convert("RGB")
                frame = self.transform(frame)
                frames.append(frame)
            except Exception as e:
                print(f"Warning: could not load frame {frame_path}: {e}")
                break

        if len(frames) < self.num_frames:
            if not frames:
                frames = [torch.zeros(3, self.height, self.width) for _ in range(self.num_frames)]
            else:
                last_frame = frames[-1]
                frames.extend([last_frame.clone() for _ in range(self.num_frames - len(frames))])
        
        return torch.stack(frames, dim=0)

    def get_metadata(self, idx):
        video_folder = self.video_folders[idx]
        video_name = os.path.basename(video_folder)
        caption_path = os.path.join(self.division_path, "caption", f'{video_name}_captions.json')
        with open(caption_path, 'r') as f:
            captions = json.load(f)
        caption_global = captions['global']
        face_captions = {
            'F': captions['face_f'],
            'R': captions['face_r'],
            'B': captions['face_b'],
            'L': captions['face_l'],
            'U': captions['face_u'],
            'D': captions['face_d'],
        }
        return {
            'caption': caption_global,
            'video_path': video_folder,
            'id': video_name,
            'face_captions': face_captions,
            'caption_path': caption_path,
        }
