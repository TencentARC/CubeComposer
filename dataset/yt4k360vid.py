import os
import json
import random
import torch
import torchvision.transforms as T
import cv2
from .base_dataset import Base360VideoDataset


class YT4K360VidDataset(Base360VideoDataset):
    def __init__(self,
                 caption_dir,
                 video_dir,
                 num_frames=16,
                 height=256,
                 width=512,
                 stride=1,
                 format='mp4',
                 use_random_stride=False,
                 sample_idx_range=None,
                 cube_map_size=512,
                 window_length=8,
                 active_faces=None,
                 perspective_params=None,
                 use_random_fov=False, use_random_num_waypoints=False,
                 trajectory_mode: str = "diverse",
                 split_list_path=None,
                 keep_original_resolution=False):
        super().__init__(num_frames=num_frames, height=height, width=width, stride=stride, use_random_stride=use_random_stride,
                         cube_map_size=cube_map_size, window_length=window_length, active_faces=active_faces,
                         perspective_params=perspective_params,
                         use_random_fov=use_random_fov, use_random_num_waypoints=use_random_num_waypoints,
                         trajectory_mode=trajectory_mode)
        self.caption_dir = caption_dir
        self.video_dir = video_dir
        self.format = format
        self.keep_original_resolution = keep_original_resolution

        if not os.path.isdir(self.caption_dir):
            raise FileNotFoundError(f"Caption directory not found: {self.caption_dir}")
        if not os.path.isdir(self.video_dir):
            raise FileNotFoundError(f"Video directory not found: {self.video_dir}")

        # Optionally restrict to a split list
        allowed_captions = None
        if split_list_path is not None:
            with open(split_list_path, 'r') as f:
                allowed_captions = set([line.strip() for line in f if line.strip()])

        all_caption_files = [f for f in os.listdir(self.caption_dir) if f.endswith('_captions.json')]
        all_caption_files.sort()
        if allowed_captions is not None:
            all_caption_files = [f for f in all_caption_files if f in allowed_captions]

        entries = []
        for cap_filename in all_caption_files:
            base = cap_filename[:-len('_captions.json')]
            video_filename = f"{base}.{self.format}"
            cap_path = os.path.join(self.caption_dir, cap_filename)
            vid_path = os.path.join(self.video_dir, video_filename)
            if os.path.isfile(vid_path):
                entries.append({
                    'caption_path': cap_path,
                    'video_path': vid_path,
                    'id': base,
                })
            else:
                print(f"Warning: video file not found for caption {cap_filename}: expected {vid_path}")

        if sample_idx_range:
            start, end = sample_idx_range
            entries = entries[start:end]

        self.entries = entries

        # If keep_original_resolution=True, don't resize (use original video resolution)
        # This preserves maximum quality but may require more memory
        if keep_original_resolution:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.ToTensor(),
            ])
            print(f"[Youtube360Video] Using original resolution (no resize)")
        else:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((self.height, self.width)),
                T.ToTensor(),
            ])
            print(f"[Youtube360Video] Resizing to {self.height}x{self.width}")

    def __len__(self):
        return len(self.entries)

    def _load_video_tensor(self, idx, stride):
        entry = self.entries[idx]
        video_path = entry['video_path']

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: failed to open video {video_path}")
            frames = [torch.zeros(3, self.height, self.width) for _ in range(self.num_frames)]
            return torch.stack(frames, dim=0)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        max_start = max(0, total_frames - (self.num_frames - 1) * stride)
        start_idx = random.randint(0, max_start) if max_start > 0 else 0

        for i in range(self.num_frames):
            frame_idx = start_idx + i * stride
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.transform(frame)
            frames.append(frame)

        cap.release()

        if len(frames) < self.num_frames:
            if len(frames) == 0:
                frames = [torch.zeros(3, self.height, self.width) for _ in range(self.num_frames)]
            else:
                last = frames[-1]
                frames += [last.clone() for _ in range(self.num_frames - len(frames))]

        return torch.stack(frames, dim=0)

    def get_metadata(self, idx):
        entry = self.entries[idx]
        caption_path = entry['caption_path']
        caption = ""
        face_captions = {}
        try:
            with open(caption_path, 'r') as f:
                data = json.load(f)
            # Use the global caption as main caption
            caption = data.get('global', '') or ''
            # Store face captions for future use
            face_captions = {
                'F': data.get('face_f', ''),
                'R': data.get('face_r', ''),
                'B': data.get('face_b', ''),
                'L': data.get('face_l', ''),
                'U': data.get('face_u', ''),
                'D': data.get('face_d', ''),
            }
        except Exception as e:
            print(f"Warning: failed to read caption {caption_path}: {e}")

        return {
            'caption': caption,
            'video_path': entry['video_path'],
            'id': entry['id'],
            'face_captions': face_captions,
            'caption_path': caption_path,
        }
