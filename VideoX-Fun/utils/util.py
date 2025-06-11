import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from torchvision import transforms

class VideoPairDataset(Dataset):
    def __init__(self, root_benchmark="./benchmark/videos-tmp/", root_predict="", sequence_len=16, frame_size=(64, 64)):
        self.gt_dir = os.path.join(root_benchmark)
        self.pred_dir = os.path.join(root_predict)
        self.sequence_len = sequence_len
        self.frame_size = frame_size

        # Assuming the video filenames are matched by name
        self.video_names = sorted(
            [f for f in os.listdir(self.gt_dir) if f.endswith('.mp4')]
        )

        # Transformation to resize and convert frames
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to [C,H,W] and scales to [0,1]
            transforms.Resize(self.frame_size),  # Resize to (64, 64)
        ])

    def __len__(self):
        return len(self.video_names)

    def read_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.frame_size)
            frames.append(frame)
        cap.release()

        # Ensure fixed sequence length
        if len(frames) < self.sequence_len:
            # Pad by repeating last frame
            frames += [frames[-1]] * (self.sequence_len - len(frames))
        elif len(frames) > self.sequence_len:
            # Truncate
            frames = frames[:self.sequence_len]

        frames = np.stack(frames)  # [T, H, W, C]
        return torch.from_numpy(frames).float() / 255.0  # Normalize to [0,1]

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        gt_path = os.path.join(self.gt_dir, video_name)
        pred_path = os.path.join(self.pred_dir, video_name)

        gt_video = self.read_video(gt_path)   # [T, H, W, C]
        pred_video = self.read_video(pred_path)  # [T, H, W, C]

        return pred_video.permute(0,3,1,2), gt_video.permute(0,3,1,2)  # Both shape [sequence_len, 64, 64, 3]

