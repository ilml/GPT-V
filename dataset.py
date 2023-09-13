from torch.utils.data import Dataset 
import torch
import os
import cv2
from utils import *

UCF101_PATH = ""

class VideoDataset(Dataset):
    def __init__(self, video_dir):
        self.video_dir = video_dir
        self.video_filenames = read_ucf101_video(video_dir)

    def __len__(self):
        return len(self.video_filenames)

    def __getitem__(self, index):
        video_filename = self.video_filenames[index]
        video_path = os.path.join(self.video_dir, video_filename)
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Normalize and preprocess frame if needed
            frame = torch.tensor(frame).permute(2, 0, 1)  # Convert to C, H, W format
            # Flatten the C, H, W tensor to a 1D tensor
            frame = frame.view(-1)
            frames.append(frame)

        cap.release()
            
        frames_tensor = torch.stack(frames)
        return frames_tensor

if __name__ == "__main__":
    video_dataset = VideoDataset(UCF101_PATH)
