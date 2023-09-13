from torch.utils.data import Dataset 
import torch
import cv2
from utils import *

UCF101_PATH = "/mnt/c/Users/mytom/Downloads/UCF101/UCF-101/"

class VideoDataset(Dataset):
    def __init__(self, video_dir, input_size=(224, 224)):
        self.video_dir = video_dir
        self.video_filenames = read_ucf101_video(video_dir)
        self.input_size = input_size

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
            frame = cv2.resize(frame, self.input_size)  # make all frames the same size
            frame = torch.tensor(frame).permute(2, 0, 1)  # Convert to C, H, W format
            # Flatten the C, H, W tensor to a 1D tensor
            frame = frame.contiguous().view(-1)
            frames.append(frame)

        cap.release()
            
        frames_tensor = torch.stack(frames)
        return frames_tensor

if __name__ == "__main__":
    video_dataset = VideoDataset(UCF101_PATH)
