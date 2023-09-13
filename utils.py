import os

def read_ucf101_video(root_dir):
    video_filepaths = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.avi'):  
                filepath = os.path.join(subdir, file)
                video_filepaths.append(filepath)
    return video_filepaths