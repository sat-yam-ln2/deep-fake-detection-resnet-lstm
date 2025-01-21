import json
import glob
import numpy as np
import cv2
import os
import face_recognition
from tqdm.autonotebook import tqdm

def frame_extract(path):
    """Extract frames from a video file."""
    vidObj = cv2.VideoCapture(path)
    success = 1
    while success:
        success, image = vidObj.read()
        if success:
            yield image

def analyze_videos(video_path):
    """Analyze videos to get frame count statistics."""
    video_files = glob.glob(os.path.join(video_path, '*.mp4'))
    frame_count = []
    
    for video_file in video_files[:]:  # Create a copy of the list to modify
        cap = cv2.VideoCapture(video_file)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 150:
            video_files.remove(video_file)
            continue
        frame_count.append(count)
        cap.release()
    
    print("Frames:", frame_count)
    print("Total number of videos:", len(frame_count))
    print('Average frame per video:', np.mean(frame_count))
    
    return video_files

def create_face_videos(path_list, out_dir):
    """Extract faces from videos and save them as new videos."""
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Check for already processed videos
    already_present_count = glob.glob(os.path.join(out_dir, '*.mp4'))
    print("Number of videos already present:", len(already_present_count))
    
    for path in tqdm(path_list):
        out_path = os.path.join(out_dir, os.path.basename(path))
        
        # Skip if file already exists
        if os.path.exists(out_path):
            print("File already exists:", out_path)
            continue
        
        frames = []
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (112,112))
        
        for idx, frame in enumerate(frame_extract(path)):
            if idx <= 150:  # Process first 150 frames
                frames.append(frame)
                if len(frames) == 4:  # Process in batches of 4 frames
                    faces = face_recognition.batch_face_locations(frames)
                    for i, face in enumerate(faces):
                        if len(face) != 0:
                            top, right, bottom, left = face[0]
                            try:
                                face_frame = cv2.resize(frames[i][top:bottom, left:right, :], (112,112))
                                out.write(face_frame)
                            except:
                                pass
                    frames = []
        
        out.release()

def main():
    # Set paths for fake videos
    fake_video_dir = "./fake_videos"  # Directory containing fake videos
    fake_output_dir = "./fake_videos_face_only"  # Directory for processed fake videos
    
    # Process fake videos
    print("Processing fake videos...")
    fake_video_files = analyze_videos(fake_video_dir)
    create_face_videos(fake_video_files, fake_output_dir)

if __name__ == "__main__":
    main()