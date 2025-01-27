import os
import cv2
import face_recognition
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import torch
    
def process_videos(input_folder, output_folder):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all the files in the input folder
    for video_file in os.listdir(input_folder):
        input_video_path = os.path.join(input_folder, video_file)

        # Ensure the file is a video
        if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue

        print(f"Processing video: {video_file}")

        # Initialize variables
        face_frames = []
        video_capture = cv2.VideoCapture(input_video_path)

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Convert the frame to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect face locations using GPU if available
            rgb_frame_gpu = torch.tensor(rgb_frame).to(device)
            face_locations = face_recognition.face_locations(rgb_frame_gpu.cpu().numpy())

            # If a face is found, extract and save it
            for top, right, bottom, left in face_locations:
                face_frame = frame[top:bottom, left:right]

                # Resize the face frame to a fixed size (e.g., 128x128)
                resized_face = cv2.resize(face_frame, (128, 128))
                face_frames.append(resized_face)

        video_capture.release()

        # Save the frames as a video
        if face_frames:
            output_video_path = os.path.join(output_folder, video_file)
            clip = ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in face_frames], fps=30)
            clip.write_videofile(output_video_path, codec='libx264')

        print(f"Saved processed video to {output_video_path}")

# Define input and output folders
input_folder = "celeb-real"
output_folder = "celeb-real-face-only"

# Process videos
process_videos(input_folder, output_folder)
