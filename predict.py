import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition
from torch import nn
from torchvision import models
import glob

class Model(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

class VideoInferenceDataset(Dataset):
    def __init__(self, video_path, sequence_length=20, transform=None):
        self.video_path = video_path
        self.transform = transform
        self.count = sequence_length
        self.frames = self.preprocess_video()
        
    def preprocess_video(self):
        frames = []
        cap = cv2.VideoCapture(self.video_path)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect faces
            faces = face_recognition.face_locations(frame)
            if faces:
                top, right, bottom, left = faces[0]
                face_frame = frame[top:bottom, left:right, :]
                
                if self.transform:
                    face_frame = self.transform(face_frame)
                    frames.append(face_frame)
                
            if len(frames) >= self.count:
                frames = frames[:self.count]
                break
                
        cap.release()
        
        if len(frames) < self.count:
            # If we don't have enough frames, duplicate the last frame
            last_frame = frames[-1] if frames else None
            while len(frames) < self.count:
                frames.append(last_frame)
                
        return torch.stack(frames)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.frames.unsqueeze(0)

def predict_video(model, video_tensor, confidence_threshold=0.7):
    model.eval()
    softmax = nn.Softmax(dim=1)
    
    with torch.no_grad():
        fmap, logits = model(video_tensor.to('cuda'))
        probabilities = softmax(logits)
        
        _, prediction = torch.max(probabilities, 1)
        confidence = probabilities[0][prediction.item()].item()
        
        result = "REAL" if prediction.item() == 1 else "FAKE"
        return result, confidence * 100

def process_videos(model_path, video_folder="model-inference-video"):
    # Set up model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(2).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Set up transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Process all videos in the folder
    video_files = glob.glob(os.path.join(video_folder, "*.mp4"))
    results = []

    for video_path in video_files:
        print(f"\nProcessing video: {os.path.basename(video_path)}")
        
        try:
            # Create dataset for the video
            video_dataset = VideoInferenceDataset(
                video_path,
                sequence_length=20,
                transform=transform
            )
            
            if len(video_dataset) == 0:
                print(f"No faces detected in {os.path.basename(video_path)}")
                continue

            # Run inference
            prediction, confidence = predict_video(model, video_dataset[0])
            
            results.append({
                'video': os.path.basename(video_path),
                'prediction': prediction,
                'confidence': confidence
            })
            
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence:.2f}%")
            
        except Exception as e:
            print(f"Error processing {os.path.basename(video_path)}: {str(e)}")

    return results

def main():
    # Configure paths
    model_path = 'best_model.pt'  # Path to your trained model
    video_folder = 'model-inference-video'  # Folder containing videos to process
    
    # Create output folder if it doesn't exist
    os.makedirs(video_folder, exist_ok=True)
    
    # Process videos and get results
    results = process_videos(model_path, video_folder)
    
    # Print summary
    print("\nSummary of Results:")
    print("-" * 50)
    for result in results:
        print(f"Video: {result['video']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print("-" * 50)

if __name__ == "__main__":
    main()