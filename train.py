import glob
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import pandas as pd
from torch import nn
from torchvision import models
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random

class VideoDataset(Dataset):
    def __init__(self, video_names, labels_df, sequence_length=60, transform=None):
        self.video_names = video_names
        self.labels_df = labels_df
        self.transform = transform
        self.count = sequence_length
        
    def __len__(self):
        return len(self.video_names)
    
    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        
        # Get video name and corresponding label
        video_name = os.path.basename(video_path)
        label = self.labels_df.loc[self.labels_df["file"] == video_name, "label"].values[0]
        label = 0 if label == 'FAKE' else 1
        
        # Extract frames
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        while frame_count < self.count:
            ret, frame = cap.read()
            if not ret:
                break
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
            frame_count += 1
        cap.release()
        
        # Stack frames and ensure correct sequence length
        frames = torch.stack(frames)
        frames = frames[:self.count]
        
        return frames, label

class DeepFakeDetector(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=2048):
        super(DeepFakeDetector, self).__init__()
        # Load pretrained ResNext
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        
        # LSTM and final layers
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers)
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(hidden_dim, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        
        # CNN feature extraction
        features = self.model(x)
        features = self.avgpool(features)
        features = features.view(batch_size, seq_length, 2048)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features, None)
        
        # Final classification
        out = self.dropout(self.linear(torch.mean(lstm_out, dim=1)))
        return out

def train_model(model, train_loader, valid_loader, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    
    best_accuracy = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_accuracy = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        valid_loss = 0
        valid_correct = 0
        valid_total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                valid_total += labels.size(0)
                valid_correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        valid_accuracy = 100. * valid_correct / valid_total
        
        # Save best model
        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            torch.save(model.state_dict(), 'best_model.pt')
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'Valid Loss: {valid_loss/len(valid_loader):.4f}, Valid Acc: {valid_accuracy:.2f}%')
        print('Confusion Matrix:')
        print(confusion_matrix(all_labels, all_predictions))
        print('--------------------')

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load videos and labels
    video_files = glob.glob('./fake_videos_face_only/*.mp4')
    labels_df = pd.read_csv('labels.csv', names=['file', 'label'])
    
    # Split data
    random.shuffle(video_files)
    train_size = int(0.8 * len(video_files))
    train_videos = video_files[:train_size]
    valid_videos = video_files[train_size:]
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets and dataloaders
    train_dataset = VideoDataset(train_videos, labels_df, sequence_length=10, transform=transform)
    valid_dataset = VideoDataset(valid_videos, labels_df, sequence_length=10, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=4)
    
    # Initialize model
    model = DeepFakeDetector().to(device)
    
    # Train model
    train_model(model, train_loader, valid_loader, num_epochs=20, device=device)

if __name__ == '__main__':
    main()