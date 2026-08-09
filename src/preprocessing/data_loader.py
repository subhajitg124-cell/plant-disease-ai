import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PlantDiseaseDataset(Dataset):
    """
    PyTorch Dataset for Plant Disease images.
    Supports OpenCV image loading and PyTorch vision transformations.
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Read image using OpenCV
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image at path: {img_path}")
        
        # Convert BGR (OpenCV standard) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
            
        label = self.labels[idx]
        return img, torch.tensor(label, dtype=torch.long)

def get_default_transforms(img_size=(128, 128)):
    """
    Returns standard train/val transformations.
    """
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_synthetic_plant_data(output_dir="data/synthetic_leaves", num_samples=100, img_size=(128, 128)):
    """
    Generates synthetic leaf images with simulated disease spots for initial experiment/testing.
    Classes: 0: Healthy, 1: Bacterial Spot, 2: Early Blight, 3: Late Blight
    """
    os.makedirs(output_dir, exist_ok=True)
    classes = {
        0: "Healthy",
        1: "Bacterial_Spot",
        2: "Early_Blight",
        3: "Late_Blight"
    }
    
    paths = []
    labels = []
    
    np.random.seed(42)
    
    for i in range(num_samples):
        cls_id = i % 4
        cls_name = classes[cls_id]
        
        # Base green leaf texture
        img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        # Green background with minor color variations
        img[:, :, 1] = np.random.randint(120, 200, size=img_size) # Green channel
        img[:, :, 0] = np.random.randint(20, 60, size=img_size)   # Blue channel
        img[:, :, 2] = np.random.randint(30, 80, size=img_size)   # Red channel
        
        # Draw disease symptoms using OpenCV
        if cls_id == 1: # Bacterial Spot (small brown spots)
            for _ in range(np.random.randint(5, 15)):
                cx, cy = np.random.randint(20, 108, size=2)
                cv2.circle(img, (cx, cy), np.random.randint(2, 5), (10, 30, 80), -1)
        elif cls_id == 2: # Early Blight (concentric brown rings)
            for _ in range(np.random.randint(2, 5)):
                cx, cy = np.random.randint(30, 98, size=2)
                r = np.random.randint(8, 16)
                cv2.circle(img, (cx, cy), r, (15, 45, 120), -1)
                cv2.circle(img, (cx, cy), max(1, r - 4), (20, 100, 50), 1)
        elif cls_id == 3: # Late Blight (large dark necrotic lesions)
            for _ in range(np.random.randint(2, 4)):
                cx, cy = np.random.randint(30, 98, size=2)
                axes = (np.random.randint(10, 25), np.random.randint(10, 25))
                cv2.ellipse(img, (cx, cy), axes, np.random.randint(0, 180), 0, 360, (20, 20, 40), -1)
                
        img_filename = os.path.join(output_dir, f"sample_{i:04d}_{cls_name}.jpg")
        cv2.imwrite(img_filename, img)
        paths.append(img_filename)
        labels.append(cls_id)
        
    return paths, labels
