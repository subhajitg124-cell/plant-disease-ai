import sys
import os
import time
# pyrefly: ignore [missing-import]
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.data_loader import create_synthetic_plant_data, PlantDiseaseDataset, get_default_transforms
from src.vision.classifier import get_model

def run_experiment(epochs=5, batch_size=16, lr=0.001):
    print("=" * 60)
    print("      PLANT DISEASE AI - FIRST CNN EXPERIMENT")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution Device: {device}")
    
    # 1. Generate Synthetic Data
    print("[1/5] Preparing Plant Disease Dataset...")
    data_dir = os.path.join("data", "synthetic_leaves")
    paths, labels = create_synthetic_plant_data(output_dir=data_dir, num_samples=160, img_size=(128, 128))
    
    # 2. Train/Validation Split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_transform, val_transform = get_default_transforms(img_size=(128, 128))
    
    train_dataset = PlantDiseaseDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = PlantDiseaseDataset(val_paths, val_labels, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"    - Training Samples:   {len(train_dataset)}")
    print(f"    - Validation Samples: {len(val_dataset)}")
    
    # 3. Model Setup
    print("[2/5] Initializing PlantDiseaseCNN Architecture...")
    model = get_model(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 4. Training Loop
    print("[3/5] Training Model...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = 100.0 * val_correct / val_total
        
        print(f"Epoch [{epoch}/{epochs}] | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.2f}%")
        
    elapsed = time.time() - start_time
    print(f"[4/5] Training Completed in {elapsed:.2f} seconds.")
    
    # 5. Save Checkpoint
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "baseline_cnn.pth")
    torch.save(model.state_dict(), model_path)
    print(f"[5/5] Model Saved Successfully to '{model_path}'!")
    print("=" * 60)

if __name__ == "__main__":
    run_experiment()
