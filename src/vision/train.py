"""
CNN Model Training Module for Plant Disease AI.
"""

import os
import sys

# Ensure root workspace directory is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from typing import Optional, Dict, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from src.vision.model import PlantDiseaseCNN
except ImportError:
    from model import PlantDiseaseCNN


def create_synthetic_dataset(num_samples: int = 100, num_classes: int = 38):
    """Generates synthetic PyTorch dataset for local training / testing."""
    if not HAS_TORCH:
        return None, None

    x_data = torch.randn(num_samples, 3, 224, 224)
    y_data = torch.randint(0, num_classes, (num_samples,))
    return x_data, y_data


def train_model(
    epochs: int = 2,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    save_path: str = "models/plant_disease_cnn.pth",
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Trains PlantDiseaseCNN and saves checkpoint.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not HAS_TORCH:
        print("PyTorch unavailable. Creating placeholder checkpoint info.")
        return {"status": "skipped", "message": "PyTorch not available."}

    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    model = PlantDiseaseCNN(num_classes=38).to(device_obj)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    x_train, y_train = create_synthetic_dataset(num_samples=64, num_classes=38)
    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "train_acc": []}

    print(f"Starting training on device: {device_obj} for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images, labels = images.to(device_obj), labels.to(device_obj)

            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc*100:.2f}%")

    checkpoint = {
        "model_version": "vision_v1",
        "num_classes": 38,
        "embedding_dim": 128,
        "state_dict": model.state_dict(),
        "final_loss": history["train_loss"][-1],
        "final_acc": history["train_acc"][-1]
    }
    torch.save(checkpoint, save_path)
    print(f"Model checkpoint saved successfully to: {save_path}")

    return {
        "status": "success",
        "save_path": save_path,
        "final_loss": history["train_loss"][-1],
        "final_acc": history["train_acc"][-1]
    }


if __name__ == "__main__":
    train_model(epochs=2)
