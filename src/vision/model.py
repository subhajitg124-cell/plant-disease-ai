"""
PyTorch Convolutional Neural Network Model Architecture for Plant Disease Classification.

Defines PlantDiseaseCNN featuring a deep feature extraction backbone, bottleneck visual 
embeddings layer, and 38-class classification head.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PlantDiseaseCNN(nn.Module):
    """
    Modular CNN Architecture for Plant Disease Diagnosis.
    
    Features:
    - 4 Convolutional Blocks with Batch Normalization, ReLU, and Max Pooling.
    - Global Average Pooling (GAP) for spatial invariance.
    - Dense Bottleneck Embedding Layer (default: 128-dim) for RAG / vector search.
    - Final Fully-Connected Linear Layer for 38 plant disease classes.
    """
    def __init__(self, num_classes: int = 38, embedding_dim: int = 128):
        super(PlantDiseaseCNN, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Block 1: Input 3 x 224 x 224 -> 32 x 112 x 112
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2: 32 x 112 x 112 -> 64 x 56 x 56
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3: 64 x 56 x 56 -> 128 x 28 x 28
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Block 4: 128 x 28 x 28 -> 256 x 14 x 14
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Global Average Pooling -> 256
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Visual Embedding Bottleneck Layer (256 -> embedding_dim)
        self.fc_embedding = nn.Linear(256, embedding_dim)
        self.bn_emb = nn.BatchNorm1d(embedding_dim)

        # Classification Head (embedding_dim -> num_classes)
        self.classifier = nn.Linear(embedding_dim, num_classes)

        self.dropout = nn.Dropout(0.3)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts L2-normalized bottleneck visual embedding vectors.
        """
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.gap(x)
        x = torch.flatten(x, 1)

        emb = F.relu(self.bn_emb(self.fc_embedding(x)))
        # L2 Normalize visual embedding vector
        emb_norm = F.normalize(emb, p=2, dim=1)
        return emb_norm

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning tuple of (logits: Tensor, embedding: Tensor).
        """
        emb = self.extract_features(x)
        x_drop = self.dropout(emb)
        logits = self.classifier(x_drop)
        return logits, emb


def get_model(num_classes: int = 38, embedding_dim: int = 128) -> PlantDiseaseCNN:
    """Instantiates PlantDiseaseCNN model."""
    return PlantDiseaseCNN(num_classes=num_classes, embedding_dim=embedding_dim)
