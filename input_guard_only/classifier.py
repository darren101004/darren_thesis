"""
MLP architectures dùng làm safety classifier cho SafeGuider input guard.

Recognizer mặc định (ThreeLayerClassifier) là kiến trúc đã được train sẵn
trong các file weight `Models/SD1.4_safeguider.pt`, `SD2.1_safeguider.pt`,
`Flux_safeguider.pt` của repo gốc. Tham số `dim` phải khớp với hidden size
của text encoder:

    - SD-V1.4 (CLIP ViT-L/14):       dim = 768
    - SD-V2.1 (OpenCLIP ViT-H/14):   dim = 1024
    - Flux.1  (T5-XXL):              dim = 4096

Các biến thể 1/5/7/9 layer ở dưới chỉ phục vụ ablation khi muốn tự train
classifier mới — KHÔNG load được trực tiếp từ các checkpoint đính kèm repo.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OneLayerClassifier(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, 2)

    def forward(self, x: torch.Tensor):
        logits = self.fc1(x)
        prob = F.softmax(logits, dim=-1)
        return logits, prob


class ThreeLayerClassifier(nn.Module):
    """Kiến trúc dùng cho mọi `.pt` đính kèm repo gốc."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)
        prob = F.softmax(logits, dim=-1)
        return logits, prob


class FiveLayerClassifier(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.7)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x)); x = self.dropout(x)
        x = F.relu(self.fc2(x)); x = self.dropout(x)
        x = F.relu(self.fc3(x)); x = self.dropout(x)
        x = F.relu(self.fc4(x)); x = self.dropout(x)
        logits = self.fc5(x)
        prob = F.softmax(logits, dim=-1)
        return logits, prob


class SevenLayerClassifier(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.7)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x)); x = self.dropout(x)
        x = F.relu(self.fc2(x)); x = self.dropout(x)
        x = F.relu(self.fc3(x)); x = self.dropout(x)
        x = F.relu(self.fc4(x)); x = self.dropout(x)
        x = F.relu(self.fc5(x)); x = self.dropout(x)
        x = F.relu(self.fc6(x)); x = self.dropout(x)
        logits = self.fc7(x)
        prob = F.softmax(logits, dim=-1)
        return logits, prob


class NineLayerClassifier(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 128)
        self.fc7 = nn.Linear(128, 64)
        self.fc8 = nn.Linear(64, 32)
        self.fc9 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.7)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x)); x = self.dropout(x)
        x = F.relu(self.fc2(x)); x = self.dropout(x)
        x = F.relu(self.fc3(x)); x = self.dropout(x)
        x = F.relu(self.fc4(x)); x = self.dropout(x)
        x = F.relu(self.fc5(x)); x = self.dropout(x)
        x = F.relu(self.fc6(x)); x = self.dropout(x)
        x = F.relu(self.fc7(x)); x = self.dropout(x)
        x = F.relu(self.fc8(x)); x = self.dropout(x)
        logits = self.fc9(x)
        prob = F.softmax(logits, dim=-1)
        return logits, prob


_REGISTRY = {
    1: OneLayerClassifier,
    3: ThreeLayerClassifier,
    5: FiveLayerClassifier,
    7: SevenLayerClassifier,
    9: NineLayerClassifier,
}


def create_model(layer: int, input_dim: int) -> nn.Module:
    if layer not in _REGISTRY:
        raise ValueError(f"Invalid layer={layer}. Choose from {list(_REGISTRY.keys())}.")
    return _REGISTRY[layer](dim=input_dim)
