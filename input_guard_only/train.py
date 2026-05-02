"""
Train ThreeLayerClassifier từ embedding dataset đã build bằng `prepare_embeddings.py`.

Input dataset format:
    {"data": [{"embedding": [float, ...], "label": int, ...}, ...]}

Lệnh:
    python train.py \
        --train embed/train.json \
        --val   embed/val.json    \
        --output weights/my_recognizer.pt \
        --epochs 50 --batch-size 32 --lr 1e-3

Có thể bỏ --val để skip evaluation.

Mặc định dùng `nn.CrossEntropyLoss`. Nếu muốn margin-based loss như paper
(safe ≥ 0.6, unsafe ∈ [0.2, 0.4]), thêm flag `--loss margin`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from classifier import create_model


# Defaults trỏ về layout chuẩn của folder input_guard_only/
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRAIN  = os.path.join(_HERE, "embed", "train.json")
DEFAULT_VAL    = os.path.join(_HERE, "embed", "val.json")
DEFAULT_OUTPUT = os.path.join(_HERE, "weights", "recognizer.pt")


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class EmbeddingDataset(Dataset):
    """Đọc dataset JSON `{"data": [{embedding, label}, ...]}`."""

    def __init__(self, json_path: str) -> None:
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if not (isinstance(d, dict) and "data" in d):
            raise ValueError(f"{json_path}: cần `{{\"data\": [...]}}`.")
        self.items = d["data"]
        if not self.items:
            raise ValueError(f"{json_path}: empty.")
        # Suy ra dim từ item đầu
        emb0 = self.items[0]["embedding"]
        # Đôi khi prepare_embeddings.py lưu (1, D) nested 1 lớp; flatten nếu cần
        if isinstance(emb0[0], list):
            self.dim = len(emb0[0])
            self._nested = True
        else:
            self.dim = len(emb0)
            self._nested = False

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        it = self.items[idx]
        emb = it["embedding"]
        if self._nested:
            emb = emb[0]
        return (
            torch.tensor(emb, dtype=torch.float32),
            torch.tensor(int(it["label"]), dtype=torch.long),
        )


def margin_loss(logits: torch.Tensor, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Loss như trong `tools/train.py` của repo gốc:
        - target=1 (safe):    P[safe] >= 0.6
        - target=0 (unsafe):  0.2 <= P[safe] <= 0.4
    """
    safe = probs[:, 1]
    pos = (targets == 1)
    neg = (targets == 0)
    pos_loss = F.relu(0.6 - safe[pos]).mean() if pos.any() else torch.tensor(0.0, device=safe.device)
    neg_upper = F.relu(safe[neg] - 0.4).mean() if neg.any() else torch.tensor(0.0, device=safe.device)
    neg_lower = F.relu(0.2 - safe[neg]).mean() if neg.any() else torch.tensor(0.0, device=safe.device)
    return pos_loss + neg_upper + neg_lower


def train_one_epoch(model, loader, optimizer, loss_kind: str, device) -> Tuple[float, float]:
    model.train()
    crit = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    for emb, target in tqdm(loader, desc="train", leave=False):
        emb, target = emb.to(device), target.to(device)
        optimizer.zero_grad()
        logits, probs = model(emb)
        if loss_kind == "ce":
            loss = crit(logits, target)
        else:
            loss = margin_loss(logits, probs, target)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * target.size(0)
        pred = probs.argmax(dim=-1)
        correct += int((pred == target).sum().item())
        total += int(target.size(0))
    return total_loss / max(total, 1), 100.0 * correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, loss_kind: str, device) -> Tuple[float, float]:
    model.eval()
    crit = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    for emb, target in loader:
        emb, target = emb.to(device), target.to(device)
        logits, probs = model(emb)
        if loss_kind == "ce":
            loss = crit(logits, target)
        else:
            loss = margin_loss(logits, probs, target)
        total_loss += float(loss.item()) * target.size(0)
        pred = probs.argmax(dim=-1)
        correct += int((pred == target).sum().item())
        total += int(target.size(0))
    return total_loss / max(total, 1), 100.0 * correct / max(total, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train SafeGuider safety classifier.")
    parser.add_argument("--train", type=str, default=DEFAULT_TRAIN,
                        help=f"Train JSON dataset. Default: {DEFAULT_TRAIN}")
    parser.add_argument("--val", type=str, default=None,
                        help=f"(optional) Val JSON dataset. Suggest: {DEFAULT_VAL}")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help=f"Path lưu .pt. Default: {DEFAULT_OUTPUT}")
    parser.add_argument("--layers", type=int, default=3, choices=[1, 3, 5, 7, 9])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam", "adamw"])
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "margin"])
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--device", type=str, default=None, choices=[None, "cuda", "cpu"])
    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    if not os.path.isfile(args.train):
        raise FileNotFoundError(
            f"Train dataset not found: {args.train!r}. "
            f"Build trước bằng `python prepare_embeddings.py --input <prompts.json> "
            f"--output {args.train} --label <0|1>`."
        )
    train_ds = EmbeddingDataset(args.train)
    print(f"[train] {len(train_ds)} samples, dim={train_ds.dim}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    val_loader = None
    if args.val:
        val_ds = EmbeddingDataset(args.val)
        print(f"[val]   {len(val_ds)} samples, dim={val_ds.dim}")
        if val_ds.dim != train_ds.dim:
            raise ValueError(f"Train/val dim mismatch: {train_ds.dim} vs {val_ds.dim}")
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = create_model(args.layers, train_ds.dim).to(device)
    if args.optimizer == "sgd":
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                        weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, args.loss, device)
        msg = f"[epoch {epoch:3d}] train loss={tr_loss:.4f} acc={tr_acc:.2f}%"
        if val_loader is not None:
            v_loss, v_acc = evaluate(model, val_loader, args.loss, device)
            msg += f"   val loss={v_loss:.4f} acc={v_acc:.2f}%"
            if v_acc > best_acc:
                best_acc = v_acc
                torch.save(model.state_dict(), args.output)
                msg += "  ✓ saved (best)"
        print(msg)

    if val_loader is None:
        torch.save(model.state_dict(), args.output)
        print(f"Saved final weights -> {args.output}")
    else:
        print(f"Best val acc = {best_acc:.2f}% — weights at {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
