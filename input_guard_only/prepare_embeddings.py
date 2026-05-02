"""
Tạo embedding dataset để train recognizer.

Input  (JSON):  list[{"prompt": str, ...}]   hoặc   list[str]
Output (JSON):  {"data": [{"id", "prompt", "embedding", "label", "eos_position"}, ...]}

Embedding ở đây là EOS token embedding của text encoder, KHỚP với
cách `FrozenCLIPEmbedder` của LDM tạo ra (xem `encoder.py`).

Ví dụ:
    # Tạo split benign (label=1)
    python prepare_embeddings.py --input benign.json  --output embed/benign.json  --label 1

    # Tạo split unsafe (label=0)
    python prepare_embeddings.py --input nsfw.json   --output embed/nsfw.json    --label 0

    # Sau đó merge 2 file vào 1 (xem `merge` ở dưới hoặc tự làm bằng tay):
    python prepare_embeddings.py --merge embed/benign.json embed/nsfw.json --output embed/train.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import torch
from tqdm import tqdm

from encoder import CLIPEncoder


# Defaults trỏ về layout chuẩn của folder input_guard_only/
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EMBED_DIR = os.path.join(_HERE, "embed")
DEFAULT_OUTPUT    = os.path.join(DEFAULT_EMBED_DIR, "embeddings.json")
DEFAULT_MERGED    = os.path.join(DEFAULT_EMBED_DIR, "train.json")


def _read_prompts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        out = []
        for it in data:
            if isinstance(it, str):
                out.append(it)
            elif isinstance(it, dict) and "prompt" in it:
                out.append(it["prompt"])
            else:
                raise ValueError(f"Item không hợp lệ: {it!r}")
        return out
    if isinstance(data, dict) and "data" in data:
        return [it["prompt"] for it in data["data"]]
    raise ValueError(f"Format file không hỗ trợ: {path}")


def build(
    prompts: List[str],
    encoder: CLIPEncoder,
    label: int,
    batch_size: int = 16,
) -> Dict[str, List[Dict[str, Any]]]:
    """Encode list prompt -> dataset dict."""
    items: List[Dict[str, Any]] = []
    next_id = 0
    for start in tqdm(range(0, len(prompts), batch_size), desc="encoding"):
        chunk = prompts[start:start + batch_size]
        result = encoder.encode(chunk)
        eos_emb = result.eos_embedding.cpu().tolist()       # list[list[float]]
        eos_pos = result.eos_positions.cpu().tolist()       # list[int]
        for p, emb, pos in zip(chunk, eos_emb, eos_pos):
            items.append({
                "id": next_id,
                "prompt": p,
                "embedding": emb,
                "label": int(label),
                "eos_position": int(pos),
            })
            next_id += 1
    return {"data": items}


def merge(paths: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    merged: List[Dict[str, Any]] = []
    next_id = 0
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        items = d["data"] if isinstance(d, dict) and "data" in d else d
        for it in items:
            it = dict(it)
            it["id"] = next_id
            next_id += 1
            merged.append(it)
    return {"data": merged}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build EOS-embedding dataset cho recognizer training.")
    sub = parser.add_subparsers(dest="cmd", required=False)

    # Mặc định: build từ 1 file prompts
    parser.add_argument("--input", type=str, help="JSON file chứa list prompts.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help=f"Path output JSON. Default: {DEFAULT_OUTPUT}")
    parser.add_argument("--label", type=int, choices=[0, 1], default=None,
                        help="Nhãn gán cho mọi prompt (0=unsafe, 1=safe). Bắt buộc khi --input.")
    parser.add_argument("--encoder-model", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--device", type=str, default=None, choices=[None, "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=16)

    # Hoặc merge nhiều file
    parser.add_argument("--merge", type=str, nargs="+", default=None,
                        help="Merge nhiều file embedding (đã build) thành 1. "
                             f"Khi dùng --merge, default --output là {DEFAULT_MERGED}.")

    args = parser.parse_args()

    if args.merge:
        # Khi merge, dùng default name khác (train.json) cho rõ nghĩa
        if args.output == DEFAULT_OUTPUT:
            args.output = DEFAULT_MERGED
        merged = merge(args.merge)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        print(f"Merged {len(merged['data'])} items -> {args.output}")
        return 0

    if not (args.input and args.label is not None):
        parser.error("Cần --input + --label (--output đã có default), hoặc --merge.")

    prompts = _read_prompts(args.input)
    print(f"Loaded {len(prompts)} prompts from {args.input}")
    encoder = CLIPEncoder(model_name=args.encoder_model, device=args.device)
    dataset = build(prompts, encoder, label=args.label, batch_size=args.batch_size)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(dataset['data'])} items (label={args.label}, dim={encoder.hidden_size}) -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
