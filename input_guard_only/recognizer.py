"""
Standalone recognizer: chỉ classify 1 prompt là SAFE / UNSAFE.

Có thể chạy 2 cách:

    1) CLI:
        python recognizer.py --prompt "your text prompt"
        python recognizer.py --from-file prompts.json

    2) Import như Python module:
        from recognizer import PromptRecognizer
        rec = PromptRecognizer(weights="weights/SD1.4_safeguider.pt")
        result = rec.classify("a cat sitting on a mat")
        # -> {"prompt": ..., "predicted_class": 1, "safety_score": 0.97,
        #     "probabilities": [0.03, 0.97], "is_safe": True}

Pipeline:
    prompt -> CLIPEncoder (HF CLIPTextModel) -> EOS embedding (D=768)
           -> ThreeLayerClassifier -> softmax -> {0: unsafe, 1: safe}

Recognizer này KHÔNG đụng tới Stable Diffusion / U-Net / VAE — chỉ
text encoder + 1 MLP nhỏ. Nhẹ và có thể chạy trên CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import torch

from classifier import ThreeLayerClassifier
from encoder import CLIPEncoder


DEFAULT_ENCODER = "openai/clip-vit-large-patch14"
DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "weights", "SD1.4_safeguider.pt")


class PromptRecognizer:
    """Class wrapper cho text encoder + safety classifier."""

    def __init__(
        self,
        weights: str = DEFAULT_WEIGHTS,
        encoder_model: str = DEFAULT_ENCODER,
        device: Optional[str] = None,
        safety_threshold: float = 0.5,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            weights:           path tới file `.pt` của classifier.
            encoder_model:     tên/path text encoder HuggingFace.
            device:            "cuda" | "cpu" | None (auto).
            safety_threshold:  ngưỡng score để verdict SAFE/UNSAFE
                               (chỉ ảnh hưởng `is_safe`, không ảnh hưởng `predicted_class`).
            verbose:           bật encoder log (token count, shape sau embedding,
                               size vector vào classifier).
        """
        if not os.path.isfile(weights):
            raise FileNotFoundError(
                f"Classifier weights not found at {weights!r}. "
                f"Hãy copy `Models/SD1.4_safeguider.pt` từ repo gốc vào folder weights/, "
                f"hoặc truyền --weights/--classifier-weights."
            )

        self.encoder = CLIPEncoder(model_name=encoder_model, device=device, verbose=verbose)
        self.device = self.encoder.device
        self.safety_threshold = float(safety_threshold)

        self.classifier = ThreeLayerClassifier(dim=self.encoder.hidden_size).to(self.device)
        state = torch.load(weights, map_location=self.device, weights_only=False)
        self.classifier.load_state_dict(state)
        self.classifier.eval()

    # ----------------------------- Inference ---------------------------- #

    @torch.no_grad()
    def classify(self, prompt: str) -> Dict[str, Any]:
        """Classify 1 prompt. Trả về dict với khóa:
            prompt, predicted_class (0|1), safety_score (float),
            probabilities (List[float] len=2), is_safe (bool).
        """
        return self.classify_batch([prompt])[0]

    @torch.no_grad()
    def classify_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Classify nhiều prompt cùng lúc, trả về list dict cùng schema."""
        eos_emb = self.encoder.eos_embedding(prompts)  # (B, D)
        logits, probs = self.classifier(eos_emb)       # (B, 2), (B, 2)
        preds = probs.argmax(dim=-1)
        results = []
        for i, p in enumerate(prompts):
            score = float(probs[i, 1].item())
            results.append({
                "prompt": p,
                "predicted_class": int(preds[i].item()),
                "safety_score": score,
                "probabilities": [float(probs[i, 0].item()), float(probs[i, 1].item())],
                "is_safe": score >= self.safety_threshold,
            })
        return results


# ============================== CLI =================================== #

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
                raise ValueError(f"Item không hợp lệ trong {path}: {it!r}")
        return out
    if isinstance(data, dict) and "data" in data:
        return [it["prompt"] for it in data["data"]]
    raise ValueError(f"Format file không hỗ trợ: {path}")


def _format_result(r: Dict[str, Any]) -> str:
    verdict = "SAFE" if r["is_safe"] else "UNSAFE"
    return (
        f"prompt   : {r['prompt']}\n"
        f"class    : {r['predicted_class']} ({'safe' if r['predicted_class']==1 else 'unsafe'})\n"
        f"score    : {r['safety_score']:.4f}  (P[unsafe]={r['probabilities'][0]:.4f}, "
        f"P[safe]={r['probabilities'][1]:.4f})\n"
        f"verdict  : {verdict}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="SafeGuider standalone safety recognizer.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", type=str, help="1 prompt để classify.")
    src.add_argument("--from-file", type=str, help="JSON file: list[str] hoặc list[{prompt}].")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS,
                        help=f"Classifier .pt path. Default: {DEFAULT_WEIGHTS}")
    parser.add_argument("--encoder-model", type=str, default=DEFAULT_ENCODER,
                        help=f"Text encoder HF id/path. Default: {DEFAULT_ENCODER}")
    parser.add_argument("--device", type=str, default=None, choices=[None, "cuda", "cpu"],
                        help="Device. Default: auto.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Safety score threshold để verdict SAFE/UNSAFE.")
    parser.add_argument("--output", type=str, default=None,
                        help="(optional) Lưu list kết quả ra JSON file.")
    parser.add_argument("--verbose", action="store_true",
                        help="In log encoder (số token thực + shape embedding + dim vào classifier).")
    args = parser.parse_args()

    rec = PromptRecognizer(
        weights=args.weights,
        encoder_model=args.encoder_model,
        device=args.device,
        safety_threshold=args.threshold,
        verbose=args.verbose,
    )

    if args.prompt is not None:
        results = [rec.classify(args.prompt)]
    else:
        prompts = _read_prompts(args.from_file)
        results = rec.classify_batch(prompts)

    for r in results:
        print(_format_result(r))
        print("-" * 60)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(results)} results -> {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
