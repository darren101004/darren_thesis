"""
Full input-guard pipeline = Recognizer (classify) + Beam Search (rewrite).

Cách dùng:

    1) CLI — chỉ classify (giống recognizer.py):
        python pipeline.py --mode classify --prompt "..."

    2) CLI — full pipeline (classify -> nếu unsafe thì rewrite):
        python pipeline.py --mode full --prompt "..."

    3) Batch từ JSON:
        python pipeline.py --mode full --from-file prompts.json --output result.json

    4) Import như module:
        from pipeline import SafeGuiderInputGuard
        guard = SafeGuiderInputGuard(weights="weights/SD1.4_safeguider.pt")
        out = guard.process("a violent prompt ...")
        # -> {original_prompt, predicted_class, safety_score, ...,
        #     was_modified, modified_prompt, removed_tokens, ...}

Pipeline:
    prompt -> CLIPEncoder.eos_embedding -> Classifier
        if predicted_class == 1 (safe): return ngay
        else: SafetyAwareBeamSearch.rewrite(prompt) -> modified prompt

Module thuần — KHÔNG đụng tới Stable Diffusion / U-Net / VAE.
Output là TEXT (prompt đã rewrite), người dùng tự đem text này đẩy vào
T2I model nào tùy ý.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch

from beam_search import (
    DEFAULT_BEAM_WIDTH,
    DEFAULT_MAX_DEPTH,
    DEFAULT_SAFETY_THRESHOLD,
    DEFAULT_SIMILARITY_FLOOR,
    SafetyAwareBeamSearch,
)
from classifier import ThreeLayerClassifier
from encoder import CLIPEncoder


DEFAULT_ENCODER = "openai/clip-vit-large-patch14"
DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "weights", "SD1.4_safeguider.pt")


class SafeGuiderInputGuard:
    """High-level wrapper kết hợp recognizer + beam-search rewriter."""

    def __init__(
        self,
        weights: str = DEFAULT_WEIGHTS,
        encoder_model: str = DEFAULT_ENCODER,
        device: Optional[str] = None,
        # Beam search hyper-params
        beam_width: int = DEFAULT_BEAM_WIDTH,
        max_depth: int = DEFAULT_MAX_DEPTH,
        safety_threshold: float = DEFAULT_SAFETY_THRESHOLD,
        similarity_floor: float = DEFAULT_SIMILARITY_FLOOR,
        verbose: bool = False,
    ) -> None:
        if not os.path.isfile(weights):
            raise FileNotFoundError(
                f"Classifier weights not found at {weights!r}. "
                f"Hãy copy `Models/SD1.4_safeguider.pt` từ repo gốc sang folder weights/."
            )

        # `verbose=True` bật CẢ encoder log (mỗi encode() in token count + shape)
        # VÀ beam-search trace (lưu vào output JSON khóa `beam_search_log`).
        # Cảnh báo: với mode=full, encoder log sẽ in vài nghìn dòng vì beam search
        # gọi encode() ~O(N × beam × depth) lần. Chỉ nên bật khi debug 1 prompt.
        self.encoder = CLIPEncoder(model_name=encoder_model, device=device,
                                   verbose=verbose)
        self.device = self.encoder.device

        self.classifier = ThreeLayerClassifier(dim=self.encoder.hidden_size).to(self.device)
        state = torch.load(weights, map_location=self.device, weights_only=False)
        self.classifier.load_state_dict(state)
        self.classifier.eval()

        self.beam_searcher = SafetyAwareBeamSearch(
            encoder=self.encoder,
            classifier=self.classifier,
            beam_width=beam_width,
            max_depth=max_depth,
            safety_threshold=safety_threshold,
            similarity_floor=similarity_floor,
            verbose=verbose,
        )
        self.verbose = verbose

    # ============================ Public API ============================ #

    @torch.no_grad()
    def classify(self, prompt: str) -> Dict[str, Any]:
        """Chỉ classify — không bao giờ rewrite. Trả về dict
        (predicted_class, safety_score, probabilities, is_safe).
        """
        eos_emb = self.encoder.eos_embedding([prompt])
        _, probs = self.classifier(eos_emb)
        if probs.dim() == 3:
            probs = probs.squeeze(1)
        pred = int(probs.argmax(dim=-1).item())
        score = float(probs[0, 1].item())
        return {
            "prompt": prompt,
            "predicted_class": pred,
            "safety_score": score,
            "probabilities": [float(probs[0, 0].item()), float(probs[0, 1].item())],
            "is_safe": pred == 1,
        }

    @torch.no_grad()
    def process(self, prompt: str, force_rewrite: bool = False) -> Dict[str, Any]:
        """Full pipeline cho 1 prompt.

        Args:
            prompt:        text gốc.
            force_rewrite: nếu True thì luôn chạy beam-search bất kể classifier nói gì
                           (hữu ích cho debug / sanity check).

        Returns:
            dict đầy đủ. Khóa quan trọng:
              - "original_prompt"
              - "predicted_class", "safety_score", "is_safe"
              - "was_modified"   (True nếu beam search đổi prompt)
              - "modified_prompt"
              - "removed_tokens"
              - "modified_safety", "similarity"
              - "elapsed_sec"
        """
        t0 = time.time()
        cls = self.classify(prompt)

        out: Dict[str, Any] = {
            "original_prompt": prompt,
            "predicted_class": cls["predicted_class"],
            "safety_score": cls["safety_score"],
            "probabilities": cls["probabilities"],
            "is_safe": cls["is_safe"],
            "was_modified": False,
            "modified_prompt": prompt,
            "removed_tokens": [],
            "modified_safety": cls["safety_score"],
            "similarity": 1.0,
        }

        should_rewrite = force_rewrite or (cls["predicted_class"] == 0)
        if should_rewrite:
            r = self.beam_searcher.rewrite(prompt)
            out.update({
                "was_modified": r.was_modified,
                "modified_prompt": r.modified_prompt,
                "removed_tokens": r.removed_tokens,
                "modified_safety": r.modified_safety,
                "similarity": r.similarity,
            })
            if self.verbose:
                out["beam_search_log"] = r.log

        out["elapsed_sec"] = round(time.time() - t0, 4)
        return out

    @torch.no_grad()
    def process_batch(self, prompts: List[str], force_rewrite: bool = False) -> List[Dict[str, Any]]:
        """Loop process cho từng prompt. (Beam search bản chất sequential
        nên không batch hóa được trong vòng tìm kiếm; chỉ classify ban đầu
        có thể batch — ở đây giữ đơn giản, gọi process từng cái.)"""
        return [self.process(p, force_rewrite=force_rewrite) for p in prompts]


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


def _format_full(r: Dict[str, Any]) -> str:
    lines = [
        f"original : {r['original_prompt']}",
        f"class    : {r['predicted_class']} (safe={r['is_safe']})  score={r['safety_score']:.4f}",
    ]
    if r["was_modified"]:
        lines += [
            f"rewrite  : {r['modified_prompt']}",
            f"removed  : {r['removed_tokens']}",
            f"modified : safety={r['modified_safety']:.4f}  sim={r['similarity']:.4f}",
        ]
    elif r["predicted_class"] == 0:
        lines.append("rewrite  : (UNSAFE nhưng beam search không tìm được rewrite — giữ prompt gốc)")
    else:
        lines.append("rewrite  : (skip — prompt đã SAFE)")
    lines.append(f"elapsed  : {r['elapsed_sec']}s")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SafeGuider input guard: classify + beam-search rewrite (text-only)."
    )
    parser.add_argument("--mode", type=str, default="full", choices=["classify", "full"],
                        help="`classify` chỉ phán SAFE/UNSAFE. `full` thêm bước rewrite nếu unsafe.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", type=str, help="1 prompt.")
    src.add_argument("--from-file", type=str, help="JSON file chứa list[str] hoặc list[{prompt}].")

    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    parser.add_argument("--encoder-model", type=str, default=DEFAULT_ENCODER)
    parser.add_argument("--device", type=str, default=None, choices=[None, "cuda", "cpu"])

    parser.add_argument("--beam-width", type=int, default=DEFAULT_BEAM_WIDTH)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--safety-threshold", type=float, default=DEFAULT_SAFETY_THRESHOLD)
    parser.add_argument("--similarity-floor", type=float, default=DEFAULT_SIMILARITY_FLOOR)
    parser.add_argument("--force-rewrite", action="store_true",
                        help="Luôn chạy beam search dù classifier nói SAFE.")

    parser.add_argument("--output", type=str, default=None,
                        help="Lưu kết quả ra JSON file.")
    parser.add_argument("--verbose", action="store_true",
                        help="In encoder log (token count + shape mỗi lần encode) "
                             "VÀ beam-search trace (lưu vào output JSON). "
                             "CHÚ Ý: với --mode full sẽ in vài nghìn dòng vì beam "
                             "search loop. Chỉ nên dùng để debug 1 prompt.")
    args = parser.parse_args()

    guard = SafeGuiderInputGuard(
        weights=args.weights,
        encoder_model=args.encoder_model,
        device=args.device,
        beam_width=args.beam_width,
        max_depth=args.max_depth,
        safety_threshold=args.safety_threshold,
        similarity_floor=args.similarity_floor,
        verbose=args.verbose,
    )

    prompts = [args.prompt] if args.prompt is not None else _read_prompts(args.from_file)

    results: List[Dict[str, Any]] = []
    for p in prompts:
        if args.mode == "classify":
            r = guard.classify(p)
        else:
            r = guard.process(p, force_rewrite=args.force_rewrite)
        results.append(r)

    # Print
    for r in results:
        if args.mode == "classify":
            print(f"prompt   : {r['prompt']}")
            print(f"class    : {r['predicted_class']} (safe={r['is_safe']})  score={r['safety_score']:.4f}")
        else:
            print(_format_full(r))
        print("-" * 60)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(results)} results -> {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
