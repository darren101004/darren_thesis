"""
Safety-aware beam search rewriter.

Khi recognizer phán prompt là UNSAFE, module này tìm 1 prompt đã chỉnh sửa
(bằng cách XÓA TỪ trong prompt gốc, không thay/insert) sao cho:

    1. safety_score (P[class=safe]) >= safety_threshold (default 0.80)
    2. cosine similarity giữa EOS embedding mới vs gốc >= similarity_floor (default 0.1)
    3. (tie-break) số từ bị xóa càng ít càng tốt

Pseudo-code (xem chi tiết ở `transfer.md` §1.3):

    1) Rank từng từ theo "đóng góp vào unsafe" (xóa 1 từ → safety tăng nhiều nhất).
    2) Beam search depth max_depth, mỗi bước thử thêm 1 token để xóa.
    3) Early stop khi tìm được candidate thỏa cả 2 ngưỡng.
    4) Fallback: nếu không có candidate thỏa, lấy candidate có safety cao nhất
       trong các candidate có similarity >= similarity_floor.

Module thuần — không phụ thuộc Stable Diffusion / LDM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from encoder import CLIPEncoder


# Mặc định khớp với bản trong `stable-diffusion-1.4/scripts/safeguider_gene.py`.
DEFAULT_BEAM_WIDTH: int = 6
DEFAULT_MAX_DEPTH: int = 25
DEFAULT_SAFETY_THRESHOLD: float = 0.80
DEFAULT_SIMILARITY_FLOOR: float = 0.1


@dataclass
class BeamSearchResult:
    original_prompt: str
    modified_prompt: str
    was_modified: bool
    original_safety: float
    modified_safety: float
    similarity: float
    removed_tokens: List[str] = field(default_factory=list)
    # Trace dùng để debug/log; không bắt buộc đọc.
    log: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_prompt": self.original_prompt,
            "modified_prompt": self.modified_prompt,
            "was_modified": self.was_modified,
            "original_safety": self.original_safety,
            "modified_safety": self.modified_safety,
            "similarity": self.similarity,
            "removed_tokens": self.removed_tokens,
        }


class SafetyAwareBeamSearch:
    """Rewriter dùng beam search trên đơn vị từ (whitespace split)."""

    def __init__(
        self,
        encoder: CLIPEncoder,
        classifier: torch.nn.Module,
        beam_width: int = DEFAULT_BEAM_WIDTH,
        max_depth: int = DEFAULT_MAX_DEPTH,
        safety_threshold: float = DEFAULT_SAFETY_THRESHOLD,
        similarity_floor: float = DEFAULT_SIMILARITY_FLOOR,
        verbose: bool = False,
    ) -> None:
        self.encoder = encoder
        self.classifier = classifier.eval()
        self.device = self.encoder.device

        self.beam_width = int(beam_width)
        self.max_depth = int(max_depth)
        self.safety_threshold = float(safety_threshold)
        self.similarity_floor = float(similarity_floor)
        self.verbose = bool(verbose)

    # ============================ Public API ============================ #

    @torch.no_grad()
    def rewrite(self, prompt: str) -> BeamSearchResult:
        """Rewrite 1 prompt UNSAFE → 1 prompt mới an toàn (hoặc gốc nếu không tìm được)."""
        words: List[str] = prompt.split()
        log: List[str] = []

        # Embedding & safety của prompt gốc
        orig_eos = self.encoder.eos_embedding([prompt])           # (1, D)
        orig_safety = self._safety_score(orig_eos)                # float in [0, 1]
        log.append(f"original safety: {orig_safety:.4f}")

        if len(words) <= 1:
            log.append("Prompt chỉ có 1 từ, không thể beam search.")
            return BeamSearchResult(
                original_prompt=prompt, modified_prompt=prompt, was_modified=False,
                original_safety=orig_safety, modified_safety=orig_safety,
                similarity=1.0, removed_tokens=[], log=log,
            )

        # 1) Ranking impact: xóa từng từ độc lập, đo safety improvement
        token_impacts: List[Tuple[int, float]] = []
        for idx in range(len(words)):
            modified = " ".join(words[:idx] + words[idx + 1:])
            if not modified.strip():
                continue
            eos_i = self.encoder.eos_embedding([modified])
            s_i = self._safety_score(eos_i)
            token_impacts.append((idx, s_i - orig_safety))
        token_impacts.sort(key=lambda x: x[1], reverse=True)
        log.append(f"impact ranking (top 5): {token_impacts[:5]}")

        # 2) Beam search
        max_depth = min(self.max_depth, len(words) - 1)
        # Mỗi candidate: (removed_indices: List[int], improvement: float, similarity: float)
        candidates: List[Tuple[List[int], float, float]] = [([], 0.0, 1.0)]

        best_modified_prompt: Optional[str] = None
        best_safety_improvement: float = 0.0
        best_similarity: float = 0.0
        best_tokens_removed: List[str] = []
        all_seen: List[Tuple[List[int], float, float, float]] = []  # for fallback

        for depth in range(max_depth):
            new_candidates_step: List[Tuple[List[int], float, float, float]] = []
            qualified: List[Tuple[List[int], float, float]] = []

            for removed_indices, _imp, _sim in candidates:
                for idx, _impact in token_impacts:
                    if idx in removed_indices:
                        continue
                    new_indices = removed_indices + [idx]
                    new_words = [w for i, w in enumerate(words) if i not in new_indices]
                    if not new_words:
                        continue
                    cur_prompt = " ".join(new_words)

                    cur_eos = self.encoder.eos_embedding([cur_prompt])
                    cur_safety = self._safety_score(cur_eos)
                    sim = float(self.encoder.cosine_similarity(orig_eos, cur_eos).item())
                    improvement = cur_safety - orig_safety

                    new_candidates_step.append((new_indices, improvement, sim, cur_safety))
                    all_seen.append((new_indices, improvement, sim, cur_safety))

                    if cur_safety >= self.safety_threshold and sim >= self.similarity_floor:
                        qualified.append((new_indices, improvement, sim))
                        # Update best (maximize improvement, then minimize #removals)
                        if (best_modified_prompt is None
                                or improvement > best_safety_improvement
                                or (improvement == best_safety_improvement
                                    and len(new_indices) < len(best_tokens_removed))):
                            best_modified_prompt = cur_prompt
                            best_safety_improvement = improvement
                            best_similarity = sim
                            best_tokens_removed = [words[i] for i in new_indices]

            if qualified:
                candidates = sorted(qualified, key=lambda x: (x[1], -len(x[0])))[-self.beam_width:]
                tag = "qualified"
            elif new_candidates_step:
                # Sort by raw safety, prefer fewer removals
                topk = sorted(new_candidates_step, key=lambda x: (x[3], -len(x[0])))[-self.beam_width:]
                candidates = [(ind, imp, sim) for ind, imp, sim, _ in topk]
                tag = "fallback"
            else:
                tag = "empty"

            log.append(f"depth {depth+1}: {len(qualified)} qualified, "
                       f"{len(new_candidates_step)} expanded, picked={tag}")

            # Early stopping
            if best_modified_prompt is not None and (best_safety_improvement + orig_safety) >= self.safety_threshold:
                log.append("early stop: tìm được satisfactory solution.")
                break

        # 3) Final selection
        if best_modified_prompt is not None:
            mod_safety = orig_safety + best_safety_improvement
            log.append(f"chose qualified: removed={best_tokens_removed} safety={mod_safety:.4f}")
            return BeamSearchResult(
                original_prompt=prompt,
                modified_prompt=best_modified_prompt,
                was_modified=True,
                original_safety=orig_safety,
                modified_safety=mod_safety,
                similarity=best_similarity,
                removed_tokens=best_tokens_removed,
                log=log,
            )

        # Fallback: candidate có safety cao nhất với sim >= similarity_floor
        valid = [c for c in all_seen if c[2] >= self.similarity_floor]
        if valid:
            best_candidate = max(valid, key=lambda x: x[3])
            best_indices, _, best_sim, best_safety = best_candidate
            new_words = [w for i, w in enumerate(words) if i not in best_indices]
            best_alt = " ".join(new_words) if new_words else prompt
            log.append(f"chose fallback: removed={[words[i] for i in best_indices]} safety={best_safety:.4f}")
            return BeamSearchResult(
                original_prompt=prompt,
                modified_prompt=best_alt,
                was_modified=(best_alt != prompt),
                original_safety=orig_safety,
                modified_safety=best_safety,
                similarity=best_sim,
                removed_tokens=[words[i] for i in best_indices],
                log=log,
            )

        log.append("không tìm được candidate thỏa similarity_floor — giữ prompt gốc.")
        return BeamSearchResult(
            original_prompt=prompt, modified_prompt=prompt, was_modified=False,
            original_safety=orig_safety, modified_safety=orig_safety,
            similarity=1.0, removed_tokens=[], log=log,
        )

    # ============================ Internals ============================== #

    @torch.no_grad()
    def _safety_score(self, eos_emb: torch.Tensor) -> float:
        """Trả về P[class=safe] (scalar) từ EOS embedding (1, D)."""
        # Một số .pt được lưu với forward expect (B, D). Một số file gốc của
        # SafeGuider lại unsqueeze thành (B, 1, D) (xem `safeguider_gene.py`
        # `compute_safety_score`). Chúng ta dùng (B, D) cho gọn — kết quả
        # không đổi vì layer Linear áp dụng theo last dim.
        _, probs = self.classifier(eos_emb)
        if probs.dim() == 3:
            probs = probs.squeeze(1)
        return float(probs[0, 1].item())
