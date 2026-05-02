"""
Text encoder wrapper cho SafeGuider input guard.

Quan trọng — đảm bảo embedding ở đây KHỚP với embedding mà recognizer
đã được train. Trong repo gốc, recognizer được train trên embedding sinh
ra bởi `FrozenCLIPEmbedder` của LDM (file `ldm/modules/encoders/modules.py`),
mà thực chất chỉ làm:

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    transformer = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    batch_encoding = tokenizer(text, truncation=True, max_length=77,
                               padding="max_length", return_tensors="pt")
    z = transformer(input_ids=batch_encoding["input_ids"]).last_hidden_state

Class `CLIPEncoder` ở đây replicate y hệt logic đó (KHÔNG truyền
`attention_mask`, vì pad_token_id của CLIP trùng với eos_token_id = 49407,
và LDM cũng không truyền attention_mask khi train).

Hỗ trợ thêm OpenCLIP (SD-V2.1, dim=1024) và T5 (Flux, dim=4096) qua
flag `--encoder-model`. Mặc định = `openai/clip-vit-large-patch14`
(SD-V1.4).
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# CLIP eos = pad = 49407. Khi tokenizer pad="max_length", PAD cũng = 49407
# nên vị trí EOS thực tế = lần xuất hiện ĐẦU TIÊN của 49407 trong input_ids.
CLIP_EOS_TOKEN_ID = 49407

# Default cache dir = <input_guard_only>/weights/. Mỗi model HF được tải về
# folder con `<weights>/<model_basename>/` (ví dụ `weights/clip-vit-large-patch14/`)
# để dễ quản lý — không lẫn vào HuggingFace cache mặc định ở ~/.cache/huggingface.
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")


# Module logger — tách riêng khỏi root logger để bật/tắt được mà không ảnh
# hưởng cấu hình logging chung. Chỉ in khi `CLIPEncoder(verbose=True)`.
log = logging.getLogger("safeguider.encoder")
if not log.handlers:
    _h = logging.StreamHandler(stream=sys.stderr)
    _h.setFormatter(logging.Formatter("[encoder] %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False


def _is_loadable_dir(path: str) -> bool:
    """Folder có chứa weight + config đủ để `from_pretrained(path)` load offline."""
    if not os.path.isdir(path):
        return False
    if not os.path.isfile(os.path.join(path, "config.json")):
        return False
    # Có ít nhất 1 file weight (an toàn cho cả pytorch_model.bin và .safetensors)
    for fn in os.listdir(path):
        low = fn.lower()
        if low.endswith((".safetensors", ".bin", ".pt", ".pth", ".msgpack")):
            return True
    return False


def resolve_encoder_path(
    model_name: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
    force_download: bool = False,
) -> str:
    """Local-first resolver cho 1 HuggingFace model id.

    Logic:
        1. Nếu `model_name` đã là 1 local folder hợp lệ → trả về luôn.
        2. Nếu `<cache_dir>/<basename>` đã hợp lệ → trả về (load offline).
        3. Ngược lại: `snapshot_download(model_name, local_dir=<cache_dir>/<basename>)`
           rồi trả về.

    Args:
        model_name:     "openai/clip-vit-large-patch14" hoặc 1 path local.
        cache_dir:      thư mục cha, default = `<input_guard_only>/weights/`.
        force_download: nếu True, luôn re-download (kể cả đã có local).

    Returns:
        Đường dẫn local folder mà `transformers.from_pretrained(...)` có thể load.
    """
    # Case 1: user truyền sẵn 1 local folder
    if _is_loadable_dir(model_name):
        return model_name

    basename = model_name.rstrip("/").split("/")[-1]
    target = os.path.join(cache_dir, basename)

    # Case 2: đã cache local
    if not force_download and _is_loadable_dir(target):
        return target

    # Case 3: download
    os.makedirs(cache_dir, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "Cần `huggingface_hub` để tự động download. "
            "Cài bằng `pip install huggingface_hub`, hoặc tự copy model "
            f"từ HuggingFace vào {target!r}."
        ) from e

    print(f"[encoder] downloading {model_name!r} -> {target}")
    snapshot_download(
        repo_id=model_name,
        local_dir=target,
        # Không tải file flax/tf để giảm dung lượng
        ignore_patterns=["*.msgpack", "*.h5", "tf_model.h5", "flax_model.msgpack"],
    )
    if not _is_loadable_dir(target):
        raise RuntimeError(
            f"Sau khi download, {target!r} vẫn không hợp lệ. "
            "Kiểm tra network / quyền ghi file."
        )
    return target


@dataclass
class EncodeResult:
    """Kết quả encode 1 batch prompt.

    Attributes:
        last_hidden_state: tensor (B, L, D) — sequence embedding đầy đủ (đã pad).
        eos_positions:     tensor (B,) int64 — vị trí EOS của mỗi prompt.
        eos_embedding:     tensor (B, D) — embedding tại EOS, dùng cho classifier.
        input_ids:         tensor (B, L) — token ids đã pad.
    """

    last_hidden_state: Tensor
    eos_positions: Tensor
    eos_embedding: Tensor
    input_ids: Tensor


class CLIPEncoder:
    """Wrapper text encoder + tokenizer (local-first).

    Args:
        model_name:    HF id `openai/clip-vit-large-patch14`, hoặc local path.
                       Default = SD-V1.4 encoder.
        cache_dir:     folder cha lưu model đã download. Default
                       `<input_guard_only>/weights/`. Mỗi model nằm trong folder con
                       cùng tên (ví dụ `weights/clip-vit-large-patch14/`).
        device:        `cuda` | `cpu` | None (auto).
        max_length:    77 cho CLIP. Đổi nếu dùng encoder khác.
        eos_token_id:  49407 cho CLIP. Đổi nếu dùng tokenizer khác (ví dụ T5).
        dtype:         torch dtype cho weight (default float32).
        force_download: nếu True, ép download lại kể cả đã cache local.

    Logic load:
        1) Nếu model_name là 1 local folder hợp lệ → load trực tiếp.
        2) Nếu `<cache_dir>/<basename>` đã có → load offline từ đó.
        3) Nếu chưa → snapshot_download từ HF về `<cache_dir>/<basename>` rồi load.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        cache_dir: str = DEFAULT_CACHE_DIR,
        device: Optional[str] = None,
        max_length: int = 77,
        eos_token_id: int = CLIP_EOS_TOKEN_ID,
        dtype: torch.dtype = torch.float32,
        force_download: bool = False,
        verbose: bool = False,
    ) -> None:
        # Import lười để CLIPEncoder không bắt buộc transformers nếu chỉ cần
        # các utility khác (ví dụ load .pt ngoại tuyến).
        from transformers import AutoTokenizer, AutoModel, CLIPTextModel, CLIPTokenizer
        from transformers import logging as hf_logging

        # Snapshot `openai/clip-vit-large-patch14` chứa FULL CLIP (text + vision +
        # 2 projection heads + logit_scale). Ta chỉ load `CLIPTextModel` (text-only),
        # nên transformers sẽ báo UNEXPECTED keys cho mọi `vision_model.*`,
        # `visual_projection`, `text_projection`, `logit_scale`. Đây là hành vi
        # mong đợi (text encoder vẫn load đầy đủ và đúng), không phải lỗi → silence.
        hf_logging.set_verbosity_error()

        self.device = torch.device(device) if device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.max_length = max_length
        self.eos_token_id = eos_token_id
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.verbose = bool(verbose)

        # Local-first resolve: trả về 1 local folder, KHÔNG dùng HF cache mặc định.
        local_path = resolve_encoder_path(model_name, cache_dir=cache_dir,
                                          force_download=force_download)
        self.local_path = local_path

        # Cố gắng dùng CLIPTokenizer / CLIPTextModel trước, fallback Auto*.
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(local_path, local_files_only=True)
            self.text_encoder = CLIPTextModel.from_pretrained(
                local_path, local_files_only=True,
            ).to(device=self.device, dtype=dtype)
        except (OSError, ValueError):
            self.tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
            self.text_encoder = AutoModel.from_pretrained(
                local_path, local_files_only=True,
            ).to(device=self.device, dtype=dtype)

        self.text_encoder.eval()
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        # Suy ra hidden_size — dùng để khởi tạo classifier.
        self.hidden_size: int = int(self.text_encoder.config.hidden_size)

        if self.verbose:
            log.info(
                f"loaded {model_name!r} from {local_path} | "
                f"device={self.device} dtype={dtype} | "
                f"hidden_size={self.hidden_size} max_length={self.max_length} "
                f"eos_token_id={self.eos_token_id}"
            )

    # ----------------------------- Core API ----------------------------- #

    @torch.no_grad()
    def encode(self, prompts: Sequence[str]) -> EncodeResult:
        """Encode 1 list prompts → EncodeResult.

        KHỚP với `FrozenCLIPEmbedder.forward` của LDM: tokenize với
        `padding="max_length"` và KHÔNG truyền attention_mask.

        Khi `verbose=True`, log:
          - raw_tokens:  số token mà tokenizer SINH RA (đã gồm BOS + EOS),
                         CHƯA truncate. Nếu > max_length thì sẽ bị cắt bớt.
          - tokens:      số token THỰC trong sequence sau khi truncate
                         (BOS + content + EOS, không tính PAD).
          - truncated:   số token đã bị cắt đi (= max(0, raw - max_length)).
          - shape của `last_hidden_state` (B, L, D).
          - shape của `eos_embedding` (B, D) — vector đưa vào classifier.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        # Pre-tokenize KHÔNG truncate/pad để biết raw token count (chỉ khi verbose
        # — tránh tốn thời gian ở vòng beam search).
        raw_token_counts: Optional[List[int]] = None
        if self.verbose:
            raw = self.tokenizer(
                list(prompts),
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )
            raw_token_counts = [len(ids) for ids in raw["input_ids"]]

        batch = self.tokenizer(
            list(prompts),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=False,
        )
        input_ids = batch["input_ids"].to(self.device)

        outputs = self.text_encoder(input_ids=input_ids)
        last_hidden_state = outputs.last_hidden_state  # (B, L, D)

        eos_positions = self._eos_positions(input_ids)
        eos_embedding = last_hidden_state[
            torch.arange(last_hidden_state.size(0), device=self.device), eos_positions
        ]

        if self.verbose:
            self._log_encode(prompts, eos_positions, last_hidden_state,
                             eos_embedding, raw_token_counts)

        return EncodeResult(
            last_hidden_state=last_hidden_state,
            eos_positions=eos_positions,
            eos_embedding=eos_embedding,
            input_ids=input_ids,
        )

    @torch.no_grad()
    def eos_embedding(self, prompts: Sequence[str]) -> Tensor:
        """Tiện ích: chỉ trả về EOS embedding, shape (B, D)."""
        return self.encode(prompts).eos_embedding

    # ----------------------------- Helpers ------------------------------ #

    def _eos_positions(self, input_ids: Tensor) -> Tensor:
        """Tìm vị trí EOS của mỗi sequence.

        Vì CLIP có pad_token_id == eos_token_id, vị trí EOS thực tế là
        lần xuất hiện ĐẦU TIÊN của eos_token_id trong sequence đã pad.
        Nếu không tìm thấy (trường hợp hiếm gặp khi truncate), fallback
        về `max_length - 1`.
        """
        is_eos = (input_ids == self.eos_token_id).int()
        if is_eos.any(dim=1).all():
            return is_eos.argmax(dim=1)

        # Fallback an toàn cho các sequence không có EOS
        positions = []
        L = input_ids.size(1)
        for row in is_eos:
            if row.any():
                positions.append(int(row.argmax().item()))
            else:
                positions.append(L - 1)
        return torch.tensor(positions, device=input_ids.device, dtype=torch.long)

    @staticmethod
    def cosine_similarity(a: Tensor, b: Tensor) -> Tensor:
        """Cosine similarity giữa 2 batch embedding (B, D)."""
        return F.cosine_similarity(a, b, dim=-1)

    def _log_encode(
        self,
        prompts: Sequence[str],
        eos_positions: Tensor,
        last_hidden_state: Tensor,
        eos_embedding: Tensor,
        raw_token_counts: Optional[List[int]] = None,
    ) -> None:
        """In token-count + shape khi `verbose=True`.

        - `raw_tokens`: số token sinh ra từ tokenizer (BOS + content + EOS),
          CHƯA truncate.
        - `tokens`:     số token THỰC sau truncate trong sequence pad
          = `eos_position + 1` (vì pad_token_id == eos_token_id ở CLIP).
        - `truncated`:  `max(0, raw_tokens - max_length)`.
        - Khi `truncated > 0`, EOS bị giữ ở vị trí cuối cùng của max_length;
          phần content giữa BOS và EOS bị cắt từ đuôi.
        """
        for i, (p, pos) in enumerate(zip(prompts, eos_positions.tolist())):
            n_tokens = int(pos) + 1
            n_pad = self.max_length - n_tokens
            preview = p if len(p) <= 60 else p[:57] + "..."

            if raw_token_counts is not None:
                raw = raw_token_counts[i]
                truncated = max(0, raw - self.max_length)
                trunc_tag = f" TRUNCATED!" if truncated > 0 else ""
                log.info(
                    f"prompt[{i}]={preview!r} | raw_tokens={raw} -> "
                    f"tokens={n_tokens} (pad={n_pad}, truncated={truncated}){trunc_tag} | "
                    f"eos_pos={pos}"
                )
            else:
                log.info(
                    f"prompt[{i}]={preview!r} | tokens={n_tokens} (pad={n_pad}) | "
                    f"eos_pos={pos}"
                )
        log.info(
            f"shapes: last_hidden_state={tuple(last_hidden_state.shape)} "
            f"-> eos_embedding={tuple(eos_embedding.shape)} "
            f"(classifier input dim={self.hidden_size})"
        )
