# Weights

Thư mục này chứa **cả 2** loại artefact:

1. **Classifier weights** — file `.pt` của safety MLP recognizer
   (ví dụ `SD1.4_safeguider.pt`). Copy thủ công từ `Models/` của repo gốc.
2. **Text encoder snapshots** — folder con như `clip-vit-large-patch14/`
   chứa `config.json` + `pytorch_model.bin`/`*.safetensors` + tokenizer.
   **Tự động** được tạo lần đầu chạy nhờ `snapshot_download` từ HuggingFace
   (xem `encoder.py:resolve_encoder_path`). Các lần sau load offline 100%.

`.gitignore` nên ignore cả folder này (file lớn, dễ tái tạo).

## Cách lấy weight

### Option A — copy từ repo gốc SafeGuider (khuyến nghị)
Nếu bạn đang clone full repo SafeGuider:
```bash
cp ../Models/SD1.4_safeguider.pt  ./SD1.4_safeguider.pt   # CLIP ViT-L/14, dim=768
cp ../Models/SD2.1_safeguider.pt  ./SD2.1_safeguider.pt   # OpenCLIP ViT-H/14, dim=1024
cp ../Models/Flux_safeguider.pt   ./Flux_safeguider.pt    # T5-XXL, dim=4096
```

### Option B — clone từ GitHub
```bash
git clone https://github.com/pgqihere/SafeGuider /tmp/safeguider
cp /tmp/safeguider/Models/*.pt ./
```

### Option C — pre-download text encoder thủ công (không bắt buộc)
Nếu muốn tải text encoder trước khi chạy lần đầu (ví dụ máy không có
internet lúc inference), chạy:
```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="openai/clip-vit-large-patch14",
    local_dir="weights/clip-vit-large-patch14",
    ignore_patterns=["*.msgpack", "*.h5", "tf_model.h5", "flax_model.msgpack"],
)
```
Hoặc đơn giản chạy 1 lệnh classify bất kỳ — folder này sẽ tự được tạo:
```bash
python recognizer.py --prompt "first run will download encoder"
```

## Bảng tương ứng encoder ↔ weight ↔ dim

| Backbone     | encoder-model (HuggingFace)                | weight                | dim   |
|--------------|--------------------------------------------|-----------------------|-------|
| SD-V1.4      | `openai/clip-vit-large-patch14`            | `SD1.4_safeguider.pt` | 768   |
| SD-V2.1      | `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`*   | `SD2.1_safeguider.pt` | 1024  |
| Flux.1       | `google/t5-v1_1-xxl` (encoder only)*       | `Flux_safeguider.pt`  | 4096  |

\* Với SD-V2.1 và Flux, encoder của HuggingFace có format khác CLIP-L; bạn
có thể cần lưu file embedding offline rồi gọi train.py thay vì pipeline
real-time. Test kỹ trước khi production.
