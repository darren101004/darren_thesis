# SafeGuider — Input Guard Only

Repo độc lập tách riêng phần **input guard** của SafeGuider:

1. **Classify** — phán 1 prompt là SAFE / UNSAFE bằng safety recognizer
   (CLIP text encoder + 1 MLP nhỏ).
2. **Beam-search rewrite** — khi prompt UNSAFE, tự động xóa từ trong prompt
   theo beam search có ràng buộc (safety + cosine similarity) để có 1
   prompt mới SAFE nhưng vẫn giữ ngữ nghĩa.

Folder này **không phụ thuộc** vào codebase Stable Diffusion / `ldm/` của
repo gốc. Bạn có thể copy nguyên folder ra và dùng như 1 micro-service
text-to-text: input là prompt, output là verdict + (tuỳ chọn) prompt
rewrite. Sau đó bạn tự đẩy text này vào bất kỳ T2I model nào (SD, Flux,
Midjourney API, …).

> Tham chiếu: `transfer.md` ở thư mục cha mô tả pipeline đầy đủ của
> SafeGuider gốc. File này chỉ tài liệu hoá phần input guard.

---

## 1. Cấu trúc

```
input_guard_only/
├── README.md                ← bạn đang đọc
├── requirements.txt
├── __init__.py              ← cho phép `import input_guard_only as ig`
│
├── encoder.py               ← CLIPEncoder (local-first load, tokenizer + EOS slice)
├── classifier.py            ← MLP architectures (ThreeLayerClassifier...)
├── recognizer.py            ← Standalone classify (có CLI)
├── beam_search.py           ← SafetyAwareBeamSearch (rewrite logic)
├── pipeline.py              ← Full pipeline classify + rewrite (có CLI)
│
├── prepare_embeddings.py    ← Build dataset cho training (text → EOS embedding JSON)
├── train.py                 ← Train classifier
│
├── weights/                 ← cả classifier .pt VÀ text-encoder folder đều ở đây
│   ├── README.md            ← hướng dẫn lấy weight
│   ├── SD1.4_safeguider.pt  ← classifier weight (copy từ Models/ của repo gốc)
│   └── clip-vit-large-patch14/   ← auto-download lần đầu chạy (text encoder)
└── examples/
    ├── prompts_demo.json
    └── run_demo.sh
```

---

## 2. Cài đặt

```bash
cd input_guard_only
python -m venv .venv && source .venv/bin/activate     # hoặc dùng conda
pip install -r requirements.txt
```

Lấy weight (xem chi tiết `weights/README.md`):
```bash
cp ../Models/SD1.4_safeguider.pt weights/
```

**Local-first encoder load.** Lần đầu chạy, `encoder.py` sẽ tự
`snapshot_download` `openai/clip-vit-large-patch14` (~600 MB) **vào folder local**
`weights/clip-vit-large-patch14/` (KHÔNG dùng HuggingFace cache mặc định ở
`~/.cache/huggingface/`). Các lần sau detect folder đó, load offline luôn.

Logic resolve trong `CLIPEncoder.__init__`:
  1. Nếu `--encoder-model` là 1 local folder hợp lệ → load trực tiếp.
  2. Nếu `weights/<basename>/` đã có → load offline.
  3. Ngược lại → download từ HF về `weights/<basename>/` rồi load.

Có thể đổi cache dir qua Python API: `CLIPEncoder(cache_dir="/some/path")`,
hoặc ép re-download: `CLIPEncoder(force_download=True)`.

Quick smoke test:
```bash
CUDA_VISIBLE_DEVICES=4 python recognizer.py --prompt "a serene mountain landscape"
```

---

## 3. Cách dùng — 4 chế độ

### 3.1. Standalone classify (CLI)
File `recognizer.py` chỉ làm classification, không bao giờ rewrite. Nhẹ,
dùng để integrate vào filter pipeline có sẵn.

```bash
# 1 prompt
python recognizer.py --prompt "a violent gory war scene"

# Batch từ JSON
python recognizer.py \
    --from-file examples/prompts_demo.json \
    --output classify_results.json

# Đổi weight / encoder
python recognizer.py \
    --prompt "..." \
    --weights weights/SD2.1_safeguider.pt \
    --encoder-model laion/CLIP-ViT-H-14-laion2B-s32B-b79K

# Bật log encoder để xem số token + shape embedding + dim vào classifier
python recognizer.py --verbose --prompt "a violent gory war scene"
```

Output mẫu:
```
prompt   : a violent gory war scene
class    : 0 (unsafe)
score    : 0.0143  (P[unsafe]=0.9857, P[safe]=0.0143)
verdict  : UNSAFE
```

Khi thêm `--verbose`, encoder in thêm trace:
```
[encoder] loaded 'openai/clip-vit-large-patch14' from .../weights/clip-vit-large-patch14 | device=cuda dtype=torch.float32 | hidden_size=768 max_length=77 eos_token_id=49407
[encoder] prompt[0]='a violent gory war scene' | raw_tokens=7 -> tokens=7 (pad=70, truncated=0) | eos_pos=6
[encoder] shapes: last_hidden_state=(1, 77, 768) -> eos_embedding=(1, 768) (classifier input dim=768)
```

Trường hợp prompt dài bị cắt:
```
[encoder] prompt[0]='A grotesque, pus-oozing, festering wound, a gaping, putri...' | raw_tokens=156 -> tokens=77 (pad=0, truncated=79) TRUNCATED! | eos_pos=76
```

Ý nghĩa các trường:
- `raw_tokens=R` = số token tokenizer SINH RA từ prompt gốc (đã gồm BOS + EOS), CHƯA truncate.
- `tokens=N` = số token THỰC sau khi truncate về `max_length=77`. Khi `raw > 77`, content bị cắt từ đuôi và EOS được giữ ở slot cuối → `tokens=77, pad=0`.
- `truncated=T` = `max(0, raw - 77)`. Nếu T > 0 thì có cảnh báo `TRUNCATED!` đỏ trong log → ngữ nghĩa cuối prompt đã mất.
- `pad=M` = số PAD lấp đầy tới `max_length=77` (CLIP có `pad_token_id == eos_token_id == 49407`).
- `eos_pos=K` = vị trí EOS (zero-indexed) — slice tại đó để lấy vector cho classifier.
- `eos_embedding=(B, D)` = vector duy nhất đưa vào ThreeLayerClassifier (D = `hidden_size`).

### 3.2. Standalone classify (Python)
```python
from recognizer import PromptRecognizer

rec = PromptRecognizer(weights="weights/SD1.4_safeguider.pt")
out = rec.classify("a violent gory war scene")
# {"prompt": "...", "predicted_class": 0, "safety_score": 0.0143,
#  "probabilities": [0.9857, 0.0143], "is_safe": False}

# Batch:
batch = rec.classify_batch(["a cat", "violent gory ...", "..."])
```

### 3.3. Full pipeline = classify + rewrite (CLI)
File `pipeline.py` là entry point chính, có **2 mode** chọn qua flag `--mode`:

| `--mode`    | Hành vi                                                                                  | Tốc độ        |
|-------------|-------------------------------------------------------------------------------------------|---------------|
| `classify`  | Chỉ phán SAFE/UNSAFE (giống `recognizer.py`). 1 lần encode + 1 lần forward classifier.    | ~0.05–0.1s    |
| `full`      | Classify; nếu UNSAFE thì chạy beam-search rewrite. Default.                                | ~1–3s/prompt  |

```bash
# Default: full mode (classify -> rewrite nếu unsafe)
python pipeline.py --mode full --prompt "a violent gory war scene"

# Chỉ classify (giống recognizer.py — tiện khi muốn 1 entry point)
python pipeline.py --mode classify --prompt "..."

# Force rewrite kể cả prompt SAFE (debug)
python pipeline.py --mode full --prompt "..." --force-rewrite

# Batch + lưu JSON + verbose
python pipeline.py \
    --mode full \
    --from-file examples/prompts_demo.json \
    --output guard_results.json \
    --verbose
```

Output mẫu (mode=full, prompt UNSAFE):
```
original : a violent gory war scene
class    : 0 (safe=False)  score=0.0143
rewrite  : a war scene
removed  : ['violent', 'gory']
modified : safety=0.9621  sim=0.6884
elapsed  : 1.83s
```

**Flag `--verbose`** (cùng tên cho cả `recognizer.py` và `pipeline.py`):
- Encoder log: in `tokens / pad / eos_pos / shape last_hidden_state / shape eos_embedding / classifier input dim` mỗi lần `encode()`.
- Beam-search trace (chỉ `pipeline.py` ở `--mode full`): thêm field `beam_search_log` vào output JSON, ghi từng depth của beam search.
- ⚠️ Với `--mode full`, encoder sẽ in **vài nghìn dòng** vì beam search gọi `encode()` ~`O(N × beam_width × max_depth)` lần. Chỉ nên bật khi debug 1 prompt; với batch nên tắt.

Hyper-params beam search (mặc định khớp paper):
| flag                  | default | ý nghĩa                                                             |
|-----------------------|---------|----------------------------------------------------------------------|
| `--beam-width`        | 6       | số candidate giữ lại mỗi depth                                       |
| `--max-depth`         | 25      | số từ tối đa được phép xóa                                           |
| `--safety-threshold`  | 0.80    | P[safe] tối thiểu để 1 candidate qualify                             |
| `--similarity-floor`  | 0.10    | cosine similarity tối thiểu (so với EOS embedding gốc) để qualify    |

### 3.4. Full pipeline (Python)
```python
from pipeline import SafeGuiderInputGuard

guard = SafeGuiderInputGuard(
    weights="weights/SD1.4_safeguider.pt",
    safety_threshold=0.80,
    similarity_floor=0.10,
    beam_width=6,
    max_depth=25,
    verbose=False,
)

# Single prompt
out = guard.process("a violent gory war scene")
# Keys: original_prompt, predicted_class, safety_score, is_safe,
#       was_modified, modified_prompt, removed_tokens,
#       modified_safety, similarity, elapsed_sec.
final_prompt = out["modified_prompt"]   # luôn là text an toàn để đẩy vào T2I

# Batch
results = guard.process_batch(["a cat", "...nsfw...", "..."])

# Cũng có thể chỉ classify, không rewrite
verdict = guard.classify("...")
```

---

## 4. Pipeline chi tiết

```
prompt
  │
  ▼
CLIPTokenizer (max_length=77, padding="max_length")
  │
  ▼
CLIPTextModel.last_hidden_state   shape = (B, 77, 768)
  │
  ▼
slice tại vị trí EOS (= lần xuất hiện đầu tiên của token id 49407)
  │
  ▼
EOS embedding   shape = (B, 768)
  │
  ▼
ThreeLayerClassifier  (Linear 768→1024 → ReLU → Dropout(0.5)
                       → Linear 1024→512 → ReLU → Dropout(0.5)
                       → Linear 512→2 → Softmax)
  │
  ┌─ argmax==1 (safe) ─► return ngay
  │
  └─ argmax==0 (unsafe)
        │
        ▼
   SafetyAwareBeamSearch.rewrite(prompt):
     1. Rank impact: xóa từng từ độc lập, đo P[safe] tăng bao nhiêu
     2. Beam search depth max_depth, mỗi bước thử thêm 1 từ để xóa,
        candidate qualified ⇔ P[safe] ≥ 0.80 AND cos_sim ≥ 0.1
     3. Early stop khi tìm được candidate qualified
     4. Fallback: max P[safe] trong các candidate có cos_sim ≥ 0.1
        (nếu không có gì → giữ prompt gốc, was_modified=False)
        │
        ▼
   modified_prompt   ← text đã rewrite
```

Embedding pipeline ở đây **khớp 100%** với `FrozenCLIPEmbedder` của LDM mà
recognizer được train. Cụ thể: cùng `CLIPTokenizer.from_pretrained` +
`CLIPTextModel.from_pretrained`, cùng `padding="max_length"`, KHÔNG
truyền `attention_mask` (vì pad_token_id của CLIP trùng với eos_token_id
= 49407, nên LDM cố tình không pass attention_mask). Nếu bạn dùng
`recognizer.py` ở repo gốc (file `SafeGuider/recognizer.py`), file đó CÓ
truyền attention_mask — về lý thuyết tạo ra embedding hơi khác. Folder
`input_guard_only/` này chọn cách của LDM để khớp với cách train.

---

## 5. Train classifier mới

3 bước: build dataset → train → swap weight.

### Bước 1 — Build embedding dataset
Input: 1 file JSON dạng `[{"prompt": "..."}, ...]` hoặc `["...", ...]`.

```bash
# Encode 1 file benign (nhãn 1 = safe)
python prepare_embeddings.py \
    --input  data/benign_prompts.json \
    --output embed/benign.json \
    --label  1 \
    --batch-size 32

# Encode 1 file unsafe (nhãn 0 = unsafe)
python prepare_embeddings.py \
    --input  data/nsfw_prompts.json \
    --output embed/nsfw.json \
    --label  0

# Merge thành 1 dataset training (default --output = embed/train.json)
python prepare_embeddings.py --merge embed/benign.json embed/nsfw.json
```

> Các default path: `--output` = `embed/embeddings.json` (build mode),
> hoặc `embed/train.json` (merge mode). Ghi đè bằng `--output` nếu cần.

Output format:
```json
{"data": [
  {"id": 0, "prompt": "...", "embedding": [..768 floats..], "label": 1, "eos_position": 7},
  ...
]}
```

### Bước 2 — Train
Khi follow đúng default path layout, lệnh tối thiểu chỉ là:
```bash
python train.py            # đọc embed/train.json, lưu weights/recognizer.pt
```

Đầy đủ:
```bash
python train.py \
    --train  embed/train.json   `# default` \
    --val    embed/val.json     `# optional` \
    --output weights/recognizer.pt `# default` \
    --layers 3 \
    --epochs 50 --batch-size 32 \
    --optimizer sgd --lr 1e-3 --momentum 0.9 \
    --loss ce
```

Các flag chính:
| flag                  | default     | ghi chú                                                       |
|-----------------------|-------------|---------------------------------------------------------------|
| `--layers`            | 3           | 1/3/5/7/9 — paper dùng 3                                     |
| `--epochs`            | 50          |                                                               |
| `--optimizer`         | sgd         | sgd / adam / adamw                                            |
| `--lr`                | 1e-3        |                                                               |
| `--momentum`          | 0.9         | (chỉ với sgd)                                                 |
| `--weight-decay`      | 0.0         |                                                               |
| `--loss`              | ce          | ce = CrossEntropy; margin = margin loss như paper             |
| `--seed`              | 111         |                                                               |

Nếu có `--val`, file `.pt` lưu là best-val-acc; nếu không có, là final epoch.

### Bước 3 — Dùng weight mới
```bash
python pipeline.py --mode full --prompt "..." \
    --weights weights/recognizer.pt
```
Hoặc đổi tên file `.pt` thành `SD1.4_safeguider.pt` để khớp default của
`recognizer.py` / `pipeline.py` và không phải truyền `--weights`.

---

## 6. Đổi sang backbone khác (SD-V2.1, Flux, …)

Kiến trúc `ThreeLayerClassifier(dim=D)` tự động co dãn theo encoder. Bạn
chỉ cần (a) trỏ `--encoder-model` về encoder mong muốn, (b) trỏ
`--weights` về `.pt` đã train cho encoder đó.

```bash
# SD-V2.1 (OpenCLIP ViT-H/14, dim=1024)
python pipeline.py --mode full --prompt "..." \
    --encoder-model laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
    --weights weights/SD2.1_safeguider.pt

# Flux.1 (T5-XXL, dim=4096)
python pipeline.py --mode full --prompt "..." \
    --encoder-model google/t5-v1_1-xxl \
    --weights weights/Flux_safeguider.pt
```

Lưu ý:
- Tokenizer T5 KHÔNG dùng EOS id 49407 — phải truyền thêm `eos_token_id`
  qua API `CLIPEncoder(eos_token_id=...)` hoặc sửa `encoder.py` cho phù
  hợp với T5 (`</s>` = id 1). CLI hiện chưa expose flag này; xem code
  `encoder.py` để mở rộng.
- Recognizer `SD2.1_safeguider.pt` và `Flux_safeguider.pt` được train
  với embedding lấy từ encoder của paper (OpenCLIP / T5-XXL trong codebase
  SD-V2.1 / Flux.1 gốc). Encoder HuggingFace không phải lúc nào cũng
  binary-identical — nếu thấy classify lệch, nên dùng đúng encoder của
  codebase gốc (tải về local rồi `--encoder-model /local/path`).

---

## 7. API tóm tắt

```python
from encoder        import CLIPEncoder, EncodeResult
from classifier     import ThreeLayerClassifier, create_model
from recognizer     import PromptRecognizer
from beam_search    import SafetyAwareBeamSearch, BeamSearchResult
from pipeline       import SafeGuiderInputGuard


# 1) Lower-level: encoder + classifier riêng
enc = CLIPEncoder(model_name="openai/clip-vit-large-patch14")
clf = ThreeLayerClassifier(dim=enc.hidden_size).to(enc.device).eval()
clf.load_state_dict(torch.load("weights/SD1.4_safeguider.pt", map_location=enc.device))

eos = enc.eos_embedding(["a cat"])      # (1, 768)
logits, probs = clf(eos)                # (1, 2)


# 2) Mid-level: classify-only
rec = PromptRecognizer(weights="weights/SD1.4_safeguider.pt")
print(rec.classify("a cat"))


# 3) High-level: full pipeline
guard = SafeGuiderInputGuard(weights="weights/SD1.4_safeguider.pt")
print(guard.process("a violent gory war scene"))


# 4) Beam search trực tiếp (nếu đã có encoder + classifier)
bs = SafetyAwareBeamSearch(encoder=enc, classifier=clf,
                            safety_threshold=0.80, similarity_floor=0.1,
                            beam_width=6, max_depth=25)
result: BeamSearchResult = bs.rewrite("a violent gory war scene")
print(result.to_dict())
```

---

## 8. Lưu ý vận hành

- **Beam search là sequential per prompt**: complexity ~ `O(N · beam_width
  · max_depth)` lượt encode (N = số từ trong prompt). Với 1 GPU, mỗi
  prompt thường mất 1-3s. Để batch nhanh hơn, gọi `process_batch` (chỉ
  parallelism ở mức I/O, không hợp nhất nhiều prompt vào 1 forward).
- **Threshold 0.80 / 0.10** copy từ `safeguider_gene.py` của repo gốc;
  có thể chỉnh qua flag CLI hoặc kwarg constructor.
- **Verdict `is_safe`** ở `recognizer.py` dùng ngưỡng `safety_score >= 0.5`,
  KHÁC với ngưỡng beam-search 0.80 (mục đích khác nhau: classify vs
  rewrite stop-criterion). Bạn có thể điều chỉnh `safety_threshold` của
  `PromptRecognizer.__init__`.
- **Output là TEXT**: pipeline này KHÔNG sinh ảnh. Sau khi có
  `modified_prompt`, đem text đó đẩy vào bất kỳ T2I model nào.
- **Prompt 1 từ**: bị skip beam search (không thể rewrite); trả về prompt
  gốc với `was_modified=False`.
- **Prompt rỗng**: beam search trả về prompt gốc; classify vẫn chạy bình
  thường.
- **Memory**: text encoder CLIP ViT-L/14 chiếm ~500 MB GPU. Classifier
  3-layer (768→1024→512→2) ~10 MB. Tổng < 1 GB → chạy được trên CPU
  hoặc GPU yếu (chỉ chậm hơn).

---

## 9. Smoke test

```bash
# Sau khi đã copy weight vào weights/SD1.4_safeguider.pt
bash examples/run_demo.sh
```

Script này chạy lần lượt: classify 1 prompt, classify batch JSON, full
pipeline 1 prompt, classify mode của pipeline. Nếu cả 4 bước in ra kết
quả mà không lỗi → setup OK.
