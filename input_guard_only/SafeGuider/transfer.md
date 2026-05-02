# SafeGuider — Tài liệu chuyển giao (transfer.md)

Tài liệu này tóm tắt toàn bộ pipeline, phương pháp (method), cấu trúc code (mỗi folder làm gì và quan hệ giữa chúng), và hướng dẫn vận hành (chuẩn bị dữ liệu, train recognizer, sinh ảnh, đánh giá). Mục tiêu: một người mới vào dự án có thể đọc xong và tự reproduce được kết quả.

---

## 1. Pipeline & Method

### 1.1. Bài toán
Các mô hình text-to-image (T2I) như Stable Diffusion (SD-V1.4, SD-V2.1) hoặc Flux.1 dễ bị lạm dụng để sinh ra nội dung không an toàn (NSFW: sexual, violence, hate, …). Các phương pháp phòng vệ phổ biến (safety checker output, blacklist từ khóa, negative prompt, retraining) hoặc làm giảm chất lượng ảnh, hoặc từ chối hoàn toàn (degraded UX), hoặc dễ bị bypass bằng adversarial prompt (VS Attack, SJ Attack, MMA, P4D, META…).

### 1.2. Idea cốt lõi của SafeGuider
SafeGuider hoạt động ở **mức embedding** của text encoder, không sửa text prompt theo từ điển và không sửa diffusion model. Hai thành phần chính:

1. **Embedding-level Recognizer (Safety Classifier)**: Một MLP nhỏ (3 tầng) nhận `EOS token embedding` từ text encoder của T2I model làm input, output xác suất {unsafe=0, safe=1}. Quan sát empirical là EOS token "tổng hợp" (aggregator) toàn bộ ngữ nghĩa của prompt nhờ cơ chế attention trong CLIP/T5; do đó NSFW prompt và benign prompt phân tách rõ ở không gian embedding của EOS token.
2. **Safety-aware Beam Search**: Khi recognizer phán prompt không an toàn, thay vì từ chối generate, SafeGuider **xóa từng từ** (token-level removal) trong prompt theo chiến lược beam search có điều kiện kép:
   - Maximize **safety score** (probability của lớp safe từ recognizer).
   - Maintain **semantic similarity** với embedding gốc (cosine similarity) ≥ ngưỡng (mặc định 0.1) để giữ phần ngữ nghĩa benign.
3. Embedding của prompt đã chỉnh sửa được đưa vào U-Net + sampler (DDIM/PLMS) để sinh ảnh an toàn nhưng vẫn có ý nghĩa.

### 1.3. Chi tiết thuật toán beam search
Tham chiếu code: `stable-diffusion-1.4/scripts/safeguider_gene.py:436-540`.

Pseudo-code:
```
input: prompt P = [w_1, ..., w_n], recognizer C, encoder E, beam_width=6, max_depth=min(25, n-1)
e_orig = EOS_emb(E(P)); s_orig = C(e_orig)
if argmax(C(e_orig)) == safe: return P  # bỏ qua, dùng prompt gốc

# Bước 1: ranking từng từ theo "đóng góp vào bất an"
for idx in 1..n:
    e_idx = EOS_emb(E(P \ w_idx))
    impact[idx] = safety(e_idx) - safety(e_orig)
sort impact desc

# Bước 2: beam search
candidates = [([], 0, 1.0)]   # (removed_indices, safety_improvement, similarity)
for depth in 1..max_depth:
    new_candidates = []
    qualified    = []
    for (R, _, _) in candidates:
        for (idx, _) in impact:
            if idx in R: continue
            P' = P with tokens R + [idx] removed
            e' = EOS_emb(E(P'))
            s' = safety(e'); sim = cos(e_orig, e')
            new_candidates.append((R+[idx], s'-s_orig, sim, s'))
            if s' >= 0.80 and sim >= 0.1:
                qualified.append((R+[idx], s'-s_orig, sim))
                update best if (improvement higher) or (same improvement, fewer removals)
    candidates = top-k(qualified) if qualified else top-k(new_candidates by safety)
    if best safety_score >= 0.80: break

if no best:
    fallback = max(s' among candidates with sim >= 0.1)
return modified prompt
```
Hai ngưỡng quan trọng (hard-coded trong file):
- `safety threshold = 0.80` (xem `safeguider_gene.py:506,537`)
- `similarity floor = 0.1` (xem `safeguider_gene.py:506,553`)

### 1.4. Sơ đồ pipeline đầy đủ (inference)
```
prompt
  │
  ▼
Tokenizer (CLIP)  ──►  input_ids ──►  Text Encoder (FrozenCLIP / OpenCLIP / T5)
                                          │
                                          ▼
                              EOS token embedding (768/1024/4096-d)
                                          │
                                          ▼
                       Recognizer (ThreeLayerClassifier 768→1024→512→2)
                                          │
                          ┌───────────────┴───────────────┐
                          │                               │
                   safe (cls=1)                      unsafe (cls=0)
                          │                               │
                          │              Safety-aware Beam Search
                          │           (token removal + semantic similarity)
                          │                               │
                          ▼                               ▼
                  use original embedding         use modified embedding
                          │                               │
                          └───────────────┬───────────────┘
                                          ▼
                               U-Net (LatentDiffusion)
                                + DDIM/PLMS sampler
                                          │
                                          ▼
                              VAE decoder (first_stage_model)
                                          │
                                          ▼
                                   Watermark + save PNG
```

---

## 2. Cấu trúc code chi tiết

```
SafeGuider/
├── recognizer.py                # CLI standalone: chỉ chạy classifier để check 1 prompt
├── README.md
├── environment.yml              # Conda env (tên 'safeguider')
├── LICENSE
│
├── asset/
│   └── framework.png            # Hình minh họa kiến trúc (dùng trong README)
│
├── Models/                      # Pretrained recognizer weights cho 3 backbone
│   ├── SD1.4_safeguider.pt        # CLIP ViT-L/14, dim=768
│   ├── SD2.1_safeguider.pt        # OpenCLIP ViT-H/14, dim=1024
│   └── Flux_safeguider.pt         # T5-XXL, dim=4096
│
├── stable-diffusion-1.4/        # Codebase SD-V1.4 (fork CompVis/stable-diffusion)
│   ├── checkpoint/                  # ❗ User tự đặt sd-v1-4-full-ema.ckpt vào đây
│   ├── configs/
│   │   ├── stable-diffusion/
│   │   │   ├── v1-inference.yaml          # config chính cho txt2img
│   │   │   └── v1-inpainting-inference.yaml
│   │   ├── autoencoder/, latent-diffusion/, retrieval-augmented-diffusion/
│   ├── ldm/                         # Core LatentDiffusion library
│   │   ├── models/
│   │   │   ├── diffusion/{ddpm.py, ddim.py, plms.py, classifier.py}
│   │   │   └── autoencoder.py             # VAE (first_stage)
│   │   ├── modules/
│   │   │   ├── encoders/modules.py        # FrozenCLIPEmbedder (cond_stage)
│   │   │   ├── diffusionmodules/{model.py, openaimodel.py, util.py}
│   │   │   ├── attention.py, ema.py, x_transformer.py
│   │   │   ├── losses/, distributions/, image_degradation/
│   │   ├── data/                          # Dataset wrappers (lsun, imagenet, base)
│   │   ├── lr_scheduler.py, util.py       # instantiate_from_config sống ở đây
│   ├── scripts/
│   │   ├── original_gene.py               # SD-V1.4 gốc (để so sánh)
│   │   ├── safeguider_gene.py             # ★ SafeGuider inference (recognizer + beam search)
│   │   ├── adaptive_ori_gene.py           # Adaptive attack baseline
│   │   ├── img2img.py, inpaint.py, knn2img.py, sample_diffusion.py, train_searcher.py
│   │   └── tests/test_watermark.py
│   ├── tools/
│   │   ├── classifier.py                  # ★ Định nghĩa các MLP (1/3/5/7/9 layer)
│   │   ├── json2embedding.py              # ★ Tạo dataset (prompt → EOS embedding)
│   │   └── train.py                       # ★ Huấn luyện recognizer
│   ├── data/index_synset.yaml
│   ├── models/                            # Config cho LDM con (kế thừa từ CompVis)
│   ├── main.py, notebook_helpers.py, setup.py, environment.yaml
│   └── README.md, Stable_Diffusion_v1_Model_Card.md
│
└── Emperical_Study/             # Phân tích empirical (Section 3 trong paper)
    ├── SAC.py                       # Semantic Attention Concentration (entropy + SAC)
    ├── Aggregator_ratio.py          # Tỉ lệ EOS làm Top-1 aggregator
    ├── MMD.py                       # Maximum Mean Discrepancy giữa benign & adv
    ├── Visualization.py             # t-SNE / UMAP / PCA / 3D PCA
    └── embedding/                   # Embedding sample đính kèm
        ├── benign_testend.json
        ├── nsfw_meta_testend.json
        └── nsfw_mma_testend.json
```

### 2.1. Quan hệ giữa các folder

- **`Models/` ↔ `recognizer.py` ↔ `stable-diffusion-1.4/scripts/safeguider_gene.py`**
  Cả hai entry point đều load `.pt` từ `Models/`. `recognizer.py` chỉ cần text encoder + tokenizer (folder `stable_diffusion_clip/{tokenizer,text_encoder}` mà user phải tải về). `safeguider_gene.py` tải cả LatentDiffusion model, dùng `model.cond_stage_model` (chính là `FrozenCLIPEmbedder` định nghĩa ở `ldm/modules/encoders/modules.py`) làm text encoder để lấy embedding.

- **`stable-diffusion-1.4/ldm/` là backbone**: tất cả script trong `scripts/` đều `from ldm.util import instantiate_from_config` và load checkpoint qua `OmegaConf.load(config)`. Config `v1-inference.yaml` khai báo:
  - `cond_stage_config`: `ldm.modules.encoders.modules.FrozenCLIPEmbedder` (text encoder).
  - `first_stage_config`: `ldm.models.autoencoder.AutoencoderKL` (VAE).
  - `unet_config`: `ldm.modules.diffusionmodules.openaimodel.UNetModel`.
  - Top-level: `ldm.models.diffusion.ddpm.LatentDiffusion`.

- **`tools/classifier.py`** định nghĩa kiến trúc MLP (chỉ `ThreeLayerClassifier` được dùng cho `.pt` đi kèm: `Linear(d→1024) → ReLU → Dropout(0.5) → Linear(1024→512) → ReLU → Dropout(0.5) → Linear(512→2) → Softmax`).
  - `tools/train.py` import từ `tools.classifier` và đọc dataset `*.json` (do `tools/json2embedding.py` tạo) để train.
  - `scripts/safeguider_gene.py` cũng import `from tools.classifier import *` để khôi phục model rồi load `.pt`.

- **`tools/json2embedding.py`** đóng vai trò "extractor": load LatentDiffusion (qua `instantiate_from_config`), gọi `model.get_learned_conditioning([prompt])` để có tensor `(1, 77, 768)`, trích slice tại EOS position (`tokens == 49407`), dump JSON `{id, prompt, embedding, label, eos_position}`. Dataset đầu ra của file này chính là input của `train.py`.

- **`Emperical_Study/`** là code phục vụ section phân tích trong paper, **độc lập** với pipeline inference. Nó dùng trực tiếp `transformers.CLIPTextModel` (không qua `ldm`), input là JSON prompt thô. Chú ý các path trong các file ở folder này là **placeholder** (`../../Datasets/...`, `../../Models/stable-diffusion-v1-4/...`, `../../Metrics/...`, `../../Results/...`) — phải sửa lại trước khi chạy.

### 2.2. Mạch dữ liệu trong `safeguider_gene.py` (file quan trọng nhất)
1. `load_model_from_config` → instantiate `LatentDiffusion` + load weight `sd-v1-4-full-ema.ckpt`.
2. `get_embedding_dim(model)` → encode "test" để dò embedding dim (768 cho SD-V1.4). Lập `ThreeLayerClassifier(dim)`, `load_state_dict("../../Models/SD1.4_safeguider.pt")`.
3. Với mỗi batch prompt:
   - `c_original = model.get_learned_conditioning(prompts)` → `(B, 77, 768)`.
   - Tokenize bằng `model.cond_stage_model.tokenizer`, tìm `eos_position` (token id `49407`).
   - `original_eos_embedding = c_original[:1, eos_position, :]` → forward classifier.
   - Nếu `argmax == 1` (safe): dùng `c_original` thẳng.
   - Nếu unsafe: tính `token_impacts` cho từng từ, chạy beam search như mô tả ở §1.3 → ra `modified_prompts`. Encode lại `c = model.get_learned_conditioning(modified_prompts)`.
4. Sampler (`DDIMSampler` hoặc `PLMSSampler` nếu `--plms`) chạy 50 step với `unconditional_guidance_scale=7.5`.
5. `model.decode_first_stage(samples)` → numpy → `Image.fromarray` → `put_watermark` (DWT-DCT invisible WM) → `samples/{base_count:05}_round{n}_prompt{idx}_image{img_idx}.png`.

### 2.3. `recognizer.py` (standalone)
- Tự load tokenizer + `CLIPTextModel` từ `stable_diffusion_clip/{tokenizer,text_encoder}` (path **hard-code** trong class, dòng 36-39).
- Forward CLIP, lấy `last_hidden_state[:, eos_pos, :]`, classify bằng `Models/SD1.4_safeguider.pt`.
- Output: predicted class, safety score, "SAFE/UNSAFE" verdict (ngưỡng `safety_score <= 0.5`).

---

## 3. Hướng dẫn sử dụng

> Lưu ý đường dẫn trong code đa phần là **relative**, nên tất cả các lệnh dưới đều giả định `cwd` đúng như chỉ định.

### 3.1. Cài đặt môi trường
```bash
git clone <repo>
cd SafeGuider
conda env create -f environment.yml          # tạo env tên 'safeguider'
conda activate safeguider
```
Nếu `environment.yml` xung đột phiên bản (CUDA / torch), tham chiếu nhanh các package then chốt: `pytorch>=1.12`, `pytorch-lightning`, `omegaconf`, `transformers`, `diffusers`, `einops`, `imwatermark`, `safetensors`, `taming-transformers`, `clip` (OpenAI), `umap-learn`, `seaborn`, `scikit-learn`.

### 3.2. Tải model weight
1. **SD-V1.4 checkpoint**: tải `sd-v1-4-full-ema.ckpt` từ HuggingFace `CompVis/stable-diffusion-v-1-4-original` rồi đặt vào:
   ```
   SafeGuider/stable-diffusion-1.4/checkpoint/sd-v1-4-full-ema.ckpt
   ```
2. **Recognizer weight**: đã có sẵn trong `Models/SD1.4_safeguider.pt` (cùng `SD2.1_safeguider.pt`, `Flux_safeguider.pt`).
3. **(chỉ cho `recognizer.py`)** tải bộ `tokenizer/` và `text_encoder/` của SD-V1.4 (từ `CompVis/stable-diffusion-v1-4`) vào:
   ```
   SafeGuider/stable_diffusion_clip/{tokenizer, text_encoder}
   ```

### 3.3. Inference: sinh ảnh với SafeGuider (SD-V1.4)
```bash
cd SafeGuider/stable-diffusion-1.4/scripts
```

Một prompt:
```bash
CUDA_VISIBLE_DEVICES=0 python safeguider_gene.py \
    --prompt "a painting of a virus monster playing guitar" \
    --ckpt ../checkpoint/sd-v1-4-full-ema.ckpt \
    --config ../configs/stable-diffusion/v1-inference.yaml \
    --precision full \
    --outdir ./outputs/safeguider_demo
```

Batch từ file JSON (mảng các object có field `prompt`):
```bash
CUDA_VISIBLE_DEVICES=0 python safeguider_gene.py \
    --from-file /path/to/prompts.json \
    --ckpt ../checkpoint/sd-v1-4-full-ema.ckpt \
    --config ../configs/stable-diffusion/v1-inference.yaml \
    --precision full \
    --outdir ./outputs/safeguider_batch
```

Các flag hữu ích:
- `--n_samples N` (batch size, default 3) → số ảnh cho mỗi prompt
- `--n_iter K` (default 1) → lặp lại K lần (đa dạng seed nội bộ)
- `--ddim_steps 50`, `--scale 7.5`, `--seed 42`
- `--plms` để dùng PLMS thay DDIM
- `--H 512 --W 512 --C 4 --f 8` (latent shape — không nên đổi với SD-V1.4)
- `--fixed_code` dùng chung start noise giữa các sample
- `--skip_grid`, `--skip_save` cho benchmark

Output: ảnh PNG tại `<outdir>/samples/{base_count:05}_round{n}_prompt{idx}_image{i}.png`.

### 3.4. Sinh ảnh bằng SD-V1.4 gốc (không SafeGuider) — để đối chứng
```bash
CUDA_VISIBLE_DEVICES=0 python original_gene.py \
    --prompt "your_prompt" \
    --ckpt ../checkpoint/sd-v1-4-full-ema.ckpt \
    --config ../configs/stable-diffusion/v1-inference.yaml \
    --precision full \
    --outdir ./outputs/original_demo
```

### 3.5. Chạy recognizer độc lập (chỉ phân loại an toàn)
```bash
cd SafeGuider                              # cwd phải là root để các path 'Models/...' và 'stable_diffusion_clip/...' đúng
python recognizer.py --prompt "your text prompt here"
```
Kết quả: predicted class (0=unsafe, 1=safe), safety score, verdict.

### 3.6. Train recognizer trên dữ liệu của bạn

#### Bước 1 — sinh embedding dataset từ prompt JSON
1. Chuẩn bị file `prompts.json` dạng:
   ```json
   [
     {"prompt": "a cat in a hat"},
     {"prompt": "violent scene with blood"}
   ]
   ```
2. Mở `stable-diffusion-1.4/tools/json2embedding.py`, sửa các giá trị hard-coded ở `main()`:
   - `config_path` (mặc định `../configs/stable-diffusion/v1-inference.yaml`)
   - `checkpoint_path` (`../checkpoint/sd-v1-4-full-ema.ckpt`)
   - `prompts_file = "path_to_your_dataset"` → trỏ đến `prompts.json`
   - `dataset_path = os.path.join(save_dir, "path_to_save_dataset")` → đặt tên file output, ví dụ `embed_dataset/train_embed.json`
3. Mặc định `label = 0` cho mọi sample (xem dòng 74). Bạn cần **chạy 2 lần** (một lần cho prompts an toàn rồi đổi `label=1`, một lần cho prompts không an toàn giữ `label=0`), sau đó merge thủ công vào một file `{"data": [...]}` rồi truyền cho `train.py`. Hoặc sửa code để đọc nhãn trực tiếp từ field `label` của input.
4. Chạy:
   ```bash
   cd SafeGuider/stable-diffusion-1.4/tools
   python json2embedding.py
   ```

#### Bước 2 — train classifier
1. Mở `stable-diffusion-1.4/tools/train.py`, sửa:
   - Dòng 195: `dim=768` (sửa thành 1024 cho SD-V2.1, 4096 cho Flux T5-XXL)
   - Dòng 200: `train_json_file = 'path_to_your_training_data'` → trỏ đến file embedding ở bước trên
   - Dòng 162: `best_model_path = './stable-diffusion-1.5/class_model/test_loss.pt'` → đổi tên/đường dẫn lưu
   - Dòng 158: số epoch (mặc định 50)
   - Dòng 196: optimizer/lr/momentum (mặc định SGD, lr=1e-3, momentum=0.9)
   - Dòng 225-227: batch size 32
2. Chạy:
   ```bash
   cd SafeGuider/stable-diffusion-1.4
   python -m tools.train          # cần module path đúng cho `from tools.classifier import *`
   ```
   Hoặc:
   ```bash
   cd SafeGuider/stable-diffusion-1.4/tools
   PYTHONPATH=.. python train.py
   ```

> Note: trong `train.py` có sẵn cả `custom_loss` (margin-based an toàn) và `nn.CrossEntropyLoss`; mặc định pipeline dùng `CrossEntropyLoss` (dòng 62, 74). Nếu muốn margin loss thì thay `criterion(logits, target)` bằng `custom_loss(logits, probs, target)`.

#### Bước 3 — đưa weight đã train vào pipeline inference
- Đặt file `.pt` mới vào `Models/` (ví dụ `Models/MyClassifier.pt`).
- Trong `safeguider_gene.py:288`, sửa path checkpoint của classifier về file của bạn.
- Trong `recognizer.py:39`, sửa `self.classifier_path` tương tự.

### 3.7. Empirical analysis (tùy chọn)
Các script ở `Emperical_Study/` không thuộc đường dẫn inference chính, nhưng giúp tái hiện các số liệu phân tích trong paper.

Trước khi chạy, sửa lại path data (các file đều dùng path tương đối tới folder không có sẵn trong repo):
- `SAC.py:14-15` — text encoder & tokenizer (cần tải về `Models/stable-diffusion-v1-4/{text_encoder,tokenizer}`).
- `SAC.py:24-25`, `Aggregator_ratio.py:23-24` — `../../Datasets/Benign/coco2017-2k.json` & `../../Datasets/NSFW/sexual/P4D.json`.
- `MMD.py`, `Visualization.py` — các file `*_testend.json` đã có ví dụ trong `Emperical_Study/embedding/`.

Lệnh chạy mẫu:
```bash
cd SafeGuider/Emperical_Study
python SAC.py                  --output_file results_sac.json     # entropy & SAC theo layer
python Aggregator_ratio.py                                        # tỉ lệ EOS = Top-1 aggregator
python MMD.py                                                     # MMD benign vs VS/SJ attack
python Visualization.py                                           # t-SNE / UMAP / PCA visual
```

### 3.8. Áp dụng cho SD-V2.1 hoặc Flux.1
Repo chỉ ship đầy đủ codebase của SD-V1.4, nhưng cung cấp recognizer pretrained cho 2 backbone khác:
- **SD-V2.1** (OpenCLIP ViT-H/14, dim=1024) → `Models/SD2.1_safeguider.pt`
- **Flux.1** (T5-XXL, dim=4096) → `Models/Flux_safeguider.pt`

Để dùng SafeGuider trên 2 backbone này, bạn cần:
1. Cài codebase SD-V2.1 (`Stability-AI/stablediffusion`) hoặc Flux (`black-forest-labs/flux`) song song.
2. Trong code inference của backbone đó, ngay sau khi có `prompt_embeds = text_encoder(...)`:
   - Cắt EOS embedding (`prompt_embeds[:, eos_pos, :]`).
   - Build `ThreeLayerClassifier(dim=1024 hoặc 4096)`, load `.pt` tương ứng.
   - Forward → nếu `argmax==0`, chạy beam search như `safeguider_gene.py:436-572` (chỉ thay `model.get_learned_conditioning` bằng hàm encode tương ứng).
   - Replace `prompt_embeds` bằng phiên bản đã chỉnh sửa, rồi tiếp tục pipeline gốc.

---

## 4. Một số điểm cần biết khi chuyển giao

- **Threshold cứng**: `0.80` (safety) và `0.1` (similarity) ở `safeguider_gene.py` là kết quả tuning thủ công, có thể cần điều chỉnh nếu đổi recognizer/dataset. `recognizer.py` dùng ngưỡng khác (`0.5`).
- **EOS token id = 49407** áp dụng cho CLIP-L/14 và OpenCLIP-H/14. Với T5 (Flux) sẽ khác — phải đổi lại.
- **Path hard-coded** xuất hiện ở: `recognizer.py:36-39`, `safeguider_gene.py:25,143,288`, `tools/json2embedding.py:100-101,114,123`, `tools/train.py:162,200`, và toàn bộ `Emperical_Study/*.py`. Khi onboarding, đây là chỗ dễ vỡ nhất.
- **Beam width = 6, max_depth = min(25, n-1)** (n = số từ trong prompt). Với prompt rất dài, complexity là `O(n × beam_width × max_depth)` lượt encode → sẽ chậm.
- **`tokenizer.encode`** trả về list số nguyên (đã có BOS+EOS), không phải dict; nên cần `torch.tensor(...).squeeze(0)`. Đây là khác biệt với `tokenizer(...)` (HuggingFace style) ở `recognizer.py`.
- **Watermark**: SafeGuider giữ DWT-DCT invisible watermark "StableDiffusionV1" của SD gốc; có thể tắt bằng cách bỏ `put_watermark`.
- **`from-file` JSON format**: phải là *list of dict* có field `prompt` (xem `safeguider_gene.py:313-318`); KHÁC với format `{"data": [...]}` của embedding dataset ở `train.py`.
- **Safety checker của diffusers** được import nhưng KHÔNG được dùng trong `safeguider_gene.py` (xem dòng 22-23) — recognizer của SafeGuider thay thế hoàn toàn nó.
- **`Emperical_Study/embedding/*.json`** đã sẵn dữ liệu mẫu để chạy `MMD.py` và `Visualization.py` mà không cần tải thêm.

---

## 5. Checklist tái hiện kết quả nhanh (smoke test)

```bash
# 1. env + checkpoint
conda activate safeguider
ls stable-diffusion-1.4/checkpoint/sd-v1-4-full-ema.ckpt   # phải tồn tại
ls Models/SD1.4_safeguider.pt                               # phải tồn tại

# 2. test recognizer standalone
cd SafeGuider
python recognizer.py --prompt "a serene mountain landscape"     # kỳ vọng SAFE
python recognizer.py --prompt "<một prompt NSFW test>"          # kỳ vọng UNSAFE

# 3. test inference 1 prompt
cd stable-diffusion-1.4/scripts
CUDA_VISIBLE_DEVICES=0 python safeguider_gene.py \
    --prompt "a serene mountain landscape" \
    --ckpt ../checkpoint/sd-v1-4-full-ema.ckpt \
    --config ../configs/stable-diffusion/v1-inference.yaml \
    --precision full --outdir ./outputs/smoke --n_samples 1
ls ./outputs/smoke/samples/                                  # phải có *.png
```

Hết.
