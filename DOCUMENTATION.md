# Team-B-Backend — Full Project Documentation

> **Retinal Disease Classifier** · EfficientNet-B4 · 45 Diseases · FastAPI · PyTorch · React
>
> This document explains **every file**, its **purpose**, and **how every part connects**.
> Intended as the single reference for any developer picking up this project.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Repository Layout — Every File Explained](#3-repository-layout--every-file-explained)
4. [Configuration Layer](#4-configuration-layer)
5. [Machine Learning Layer (Training Pipeline)](#5-machine-learning-layer-training-pipeline)
   - 5.1 [model.py](#51-modelpy)
   - 5.2 [dataset.py](#52-datasetpy)
   - 5.3 [train.py](#53-trainpy)
   - 5.4 [inference.py (standalone CLI)](#54-inferencepy-standalone-cli)
6. [API Layer (Serving Pipeline)](#6-api-layer-serving-pipeline)
   - 6.1 [app/main.py](#61-appmainpy)
   - 6.2 [app/config.py](#62-appconfigpy)
   - 6.3 [app/schemas.py](#63-appschemaspy)
   - 6.4 [app/routers/predict.py](#64-approuterspredicthy)
   - 6.5 [app/services/model_service.py](#65-appservicesmodel_servicepy)
   - 6.6 [app/services/inference_service.py](#66-appservicesinference_servicepy)
   - 6.7 [app/services/advisory_service.py](#67-appservicesadvisory_servicepy)
7. [Frontend Layer (React App)](#7-frontend-layer-react-app)
   - 7.1 [Entry Point & App Shell](#71-entry-point--app-shell)
   - 7.2 [Pages](#72-pages)
   - 7.3 [Components — Layout](#73-components--layout)
   - 7.4 [Components — Prediction](#74-components--prediction)
   - 7.5 [Components — Charts](#75-components--charts)
   - 7.6 [Components — History](#76-components--history)
   - 7.7 [Components — UI Primitives](#77-components--ui-primitives)
   - 7.8 [Services (Frontend)](#78-services-frontend)
   - 7.9 [Hooks](#79-hooks)
   - 7.10 [State Management (Zustand)](#710-state-management-zustand)
   - 7.11 [Utilities](#711-utilities)
8. [Data Flow — Request Lifecycle](#8-data-flow--request-lifecycle)
9. [Data Flow — Training Lifecycle](#9-data-flow--training-lifecycle)
10. [Data Flow — Frontend to Backend](#10-data-flow--frontend-to-backend)
11. [Disease Labels Reference](#11-disease-labels-reference)
12. [Risk Level Logic](#12-risk-level-logic)
13. [API Endpoints Quick Reference](#13-api-endpoints-quick-reference)
14. [Key Design Decisions](#14-key-design-decisions)
15. [Commands Cheat Sheet](#15-commands-cheat-sheet)

---

## 1. What This Project Does

This project is a **full-stack retinal disease screening application** — a React frontend paired with a FastAPI backend powered by a deep learning model.

**User flow:**
1. User opens the web app and goes to the **Analyze** page.
2. They either **drag-and-drop / browse** a fundus image, or **capture one live** using their system camera.
3. After clicking **Start Analysis**, a **retinal scan animation** plays over the preview image.
4. The image is sent to the FastAPI backend, which runs it through **EfficientNet-B4**.
5. The model produces **probability scores for 45 retinal diseases** simultaneously.
6. A **risk level** (`LOW / MODERATE / HIGH / CRITICAL`) is computed and an advisory is generated.
7. Results are displayed: top prediction, confidence bars, bar chart, clinical advisory, and disclaimer.
8. The result is **saved to history** (last 5 predictions, persisted in localStorage).

The model was trained on the **RFMiD dataset** (1,920 fundus images) and achieves a **mean AUC-ROC of 0.82** across 45 classes.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│               BROWSER  (React · Vite · localhost:5173)              │
│                                                                     │
│   Landing → Predict → History → HowItWorks                         │
│                                                                     │
│   ImageUpload (Upload tab | Camera tab)                             │
│      ├─ ScanOverlay  — retinal scan animation while loading         │
│      └─ CameraCapture — live webcam → snapshot → File              │
│                                                                     │
│   usePrediction (TanStack Query mutation)                           │
│      └─ prediction.js → axios → POST /predict                       │
│                                                                     │
│   useHistoryStore (Zustand + localStorage)                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │ HTTP multipart/form-data
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│               FastAPI  (app/main.py · localhost:8000)               │
│  • CORS middleware (allow_origins=["*"])                             │
│  • On startup: ModelService.load() — downloads from HuggingFace     │
│  • Routes → app/routers/predict.py                                  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│              app/routers/predict.py (Route handlers)                │
│  • Validate MIME type, file size, threshold                         │
│  • Call InferenceService → {label: prob} dict + elapsed_ms          │
│  • Call AdvisoryService  → risk level + advisory text               │
│  • Return PredictionResponse (Pydantic)                             │
└───────────┬────────────────────────────┬────────────────────────────┘
            │                            │
            ▼                            ▼
┌───────────────────────┐   ┌────────────────────────────────────────┐
│  inference_service.py │   │         advisory_service.py            │
│  PIL decode           │   │  Count detected diseases               │
│  albumentations       │   │  Identify high-risk ones               │
│  forward pass         │   │  Assign LOW/MODERATE/HIGH/CRITICAL     │
│  sigmoid → probs      │   │  Return advisory text                  │
│  filter by threshold  │   └────────────────────────────────────────┘
└───────────┬───────────┘
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  model_service.py (Singleton)                       │
│  HuggingFace download → EfficientNet-B4 → GPU/CPU memory           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Repository Layout — Every File Explained

```
Team-B-Backend/
│
├── ── BACKEND (Python / FastAPI) ───────────────────────────────────
│
├── app/
│   ├── __init__.py                  ← marks app/ as a Python package
│   ├── main.py                      ← FastAPI app, CORS, lifespan startup hook
│   ├── config.py                    ← ALL API constants (labels, thresholds, metrics)
│   ├── schemas.py                   ← Pydantic response models (typed JSON contracts)
│   ├── routers/
│   │   ├── __init__.py
│   │   └── predict.py               ← HTTP handlers: /predict, /predict-batch, /health, /info
│   └── services/
│       ├── __init__.py
│       ├── model_service.py         ← Singleton: HuggingFace download + load + hold in memory
│       ├── inference_service.py     ← Image decode → preprocess → forward pass → probs
│       └── advisory_service.py     ← Risk scoring + advisory text generation
│
├── model.py                         ← EfficientNet-B4 architecture (shared by training + API)
├── config.py                        ← Training constants (paths, hyperparams, label list)
├── dataset.py                       ← PyTorch Dataset + augmentations + pos_weight calc
├── train.py                         ← Full training loop (AMP, AdamW, CosineAnnealingLR)
├── inference.py                     ← Standalone CLI inference (loads local .pt checkpoint)
│
├── ── FRONTEND (React / Vite) ──────────────────────────────────────
│
├── cnn/retinal-frontend/
│   ├── src/
│   │   ├── main.jsx                 ← React DOM entry point
│   │   ├── App.jsx                  ← Router, QueryClient, Toaster, AnimatePresence
│   │   ├── index.css                ← Tailwind base + CSS variable tokens (light/dark)
│   │   │
│   │   ├── pages/
│   │   │   ├── Landing.jsx          ← Home: stats, features, CTA
│   │   │   ├── Predict.jsx          ← Main analysis page: upload + results grid
│   │   │   ├── History.jsx          ← Last 5 predictions page
│   │   │   ├── HowItWorks.jsx       ← Step-by-step explainer
│   │   │   └── NotFound.jsx         ← 404 page
│   │   │
│   │   ├── components/
│   │   │   ├── layout/
│   │   │   │   ├── Layout.jsx       ← Root shell: Navbar + <Outlet/> + Footer
│   │   │   │   ├── Navbar.jsx       ← Sticky nav, dark mode toggle, mobile menu
│   │   │   │   ├── Footer.jsx       ← Footer with brand + disclaimer
│   │   │   │   └── PageTransition.jsx ← Framer Motion fade wrapper for pages
│   │   │   │
│   │   │   ├── prediction/
│   │   │   │   ├── ImageUpload.jsx  ← Upload/Camera tab panel + submit button
│   │   │   │   ├── ScanOverlay.jsx  ← Retinal scan animation overlay (new)
│   │   │   │   ├── CameraCapture.jsx← Live webcam feed + snapshot capture (new)
│   │   │   │   ├── ResultsPanel.jsx ← Disease results, risk badge, advisory text
│   │   │   │   └── ConfidenceBar.jsx← Animated progress bar per disease
│   │   │   │
│   │   │   ├── charts/
│   │   │   │   └── PredictionChart.jsx ← Recharts bar chart of disease probabilities
│   │   │   │
│   │   │   ├── history/
│   │   │   │   ├── HistoryList.jsx  ← List of past predictions with clear-all button
│   │   │   │   └── HistoryCard.jsx  ← Single history entry: disease, confidence, timestamp
│   │   │   │
│   │   │   └── ui/                  ← Radix UI primitives styled with Tailwind (shadcn-style)
│   │   │       ├── badge.jsx
│   │   │       ├── button.jsx
│   │   │       ├── card.jsx
│   │   │       ├── separator.jsx
│   │   │       ├── skeleton.jsx
│   │   │       └── tooltip.jsx
│   │   │
│   │   ├── services/
│   │   │   ├── api.js               ← Axios instance (baseURL from VITE_API_URL)
│   │   │   └── prediction.js        ← predictImage(), checkHealth(), getModelInfo()
│   │   │                              + transforms backend snake_case → camelCase
│   │   │
│   │   ├── hooks/
│   │   │   ├── usePrediction.js     ← TanStack Query mutation wrapping predictImage()
│   │   │   └── useDarkMode.js       ← Reads/applies dark mode from Zustand store
│   │   │
│   │   ├── store/
│   │   │   ├── useHistoryStore.js   ← Zustand: last 5 predictions in localStorage
│   │   │   └── useThemeStore.js     ← Zustand: dark/light mode in localStorage
│   │   │
│   │   └── utils/
│   │       ├── cn.js                ← clsx + tailwind-merge helper
│   │       ├── formatters.js        ← formatConfidence, formatFileSize, formatTimestamp
│   │       └── validators.js        ← validateImageFile, validateMimeType (magic bytes)
│   │
│   ├── vite.config.js               ← Vite: @ alias → src/, port 5173
│   ├── tailwind.config.js           ← Tailwind: dark class, CSS var tokens, animations
│   ├── postcss.config.js
│   ├── jsconfig.json
│   ├── eslint.config.js
│   └── package.json                 ← React 19, Vite 7, Zustand, TanStack Query, Recharts...
│
├── ── DOCS & CONFIG ────────────────────────────────────────────────
│
├── docs/                            ← Reference documentation
│   ├── API_REFERENCE.md
│   ├── ARCHITECTURE.md
│   ├── BACKEND.md
│   ├── DEVELOPER.md
│   ├── INFERENCE.md
│   ├── MODEL_CARD.md
│   ├── SETUP.md
│   ├── TRAINING.md
│   ├── TROUBLESHOOTING.md
│   └── USER_GUIDE.md
│
├── DOCUMENTATION.md                 ← This file — full project documentation
├── README.md                        ← Quick-start guide
├── requirements.txt                 ← Python dependencies
├── run.sh                           ← Shell script to start the API server
├── hf_config.json                   ← HuggingFace repo metadata
└── pyrightconfig.json               ← Pyright type-checker config
```

---

## 4. Configuration Layer

### `config.py` (root) — Training Config

| Constant | Value | Purpose |
|---|---|---|
| `DATASET_ROOT` | `"dataset"` | Root folder for all dataset splits |
| `TRAIN_IMG_DIR` | `dataset/Training_Set/.../Training` | Training images |
| `VALID_IMG_DIR` | `dataset/Evaluation_Set/.../Validation` | Validation images |
| `TEST_IMG_DIR` | `dataset/Test_Set/.../Test` | Test images |
| `TRAIN_CSV` | `.../RFMiD_Training_Labels.csv` | Training labels |
| `VALID_CSV` | `.../RFMiD_Validation_Labels.csv` | Validation labels |
| `TEST_CSV` | `.../RFMiD_Testing_Labels.csv` | Test labels |
| `OUTPUT_DIR` | `"outputs"` | Root for training output |
| `CHECKPOINT_DIR` | `outputs/checkpoints` | Saved `.pt` model files |
| `LOG_DIR` | `outputs/logs` | `training_log.csv` |
| `PLOT_DIR` | `outputs/plots` | `loss_curve.png` |
| `BEST_MODEL_PATH` | `outputs/checkpoints/best_model.pt` | Best AUC checkpoint |
| `LAST_MODEL_PATH` | `outputs/checkpoints/last_model.pt` | Last epoch checkpoint |
| `IMG_SIZE` | `384` | Input resolution |
| `BATCH_SIZE` | `5` | Tuned for 6GB VRAM |
| `NUM_WORKERS` | `2` | DataLoader workers |
| `EPOCHS` | `50` | Total training epochs |
| `LR` | `1e-4` | Classifier head LR |
| `WEIGHT_DECAY` | `1e-2` | AdamW weight decay |
| `POS_WEIGHT_CLIP` | `10.0` | Cap on per-class positive weight |
| `LABEL_COLS` | 45 strings | Disease codes, in model output order |
| `NUM_CLASSES` | `45` | `len(LABEL_COLS)` |
| `DEVICE` | `cuda` or `cpu` | Auto-selected at import |

### `app/config.py` — API / Serving Config

| Constant | Value | Purpose |
|---|---|---|
| `HF_MODEL_NAME` | `lebiraja/retinal-disease-classifier` | HuggingFace repo ID |
| `IMAGE_SIZE` | `384` | Must match training |
| `NORM_MEAN` | `(0.485, 0.456, 0.406)` | ImageNet mean |
| `NORM_STD` | `(0.229, 0.224, 0.225)` | ImageNet std |
| `DEFAULT_THRESHOLD` | `0.5` | Default sigmoid threshold |
| `MAX_FILE_SIZE_MB` | `10` | Per-upload size limit |
| `ALLOWED_MIME_TYPES` | jpeg, png, bmp, tiff | Accepted image formats |
| `MODEL_VERSION` | `"1.0"` | Exposed via `/info` |
| `BEST_AUC` | `0.8204` | Reported in `/info` |
| `BEST_EPOCH` | `39` | Reported in `/info` |
| `TRAIN_LOSS` | `0.2118` | Reported in `/info` |
| `VAL_LOSS` | `0.2578` | Reported in `/info` |
| `MACRO_F1` | `0.1517` | Reported in `/info` |
| `MICRO_F1` | `0.4450` | Reported in `/info` |
| `DISEASE_LABELS` | 45 codes | Same order as model outputs |
| `DISEASE_FULL_NAMES` | `{code: name}` | Human-readable display names |
| `HIGH_RISK_DISEASES` | set of 12 codes | Drives CRITICAL/HIGH escalation |

> **Why two config files?** Root `config.py` is training-only (dataset paths, hyperparams). `app/config.py` is API-only (thresholds, labels, MIME types). Keeps serving code independent of training code.

---

## 5. Machine Learning Layer (Training Pipeline)

### 5.1 `model.py`

**Purpose:** Define the neural network architecture — used identically by training and the API.

```python
def build_model(num_classes=45) -> nn.Module
def get_param_groups(model, lr) -> list
```

**Architecture:**
```
torchvision EfficientNet-B4 (ImageNet pretrained)
  └── model.classifier replaced with:
        ├── Dropout(p=0.4)
        └── Linear(1792 → 45)   ← one logit per disease
```

- `build_model()` — creates the model. Called by `train.py`, `inference.py`, and `model_service.py`.
- `get_param_groups()` — backbone gets `lr × 0.1` (gentle fine-tune), head gets full `lr` (train from scratch).

---

### 5.2 `dataset.py`

**Purpose:** Load and augment retinal images + binary labels for training/validation/testing.

- `RetinalDataset(split)` — reads CSV, builds filepaths, returns `(image[3,384,384], labels[45])`.
- **Train transforms:** resize → hflip → vflip → brightness/contrast → shift/scale/rotate → normalise → tensor
- **Val/Test transforms:** resize → normalise → tensor (no augmentation)
- `get_pos_weights()` — computes `neg/pos` per class, clipped at `POS_WEIGHT_CLIP=10`, fed to `BCEWithLogitsLoss`.

---

### 5.3 `train.py`

**Purpose:** Full training loop.

**Steps:**
1. `make_dirs()` — creates output directories.
2. DataLoaders for train + val splits.
3. `build_model()` → move to device.
4. `BCEWithLogitsLoss(pos_weight=...)` — handles class imbalance.
5. `AdamW` with differential LRs from `get_param_groups()`.
6. `CosineAnnealingLR` — smooth LR decay.
7. `GradScaler` + `autocast` — AMP for faster training / lower VRAM.
8. Per epoch: train → validate → compute metrics → checkpoint → log CSV.
9. `save_plots()` — generates `outputs/plots/loss_curve.png`.

**Output files:**
```
outputs/checkpoints/best_model.pt     ← best AUC (uploaded to HuggingFace)
outputs/checkpoints/last_model.pt     ← most recent epoch
outputs/logs/training_log.csv         ← epoch, losses, AUC, F1, LR
outputs/plots/loss_curve.png          ← training curves
```

---

### 5.4 `inference.py` (standalone CLI)

**Purpose:** Test a trained model on a single image without starting the API server.

```bash
python inference.py --image path/to/fundus.png
python inference.py --image path/to/fundus.png --threshold 0.4
```

Loads from a **local `.pt` checkpoint** (not HuggingFace). Returns all 45 probabilities + detected list as JSON.

---

## 6. API Layer (Serving Pipeline)

### 6.1 `app/main.py`

- Creates `FastAPI` instance with title, version, description, Swagger/ReDoc URLs.
- `lifespan` context manager: calls `ModelService.load()` on startup before accepting requests.
- Mounts `CORSMiddleware` (`allow_origins=["*"]`).
- Registers all routes via `app.include_router(router)`.

Start command:
```bash
uvicorn app.main:app --app-dir . --host 0.0.0.0 --port 8000
```
> `--app-dir .` puts project root on `sys.path` so `model.py` is importable.

---

### 6.2 `app/config.py`

API-only constants — see [Section 4](#4-configuration-layer).

---

### 6.3 `app/schemas.py`

Pydantic models — the **typed JSON contract** between backend and frontend.

| Schema | Used for |
|---|---|
| `PredictionResponse` | `POST /predict` response + each item in batch |
| `BatchPredictionResponse` | `POST /predict-batch` response |
| `HealthResponse` | `GET /health` response |
| `ModelMetrics` | Nested inside `ModelInfoResponse` |
| `ModelInfoResponse` | `GET /info` response |

**`PredictionResponse` fields:**

| Field | Type | Description |
|---|---|---|
| `disease_risk` | `bool` | True if ≥1 disease detected |
| `predictions` | `Dict[str, float]` | `{label: prob}` for detected diseases only |
| `detected_diseases` | `List[str]` | Short codes e.g. `["DR", "MCA"]` |
| `detected_diseases_full` | `List[str]` | Full names |
| `num_detected` | `int` | Count |
| `top_prediction` | `Optional[str]` | Highest-confidence code |
| `confidence` | `float` | Mean probability of detected diseases |
| `risk_level` | `str` | `LOW / MODERATE / HIGH / CRITICAL` |
| `advisory` | `str` | Informational advisory message |
| `elapsed_ms` | `float` | Inference wall-clock time |
| `threshold` | `float` | Threshold used |
| `disclaimer` | `str` | Fixed medical disclaimer |

---

### 6.4 `app/routers/predict.py`

All HTTP route handlers.

| Route | Handler | Description |
|---|---|---|
| `POST /predict` | `predict()` | Single image classification |
| `POST /predict-batch` | `predict_batch()` | Up to 10 images |
| `GET /health` | `health()` | Server liveness check |
| `GET /info` | `info()` | Model metadata + metrics |

**Validation helpers:**

| Helper | Checks |
|---|---|
| `_validate_upload` | MIME type in `ALLOWED_MIME_TYPES` |
| `_validate_bytes` | Not empty; not > `MAX_FILE_SIZE_MB` |
| `_validate_threshold` | 0.0 ≤ threshold ≤ 1.0 |

---

### 6.5 `app/services/model_service.py`

Singleton class — downloads, loads, and holds the model in memory for the server lifetime.

| Method | Description |
|---|---|
| `load()` | Download `pytorch_model.bin` from HuggingFace (cached), load state dict, move to device, eval mode. Called once at startup. |
| `get_model()` | Returns loaded `nn.Module`. Raises `RuntimeError` if not loaded. |
| `get_device()` | Returns `torch.device`. |
| `device_str()` | Returns `"cuda"` or `"cpu"` string. |
| `is_loaded()` | Returns `True` if model is in memory. |

**Checkpoint format:**
```python
{
    "epoch": int,
    "model_state_dict": OrderedDict,
    "optimizer_state_dict": OrderedDict,
    "mean_auc": float,
}
```
HuggingFace caches to `~/.cache/huggingface/hub/` — fast on subsequent starts.

---

### 6.6 `app/services/inference_service.py`

Module-level `_transform` pipeline (built once at import):
```
Resize(384, 384) → Normalize(ImageNet) → ToTensorV2
```
Matches the val transform in `dataset.py` exactly.

**`run_inference(file_bytes, threshold)`:**
1. `_preprocess(bytes)` — PIL → RGB → NumPy → transform → `[1, 3, 384, 384]` tensor
2. `model(tensor)` → logits `[1, 45]`
3. `sigmoid(logits)[0]` → probs `[45]`
4. Filter `prob >= threshold` → `{label: prob}` dict
5. Return `(predictions_dict, elapsed_ms)`

---

### 6.7 `app/services/advisory_service.py`

**`generate_advisory(predictions)`** converts `{label: prob}` (above-threshold only) into a full structured result dict including risk level and advisory text.

Risk logic:
```
num_detected == 0                     → LOW
num_high_risk >= 2                    → CRITICAL
num_high_risk == 1 OR num_detected >= 3  → HIGH
else                                  → MODERATE
```

For HIGH / CRITICAL, full names of high-risk diseases are **prepended** to the advisory text.

---

## 7. Frontend Layer (React App)

**Stack:** React 19 · Vite 7 · Tailwind CSS 3 · Zustand · TanStack Query v5 · Framer Motion · Radix UI · Recharts · Axios

### 7.1 Entry Point & App Shell

**`src/main.jsx`** — `ReactDOM.createRoot` → renders `<App />`.

**`src/App.jsx`** — sets up the entire application shell:
- `QueryClientProvider` — TanStack Query with `retry: 2`, `staleTime: 5min`, no refetch on focus
- `TooltipProvider` — Radix tooltip context for the whole app
- `BrowserRouter` + `Routes` — client-side routing
- `Toaster` (sonner) — top-right toast notifications
- `AnimatePresence` (Framer Motion) — enables exit animations on route change
- All pages are **lazy-loaded** (`React.lazy`) with a `<Suspense>` skeleton fallback

**Routes:**
| Path | Page |
|---|---|
| `/` | Landing |
| `/predict` | Predict |
| `/history` | History |
| `/how-it-works` | HowItWorks |
| `*` | NotFound |

All routes share the `<Layout>` shell (Navbar + Footer).

---

### 7.2 Pages

| File | Purpose |
|---|---|
| `Landing.jsx` | Hero section with stats (45 diseases, 0.82 AUC, <1s, FREE), feature cards, step-by-step overview, CTA buttons to `/predict` |
| `Predict.jsx` | Main analysis page — renders `ImageUpload` (left, 2 cols) and `ResultsPanel` / loading skeletons / error state / empty state (right, 3 cols). Drives everything via `usePrediction` hook |
| `History.jsx` | Thin wrapper — renders `HistoryList` inside a max-width container |
| `HowItWorks.jsx` | Detailed step-by-step explainer: Upload → Process → Analyze → Report. Includes architecture info and risk level table |
| `NotFound.jsx` | 404 with animated "404" text and back-to-home button |

---

### 7.3 Components — Layout

| File | Purpose |
|---|---|
| `Layout.jsx` | Root shell: sticky Navbar + `<Outlet />` + Footer. Adds a subtle background gradient. |
| `Navbar.jsx` | Sticky header with logo, nav links (Home, How It Works, Analyze, History), dark mode toggle (Moon/Sun icon), hamburger menu for mobile. Uses `useDarkMode` hook. Active link highlighted with Framer Motion underline. |
| `Footer.jsx` | Brand name + separator + disclaimer text. |
| `PageTransition.jsx` | Wraps each page in `motion.div` with `opacity + y` fade-in / fade-out. Used as the outermost element on every page. |

---

### 7.4 Components — Prediction

#### `ImageUpload.jsx`
The main input panel on the Predict page. Contains **two tabs**:

**Upload tab:**
- `react-dropzone` drag-and-drop zone (accepts JPEG/PNG, max 5MB)
- File preview (`<img>`) with remove button
- **`ScanOverlay`** animates over the preview image while `isLoading` is true
- Submit button ("Start Analysis") triggers `onSubmit(file)`

**Camera tab:**
- Renders `CameraCapture` component
- After capture, auto-switches back to Upload tab to show preview + submit button

**Validation:** `validateImageFile()` (size/type) + `validateMimeType()` (magic bytes) before storing.

---

#### `ScanOverlay.jsx` *(new)*
A pure visual overlay rendered **on top of the image preview** while analysis is running. Mimics a retinal / barcode scanner effect:

- Semi-transparent dark backdrop
- **Four corner brackets** in green forming a scanner viewfinder
- **Subtle horizontal grid lines** across the image
- **Sweeping green laser line** that bounces top ↔ bottom with a soft glow trail (Framer Motion animation)
- Blinking **"SCANNING RETINA"** status label at the bottom

All animations loop continuously using `repeatType: 'reverse'` until `isLoading` becomes false.

---

#### `CameraCapture.jsx` *(new)*
Live system camera component with a 4-phase state machine:

| Phase | What's shown |
|---|---|
| `idle` | "Start Camera" button — user explicitly triggers permission |
| `starting` | Pulsing camera icon while `getUserMedia` resolves |
| `live` | Live `<video>` feed with viewfinder brackets + blinking LIVE badge |
| `error` | Error message with Retry + Back buttons |

**Key implementation details:**
- Camera is **not auto-started** — avoids `videoRef` race condition and gives cleaner permission UX
- **3-level constraint fallback:** HD + facingMode → facingMode only → `{ video: true }` (most permissive)
- `onCanPlay` event on `<video>` sets phase to `live` — more reliable than awaiting `play()`
- **Flip camera** button toggles `user` ↔ `environment` (front vs back — useful on mobile)
- **Stop camera** button kills the stream and returns to idle
- Stream is stopped in `useEffect` cleanup — no camera leaks on unmount
- Snapshot via `canvas.drawImage(video)` → `canvas.toBlob()` → `new File(...)` → `onCapture(file)`
- After capture, `ImageUpload` auto-switches to the Upload tab to show preview + submit

**Error messages are specific per error name:**
| Error | Message |
|---|---|
| `NotAllowedError` | Permission denied — check browser address bar |
| `NotFoundError` | No camera found on this device |
| `NotReadableError` | Camera in use by another app |
| `OverconstrainedError` | Constraints relaxed on retry |
| Others | Error name + message shown for debugging |

---

#### `ResultsPanel.jsx`
Displayed after a successful prediction. Contains:
- **Top result banner** — disease name, risk badge (`LOW/MODERATE/HIGH/CRITICAL`), confidence %, processing time, condition count
- `ConfidenceBar` for the top disease
- **Clinical Advisory** card with advisory text from the backend
- **All Predictions** list — one `ConfidenceBar` per detected disease (staggered animation)
- **Confidence Distribution** — `PredictionChart` bar chart
- **Medical Disclaimer** box

If no diseases detected (`diseaseRisk: false`), shows a "No Disease Detected" green banner.

---

#### `ConfidenceBar.jsx`
Animated horizontal progress bar for a single disease prediction.
- Label left, percentage right (coloured by risk: red / amber / green)
- Bar animates from 0 → actual width on mount with stagger delay based on `index`

---

### 7.5 Components — Charts

#### `PredictionChart.jsx`
Recharts `BarChart` showing all detected disease probabilities as a colour-coded bar chart.
- Bars coloured with 8-colour palette cycling
- Custom tooltip showing disease name + exact confidence %
- Rotated X-axis labels for readability
- Y-axis 0–100%

---

### 7.6 Components — History

| File | Purpose |
|---|---|
| `HistoryList.jsx` | Renders list header (count badge, Clear All button) + `AnimatePresence`-wrapped list of `HistoryCard`. Shows empty state with illustration if no history. |
| `HistoryCard.jsx` | Single entry: top disease name, risk badge, confidence %, timestamp, mini confidence bar. Hover reveals a Trash icon to remove the entry. Staggered fade-in on mount. |

---

### 7.7 Components — UI Primitives

All live in `src/components/ui/` — Radix UI primitives styled with Tailwind CSS (shadcn/ui pattern). They use CSS variable tokens so they work in both light and dark modes automatically.

| File | Primitive |
|---|---|
| `badge.jsx` | Inline label with variant colours (default, destructive, warning, success) |
| `button.jsx` | Button with variants (default, outline, ghost, destructive) and sizes |
| `card.jsx` | Card, CardContent, CardHeader, CardTitle |
| `separator.jsx` | Horizontal/vertical divider |
| `skeleton.jsx` | Animated grey pulse placeholder for loading states |
| `tooltip.jsx` | Radix Tooltip with styled content |

---

### 7.8 Services (Frontend)

#### `src/services/api.js`
Centralised Axios instance:
- `baseURL`: `VITE_API_URL` env var or `http://localhost:8000`
- `timeout`: 30 seconds
- Request interceptor: logs `[API] METHOD /path` in dev mode
- Response interceptor: normalises all errors to `new Error(message)` using `detail` → `message` → `error.message` chain

#### `src/services/prediction.js`
Functions that use `api.js`:

| Function | Description |
|---|---|
| `predictImage(file, threshold)` | Builds FormData, `POST /predict`, transforms response |
| `checkHealth()` | `GET /health` |
| `getModelInfo()` | `GET /info` |

**`transformPrediction(raw)`** — converts backend snake_case response to frontend camelCase:
- `predictions` dict `{label: prob}` → sorted array `[{disease, fullName, confidence}]`
- `disease_risk` → `diseaseRisk`
- `elapsed_ms` → `elapsedMs` + `processing_time` (seconds)
- All other fields mapped to camelCase

---

### 7.9 Hooks

#### `usePrediction.js`
TanStack Query `useMutation` wrapping `predictImage`:
- On success: calls `addPrediction(data)` to save to history, shows a success toast with top finding name
- On error: shows an error toast with the error message
- Exposes: `predict`, `predictAsync`, `data`, `isLoading`, `isError`, `error`, `reset`

#### `useDarkMode.js`
- Reads `isDark` + `toggleTheme` from `useThemeStore`
- Calls `initTheme()` on mount to apply the persisted theme class to `<html>`
- Exposes: `{ isDark, toggleTheme }`

---

### 7.10 State Management (Zustand)

#### `useHistoryStore.js`
Persisted to `localStorage` under key `retinal-history`.

| Action | Description |
|---|---|
| `addPrediction(result)` | Prepends new entry, keeps last 5 (FIFO) |
| `clearHistory()` | Empties the list |
| `removeEntry(id)` | Removes a single entry by timestamp ID |

Each stored entry contains: `id`, `disease`, `confidence`, `riskLevel`, `numDetected`, `allPredictions`, `processingTime`, `timestamp`.

#### `useThemeStore.js`
Persisted to `localStorage` under key `retinal-theme`.

| Action | Description |
|---|---|
| `toggleTheme()` | Flips `isDark`, applies/removes `dark` class on `<html>` |
| `initTheme()` | Called on mount — re-applies persisted theme |

---

### 7.11 Utilities

| File | Exports | Purpose |
|---|---|---|
| `cn.js` | `cn(...classes)` | `clsx` + `tailwind-merge` — prevents conflicting Tailwind classes |
| `formatters.js` | `formatConfidence(v)` | `0.78` → `"78.0%"` |
| | `formatProcessingTime(s)` | `0.018` → `"18ms"`, `1.2` → `"1.20s"` |
| | `formatFileSize(bytes)` | `1048576` → `"1.00 MB"` |
| | `formatTimestamp(date)` | `"Feb 27, 05:30 PM"` |
| | `getRiskLevel(confidence)` | `0.8` → `{level: "High", color: "destructive"}` |
| `validators.js` | `validateImageFile(file)` | Checks type (JPEG/PNG) + size (max 5MB) |
| | `validateMimeType(file)` | Reads first 4 bytes (magic bytes) to verify actual image format |

---

## 8. Data Flow — Request Lifecycle

```
Client uploads fundus.jpg + threshold=0.5
        │
        ▼
[app/routers/predict.py] predict()
  1. _validate_upload    → check MIME type
  2. await image.read()  → raw bytes
  3. _validate_bytes     → check size
  │
  ├──▶ [inference_service.py] run_inference(bytes, 0.5)
  │         PIL.open → RGB → np.array
  │         → Resize(384,384) → Normalize → ToTensorV2 → [1,3,384,384]
  │         → model(tensor) → logits [1,45]
  │         → sigmoid → probs [45]
  │         → filter prob >= 0.5
  │         → { "DR": 0.78, "MCA": 0.61 } + elapsed_ms
  │
  ├──▶ [advisory_service.py] generate_advisory({"DR":0.78, "MCA":0.61})
  │         num_detected = 2
  │         high_risk_detected = ["DR"]
  │         risk_level = "HIGH"
  │         advisory = "Detected indicator(s): Diabetic Retinopathy. ..."
  │
  └──▶ return PredictionResponse(
             disease_risk=True,
             predictions={"DR":0.78, "MCA":0.61},
             detected_diseases=["DR","MCA"],
             detected_diseases_full=["Diabetic Retinopathy","Microaneurysms"],
             num_detected=2, top_prediction="DR",
             confidence=0.695, risk_level="HIGH",
             advisory="...", elapsed_ms=18.4,
             threshold=0.5, disclaimer="..."
       )
```

---

## 9. Data Flow — Training Lifecycle

```
dataset/ CSVs + PNG images
        │
        ▼
[dataset.py] RetinalDataset(split)
  CSV → filepaths → PIL.open → augment → normalize → tensor [3,384,384]
  labels [45] (binary, float32)
        │
        ▼
[train.py] DataLoader(batch_size=5)
        │
        ▼
[model.py] build_model()   EfficientNet-B4 → Dropout → Linear(1792,45)
        │
        ▼
[train.py] per epoch:
  train_one_epoch()  → autocast → forward → BCEWithLogitsLoss → GradScaler
  evaluate()         → collect probs+targets → AUC + F1
  if best AUC → save outputs/checkpoints/best_model.pt
  always     → save outputs/checkpoints/last_model.pt
              → append outputs/logs/training_log.csv
        │
        ▼ (after all epochs)
  save_plots() → outputs/plots/loss_curve.png
```

---

## 10. Data Flow — Frontend to Backend

```
User on /predict page
        │
        ├─ Tab: Upload ──────────────────────────────────────────────────
        │   react-dropzone onDrop(file)
        │     validateImageFile(file)    ← type + size check
        │     validateMimeType(file)     ← magic bytes check
        │     setPreview(URL.createObjectURL(file))
        │
        ├─ Tab: Camera ──────────────────────────────────────────────────
        │   CameraCapture
        │     navigator.mediaDevices.getUserMedia() → MediaStream
        │     <video> autoPlay + onCanPlay → phase = 'live'
        │     "Capture Photo" → canvas.drawImage(video)
        │     canvas.toBlob() → new File(...) → onCapture(file)
        │     → auto-switch to Upload tab → show preview
        │
        └─ "Start Analysis" button ──────────────────────────────────────
            ScanOverlay renders over image (isLoading=true)
            usePrediction.predict(file)
              └─ TanStack Query mutationFn: predictImage(file, threshold)
                    FormData.append('image', file)
                    axios.post('/predict', formData, { params: { threshold }})
                    ← transformPrediction(raw) → camelCase result
            onSuccess:
              addPrediction(data) → useHistoryStore (localStorage)
              toast.success("Analysis complete")
              ScanOverlay disappears (isLoading=false)
              ResultsPanel renders with data
```

---

## 11. Disease Labels Reference

45 conditions, fixed order matching model output indices:

| Index | Code | Full Name | High-Risk |
|---|---|---|---|
| 0 | DR | Diabetic Retinopathy | ✅ |
| 1 | ARMD | Age-Related Macular Degeneration | ✅ |
| 2 | MH | Macular Hole | |
| 3 | DN | Drusen | |
| 4 | MYA | Myopic Astigmatism | |
| 5 | BRVO | Branch Retinal Vein Occlusion | ✅ |
| 6 | TSLN | Tessellation | |
| 7 | ERM | Epiretinal Membrane | |
| 8 | LS | Laser Scar | |
| 9 | MS | Macular Scar | |
| 10 | CSR | Central Serous Retinopathy | |
| 11 | ODC | Optic Disc Cupping | |
| 12 | CRVO | Central Retinal Vein Occlusion | ✅ |
| 13 | TV | Tire Venture | |
| 14 | AH | Anterior Chamber | |
| 15 | ODP | Optic Disc Pallor | |
| 16 | ODE | Optic Disc Edema | |
| 17 | ST | Shunt | |
| 18 | AION | Anterior Ischemic Optic Neuropathy | ✅ |
| 19 | PT | Parafoveal Telangiectasia | |
| 20 | RT | Retinal Traction | |
| 21 | RS | Retinal Scar | |
| 22 | CRS | Corneal Reflex Shadow | |
| 23 | EDN | Exudates | |
| 24 | RPEC | RPE Changes | |
| 25 | MHL | Macular Hole (Large) | |
| 26 | RP | Retinitis Pigmentosa | ✅ |
| 27 | CWS | Cotton Wool Spots | |
| 28 | CB | Conjunctival Bleed | |
| 29 | ODPM | Optic Disc Pallor Margin | |
| 30 | PRH | Peripapillary Retinal Hemorrhage | |
| 31 | MNF | Macular Neovascularization | ✅ |
| 32 | HR | Hard Retinal Exudate | |
| 33 | CRAO | Central Retinal Artery Occlusion | ✅ |
| 34 | TD | Temporal Disc | |
| 35 | CME | Cystoid Macular Edema | ✅ |
| 36 | PTCR | Posterior Capsular Rent | |
| 37 | CF | Cotton Fiber | |
| 38 | VH | Vitreous Hemorrhage | ✅ |
| 39 | MCA | Microaneurysms | |
| 40 | VS | Vitreous Synchysis | |
| 41 | BRAO | Branch Retinal Artery Occlusion | ✅ |
| 42 | PLQ | Placoid Lesion | |
| 43 | HPED | Hemorrhagic Pigment Epithelial Detachment | ✅ |
| 44 | CL | Cotton Lint | |

**High-risk disease codes (12):** DR, ARMD, BRVO, CRVO, AION, RP, MNF, CRAO, CME, VH, BRAO, HPED

---

## 12. Risk Level Logic

Implemented in `app/services/advisory_service.py`:

```
if num_detected == 0:                              → LOW
elif count(high_risk_detected) >= 2:               → CRITICAL
elif count(high_risk_detected) == 1
     OR num_detected >= 3:                         → HIGH
else:  # 1–2 detected, none high-risk              → MODERATE
```

| Level | Advisory summary |
|---|---|
| LOW | No abnormalities detected. Regular exams recommended. |
| MODERATE | Possible changes found. Schedule an ophthalmologist consultation. |
| HIGH | Significant pathology detected. Prompt evaluation strongly recommended. |
| CRITICAL | Sight-threatening indicators. Immediate consultation advised. |

For HIGH and CRITICAL, full names of triggering diseases are **prepended** to the advisory text.

---

## 13. API Endpoints Quick Reference

Base URL: `http://localhost:8000`

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness: is server up + model loaded |
| `GET` | `/info` | Model metadata, architecture, training metrics |
| `POST` | `/predict` | Single fundus image → disease predictions |
| `POST` | `/predict-batch` | Up to 10 images → list of predictions |
| `GET` | `/docs` | Swagger UI (interactive API explorer) |
| `GET` | `/redoc` | ReDoc UI |

**Error codes:**

| Status | Cause |
|---|---|
| `400` | Wrong MIME type, empty file, no images in batch |
| `413` | File > 10 MB |
| `422` | threshold outside 0.0–1.0 |
| `500` | Inference failure |

---

## 14. Key Design Decisions

| Decision | Reason |
|---|---|
| **EfficientNet-B4** | Best accuracy/speed tradeoff for medical image classification at 384px |
| **Multi-label output** | Patients can have multiple diseases simultaneously — 45 independent sigmoids |
| **BCEWithLogitsLoss + pos_weight** | Dataset is heavily imbalanced; some diseases appear in <1% of images |
| **Differential learning rates** | Pretrained backbone needs gentle fine-tuning (lr×0.1); new head needs full lr |
| **AMP training** | ~1.5–2× speedup + halves VRAM. No accuracy loss |
| **Singleton ModelService** | Model (~100M params) loaded once at startup, shared across all requests |
| **HuggingFace for weights** | Keeps large binary out of git. `hf_hub_download` handles caching automatically |
| **Two config files** | Root `config.py` = training only. `app/config.py` = API only. Clean separation |
| **`--workers 1` for uvicorn** | Multiple workers would each load the full model — multiplies VRAM |
| **Threshold as query param** | Frontend can tune sensitivity/specificity per use case without backend changes |
| **Camera not auto-started** | Avoids React ref race condition; gives user explicit permission UX |
| **3-level camera fallback** | HD+facing → facing only → `{video:true}` — handles all device/browser combos |
| **`onCanPlay` for camera ready** | More reliable than `await video.play()` which can throw on tab switch |
| **ScanOverlay as separate component** | Decoupled from upload logic; easy to swap or disable |
| **Zustand + persist** | Simple state management for history/theme with automatic localStorage sync |
| **TanStack Query mutation** | Handles loading/error states, retries, and cache invalidation for the prediction call |
| **transformPrediction()** | Single place to convert backend snake_case → frontend camelCase; keeps components clean |

---

## 15. Commands Cheat Sheet

```bash
# ── Backend setup ─────────────────────────────────────────────────
pip install -r requirements.txt

# ── Start API server ──────────────────────────────────────────────
bash run.sh
# or:
uvicorn app.main:app --app-dir . --host 0.0.0.0 --port 8000 --reload

# ── Test API ──────────────────────────────────────────────────────
curl http://localhost:8000/health
curl http://localhost:8000/info
curl -X POST http://localhost:8000/predict \
  -F "image=@/path/to/fundus.jpg" \
  -F "threshold=0.5"
curl -X POST http://localhost:8000/predict-batch \
  -F "images=@img1.jpg" -F "images=@img2.jpg"

# ── Frontend setup ────────────────────────────────────────────────
cd cnn/retinal-frontend
npm install
npm run dev          # http://localhost:5173
npm run build        # production build → dist/
npm run preview      # preview production build

# ── Environment variable (optional) ──────────────────────────────
# Create cnn/retinal-frontend/.env
VITE_API_URL=http://localhost:8000

# ── Training (requires RFMiD dataset in dataset/) ─────────────────
python train.py
python train.py --epochs 2    # quick sanity check

# ── Standalone inference (no server, local checkpoint) ────────────
python inference.py --image path/to/fundus.png
python inference.py --image path/to/fundus.png --threshold 0.3

# ── Docs & Swagger ────────────────────────────────────────────────
open http://localhost:8000/docs
open http://localhost:8000/redoc
```

---

> **Medical Disclaimer:** This system is NOT a medical diagnostic device.
> Results are for screening and educational purposes only.
> Always consult a qualified ophthalmologist for clinical decisions.

---

*Team B · Retinal Disease Classifier · Documentation v2.0 · Updated February 2026*
