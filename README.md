# WasteOps v1.0

**Image-based waste material intelligence — first public release.**

WasteOps identifies broad waste-material categories from a photograph and provides cautious, general handling guidance. It is a software-only MVP built around a pretrained CLIP zero-shot model, with an optional supervised MobileNetV3 training pipeline using UCI RealWaste.

> **Important scope:** v1.0 predicts one dominant material category for an image. It does not reliably detect multiple objects, certify recyclability, assess contamination, estimate recoverable value, or determine local disposal rules. Recommendations are general guidance; local authorities and waste contractors take precedence.

## Features

- Browser-based responsive image upload interface.
- FastAPI backend with `/api/analyze`, `/api/health`, `/api/classes` and `/docs`.
- CLIP ViT-B/32 zero-shot material recognition (model weights download on first use).
- Optional fine-tuned MobileNetV3-Small checkpoint takes precedence when available.
- Unknown/low-margin prediction handling; displayed scores are model scores, **not calibrated probabilities**.
- Material guidance catalogue with explicit local-rule caveats.
- UCI RealWaste downloader, dataset validator, stratified split and MobileNetV3 training/evaluation scripts.
- Unit tests, Dockerfile and docker-compose.

## Quick start (macOS / Linux)

Python 3.10–3.12 recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
uvicorn wasteops.api:app --host 127.0.0.1 --port 8000 --reload
```

Open http://127.0.0.1:8000. The first analysis downloads `openai/clip-vit-base-patch32` from Hugging Face (internet required; several hundred MB). You can instead download the model in advance with:

```bash
python scripts/warm_model.py
```

### First run notes

- The first inference can take longer while model weights download and load.
- CPU inference works; an Apple Silicon machine can use PyTorch's MPS where supported. Set `DEVICE=cpu` to force CPU.
- If Hugging Face is unreachable, the API returns a clear model-unavailable error; it does not fabricate a prediction.
- For best results, photograph one item against a reasonably clear background, in good light, with the whole item visible.

## Train a domain model on RealWaste

RealWaste contains 4,752 images across nine categories. UCI lists it as CC BY 4.0. The images are not included in this repository; download them directly from UCI and retain attribution. Dataset record: https://archive.ics.uci.edu/dataset/908/realwaste

```bash
python scripts/download_realwaste.py
python scripts/validate_dataset.py --data-dir data/RealWaste
python scripts/train.py --data-dir data/RealWaste --epochs 12 --batch-size 32
```

The downloader fetches the official UCI ZIP to `data/raw/` and extracts it. The training script creates a deterministic stratified 80/10/10 split, trains MobileNetV3-Small with ImageNet transfer learning, saves the best checkpoint to `models/wasteops_mobilenetv3.pt`, and writes metrics to `reports/`.

Once a checkpoint exists, the API automatically uses it instead of CLIP. To force CLIP, set `MODEL_BACKEND=clip`; to require a trained checkpoint, set `MODEL_BACKEND=finetuned`.

**Research hygiene:** the supplied split is image-level and may not guarantee site-level independence. Do not present its test score as cross-facility generalization. For a paper, create a site- or source-held-out test set and document any near-duplicates.

## API

### `POST /api/analyze`
Multipart form fields:
- `file`: JPEG, PNG or WebP image (max 10 MB)
- `region`: optional free-text region label, for logging/display only; it does not change disposal rules

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -F 'file=@sample.jpg' -F 'region=London, UK'
```

### `GET /api/health`
Reports service and model state. `model_ready=false` means the first inference has not loaded weights yet.

### `GET /api/classes`
Returns supported broad material categories.

Interactive OpenAPI documentation: http://127.0.0.1:8000/docs

## Configuration

Copy `.env.example` to `.env`.

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_BACKEND` | `auto` | `auto`, `clip`, or `finetuned` |
| `MODEL_ID` | `openai/clip-vit-base-patch32` | Hugging Face CLIP model |
| `MODEL_PATH` | `models/wasteops_mobilenetv3.pt` | Fine-tuned checkpoint |
| `DEVICE` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |
| `MAX_UPLOAD_MB` | `10` | Upload limit |
| `MIN_SCORE` | `0.24` | Conservative CLIP score floor; not a calibrated probability |
| `MIN_MARGIN` | `0.025` | Minimum gap to runner-up before abstaining |

## Run tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

Tests do not download model weights. The inference tests use a mocked model to keep CI deterministic.

## Docker

```bash
docker compose up --build
```

Open http://localhost:8000. The first image analysis downloads model weights into the mounted Hugging Face cache. For production, use HTTPS, authentication/rate limiting, persistent model cache, and a privacy policy before accepting user uploads.

## Repository layout

```text
wasteops-v1.0/
├── wasteops/              # API, model inference, guidance, config
├── web/                   # Browser UI
├── scripts/               # Dataset, training, evaluation utilities
├── tests/                 # Unit and API tests
├── data/                  # Downloaded data (gitignored)
├── models/                # Trained checkpoints (gitignored)
├── reports/               # Training metrics (gitignored)
├── docs/                  # Architecture and dataset notes
├── Dockerfile
├── docker-compose.yml
├── requirements*.txt
└── README.md
```

## Model and dataset attribution

- CLIP: Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, 2021. Model card: https://huggingface.co/openai/clip-vit-base-patch32. Check model card and upstream terms before redistribution or commercial deployment.
- RealWaste: Single, S., Iranmanesh, S., & Raad, R. (2023). *RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning*. Information, 14(12), 633. https://doi.org/10.3390/info14120633. UCI dataset record: https://doi.org/10.24432/C5SS4G. Cite the dataset and verify license terms for your intended use.

## Known limitations / roadmap

- CLIP zero-shot output is a broad visual similarity classification, not a validated waste-sorting instrument.
- The fine-tuned classifier predicts one dominant class and is not an object detector or segmenter.
- The current guidance is intentionally generic and must not be treated as a country-specific recycling decision.
- No contamination classifier or material purity estimator is included in v1.0.
- Next research version: open-set recognition, contamination labels, site-held-out evaluation, calibrated uncertainty, and verified region-specific acceptance rules.

## License

The WasteOps source code is released under the MIT License. This source-code license does not relicense model weights, datasets, or third-party assets. See `DATASET_NOTICE.md`.
