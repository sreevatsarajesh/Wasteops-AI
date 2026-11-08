from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from .config import get_settings
from .guidance import get_guidance
from .image_utils import decode_image
from .inference import service, ModelUnavailable
from .taxonomy import CLASSES, DISPLAY

ROOT = Path(__file__).resolve().parents[1]

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="WasteOps API", version="1.0.0", description="Image-based waste material recognition and general handling guidance.", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index():
    return (ROOT / "web" / "index.html").read_text(encoding="utf-8")

@app.get("/api/health")
async def health():
    return {"status":"ok", "model_ready":service.ready, "backend":service.backend, "version":"1.0.0"}

@app.get("/api/classes")
async def classes():
    return {"classes":[{"id":c,"name":DISPLAY[c]} for c in CLASSES]}

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), region: str = Form(default="")):
    settings = get_settings()
    content_type = (file.content_type or "").lower()
    if content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG or WebP image.")
    raw = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image exceeds {settings.max_upload_mb} MB.")
    try:
        image = decode_image(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        prediction = service.predict(image)
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    label = prediction.label
    result = {
        "label": label,
        "name": "Needs manual review" if label == "unknown" else DISPLAY[label],
        "score": round(prediction.score,4),
        "score_is_calibrated_probability": False,
        "abstained": prediction.abstained,
        "alternatives": prediction.alternatives,
        "backend": prediction.backend,
        "guidance": get_guidance(label),
        "region": region[:100],
        "limitations": ["One dominant category is predicted; multiple objects may be missed.",
            "Visual appearance cannot establish material composition or local recyclability.",
            "Contamination and material purity are not assessed in v1.0."],
    }
    return result
