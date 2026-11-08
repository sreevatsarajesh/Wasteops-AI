from __future__ import annotations
from pathlib import Path
import threading
import torch
from PIL import Image
from .config import get_settings
from .models import Prediction
from .taxonomy import CLASSES, DISPLAY, PROMPTS

class ModelUnavailable(RuntimeError):
    pass

class InferenceService:
    def __init__(self):
        self.settings = get_settings()
        self._model = None
        self._processor = None
        self._backend = None
        self._lock = threading.Lock()

    def _device(self):
        requested = self.settings.device.lower()
        if requested != "auto":
            if requested == "cuda" and not torch.cuda.is_available():
                raise ModelUnavailable("DEVICE=cuda was selected but CUDA is unavailable.")
            if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise ModelUnavailable("DEVICE=mps was selected but Apple MPS is unavailable.")
            return torch.device(requested)
        if torch.cuda.is_available(): return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")

    @property
    def ready(self):
        return self._model is not None

    @property
    def backend(self):
        return self._backend

    def load(self):
        if self.ready: return
        with self._lock:
            if self.ready: return
            try:
                device = self._device()
                mode = self.settings.model_backend.lower()
                checkpoint = self.settings.checkpoint_path
                if mode not in {"auto", "clip", "finetuned"}:
                    raise ModelUnavailable("MODEL_BACKEND must be auto, clip or finetuned.")
                if mode == "finetuned" or (mode == "auto" and checkpoint.exists()):
                    self._load_finetuned(checkpoint, device)
                else:
                    self._load_clip(device)
            except ModelUnavailable:
                raise
            except Exception as exc:
                raise ModelUnavailable(f"Could not load model: {exc}") from exc

    def _load_clip(self, device):
        from transformers import CLIPModel, CLIPProcessor
        self._processor = CLIPProcessor.from_pretrained(self.settings.model_id)
        self._model = CLIPModel.from_pretrained(self.settings.model_id).to(device).eval()
        self._device_value = device
        self._backend = "clip-zero-shot"
        self._text_features = []
        texts, owners = [], []
        for label in CLASSES:
            for prompt in PROMPTS[label]:
                texts.append(prompt); owners.append(label)
        with torch.inference_mode():
            tokens = self._processor(text=texts, return_tensors="pt", padding=True).to(device)
            features = self._model.get_text_features(**tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        self._text_features = features
        self._owners = owners

    def _load_finetuned(self, checkpoint: Path, device):
        from torchvision import models
        if not checkpoint.exists():
            raise ModelUnavailable(f"Fine-tuned checkpoint not found: {checkpoint}. Run scripts/train.py first.")
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(CLASSES))
        model.load_state_dict(payload["state_dict"])
        model.to(device).eval()
        self._model = model
        self._device_value = device
        self._backend = "mobilenetv3-finetuned"
        from torchvision import transforms
        self._transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    def predict(self, image: Image.Image) -> Prediction:
        self.load()
        with torch.inference_mode():
            if self._backend == "clip-zero-shot":
                inputs = self._processor(images=image, return_tensors="pt").to(self._device_value)
                image_features = self._model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                sims = (image_features @ self._text_features.T).squeeze(0)
                per_class = {}
                for idx, owner in enumerate(self._owners):
                    per_class[owner] = max(per_class.get(owner, -1.0), float(sims[idx].item()))
                ordered = sorted(per_class.items(), key=lambda item: item[1], reverse=True)
                # Convert cosine similarities to a relative ranking score. This is not a probability.
                vals = torch.tensor([v for _,v in ordered])
                relative = torch.softmax(vals * 12.0, dim=0).tolist()
                ranked = [{"label": k, "name": DISPLAY[k], "score": round(float(s),4)} for (k,_),s in zip(ordered, relative)]
                top_score = ranked[0]["score"]
                margin = ranked[0]["score"] - ranked[1]["score"]
                abstain = top_score < self.settings.min_score or margin < self.settings.min_margin
                label = "unknown" if abstain else ranked[0]["label"]
                return Prediction(label, top_score, ranked[:3], self._backend, abstain)
            tensor = self._transform(image).unsqueeze(0).to(self._device_value)
            logits = self._model(tensor).squeeze(0)
            probs = torch.softmax(logits, dim=0)
            vals, indices = torch.topk(probs, k=3)
            ranked = [{"label": CLASSES[int(i)], "name": DISPLAY[CLASSES[int(i)]], "score": round(float(v),4)} for v,i in zip(vals,indices)]
            score = ranked[0]["score"]
            # Conservative floor; validation/calibration must be performed on an external set.
            abstain = score < 0.40
            return Prediction("unknown" if abstain else ranked[0]["label"], score, ranked, self._backend, abstain,
                "Softmax score; not calibrated. Validate on representative local data before operational use.")

service = InferenceService()
