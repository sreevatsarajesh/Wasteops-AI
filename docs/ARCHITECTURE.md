# Architecture

1. Browser posts an image to FastAPI `/api/analyze`.
2. API validates MIME type, byte size, image format and pixel count.
3. Inference service lazily loads either a fine-tuned MobileNetV3 checkpoint or CLIP ViT-B/32.
4. The model returns a ranked set of broad material classes. Low-confidence/low-margin CLIP outputs abstain.
5. A deterministic guidance catalogue provides generic handling suggestions and always warns that local acceptance rules vary.

The inference service is lazy-loaded so health checks and API docs work before model weights are downloaded. It is process-local; production multi-worker deployments should account for one model copy per worker.
