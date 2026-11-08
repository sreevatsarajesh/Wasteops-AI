from io import BytesIO
from PIL import Image, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

def decode_image(raw: bytes, max_pixels: int = 20_000_000) -> Image.Image:
    if not raw:
        raise ValueError("The uploaded file is empty.")
    try:
        with Image.open(BytesIO(raw)) as image:
            if image.format not in ALLOWED_FORMATS:
                raise ValueError("Unsupported image format. Upload JPEG, PNG or WebP.")
            if image.width * image.height > max_pixels:
                raise ValueError("Image dimensions are too large. Maximum is 20 megapixels.")
            image.verify()
        with Image.open(BytesIO(raw)) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid supported image.") from exc
