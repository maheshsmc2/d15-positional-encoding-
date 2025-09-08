from PIL import Image
from transformers import pipeline

__version__ = "1.0.0"

try:
    captioner = pipeline(
        "image-to-text",
        model="nlpconnect/vit-gpt2-image-captioning",
        device=-1
    )
except Exception as e:
    captioner = None
    _load_error = str(e)
else:
    _load_error = None

def predict(image: Image.Image, max_new_tokens: int = 32) -> str:
    if captioner is None:
        raise RuntimeError(f"Failed to load pipeline: {_load_error}")
    out = captioner(image, max_new_tokens=max_new_tokens)
    if isinstance(out, list) and len(out) and "generated_text" in out[0]:
        return out[0]["generated_text"]
    return str(out)
