# detect_ai.py
from PIL import Image

def predict_ai_generated(image_path):
    """
    A simple placeholder for AI detection logic.
    Replace with your own model or logic.
    """
    image = Image.open(image_path)

    # Example logic: use image size or metadata (just for demo)
    if image.size[0] > 1024 and image.size[1] > 1024:
        return "AI-generated"
    else:
        return "Real"
