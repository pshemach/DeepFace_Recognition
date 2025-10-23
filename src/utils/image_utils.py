from PIL import Image
import yaml
from functools import lru_cache
import io
import os
import hashlib
from werkzeug.utils import secure_filename

def get_safe_filename(filename):
    name, ext = os.path.splitext(secure_filename(filename))
    short_hash = hashlib.sha256(name.encode()).hexdigest()[:16]
    return f"{short_hash}{ext}"


def save_image(file_storage, save_path):
    try:
        print(f"Trying to save image: {file_storage.filename}")

        image_bytes = file_storage.read()

        if len(image_bytes) == 0:
            return False, "Uploaded image is empty"

        image_stream = io.BytesIO(image_bytes)
        image = Image.open(image_stream)
        image.verify()  # Validate format

        # Re-open for saving
        image_stream.seek(0)
        image = Image.open(image_stream)

        # Create directory if missing
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save image
        image_format = image.format if image.format else 'JPEG'
        image.save(save_path, format=image_format)

        print(f"Image successfully saved to: {save_path}")
        return True, None

    except Exception as e:
        print(f"Exception in save_image: {e}")
        return False, str(e)

def load_image(image_path):
    """Loads an image and returns the PIL Image object."""
    return Image.open(image_path).convert("RGB")


@lru_cache(maxsize=1)
def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
