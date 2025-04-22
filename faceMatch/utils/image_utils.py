from PIL import Image
import yaml
from functools import lru_cache


def load_image(image_path):
    """Loads an image and returns the PIL Image object."""
    return Image.open(image_path).convert("RGB")


@lru_cache(maxsize=1)
def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
