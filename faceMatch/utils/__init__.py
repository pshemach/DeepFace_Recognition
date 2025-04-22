import yaml
import os
from PIL import Image
from faceMatch.constant import ALLOWED_FILE_EXTENSIONS, REFERENCE_FOLDER
import uuid
import shutil
import tempfile
import time
from functools import lru_cache
import glob
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {ext.lower() for ext in ALLOWED_FILE_EXTENSIONS}
previous_temp_dirs = []


@lru_cache(maxsize=1)
def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def load_image(image_path):
    """Loads an image and returns the PIL Image object."""
    return Image.open(image_path).convert("RGB")


def is_allowed(filename):
    """Check if the file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def make_dir(path):
    """Create a directory; if it exists, clear its contents."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def delete_previous_files(upload_dir, result_path):
    """Deletes previous uploaded image and result image from the directories."""
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)
        os.makedirs(upload_dir)
    if os.path.exists(result_path):
        os.remove(result_path)


def make_temp_folder(root_dir):
    """Creates a unique temp directory per request"""
    return tempfile.mkdtemp(dir=root_dir)


def unique_filenameuni(image_name):
    """Generate a unique filename for the uploaded image"""
    return f"{uuid.uuid4().hex}_{image_name}"


def create_temp_directory_with_age_limit(root_dir, max_age=60):
    """Deletes old temporary directories and creates a new temporary directory."""
    global previous_temp_dirs
    current_time = time.time()

    # Filter out and delete expired directories
    previous_temp_dirs = [
        (temp_dir, creation_time)
        for temp_dir, creation_time in previous_temp_dirs
        if current_time - creation_time <= max_age
        or not shutil.rmtree(temp_dir, ignore_errors=True)
    ]

    # Create a new temporary directory and store it with the current time
    new_temp_dir = tempfile.mkdtemp(dir=root_dir)
    previous_temp_dirs.append((new_temp_dir, current_time))
    return new_temp_dir


def save_reference_image(image_file, key):
    """Save a reference image with the given key.

    Args:
        image_file: The uploaded image file object
        key: The unique key to identify this reference image

    Returns:
        The path to the saved reference image
    """
    print(f"Saving reference image with key: {key}")
    print(f"Image filename: {image_file.filename}")

    # Ensure the main reference directory exists
    os.makedirs(REFERENCE_FOLDER, exist_ok=True)
    print(f"Reference folder: {REFERENCE_FOLDER}")

    # Create a directory for this specific key
    key_folder = os.path.join(REFERENCE_FOLDER, key)
    os.makedirs(key_folder, exist_ok=True)
    print(f"Key folder: {key_folder}")

    # Get file extension from the original filename
    filename = secure_filename(image_file.filename)
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'jpg'
    print(f"File extension: {ext}")

    # Create a filename using the key
    reference_filename = f"{key}.{ext}"
    reference_path = os.path.join(key_folder, reference_filename)
    print(f"Reference path: {reference_path}")

    # Save the image
    image_file.save(reference_path)
    print(f"Image saved at: {reference_path}")

    return reference_path


def get_reference_image_path(key):
    """Get the path to a reference image by its key.

    Args:
        key: The unique key of the reference image

    Returns:
        The path to the reference image or None if not found
    """
    print(f"Looking for reference image with key: {key}")

    # Check if the reference directory exists
    if not os.path.exists(REFERENCE_FOLDER):
        print(f"Reference folder does not exist: {REFERENCE_FOLDER}")
        return None

    # Check if the key folder exists
    key_folder = os.path.join(REFERENCE_FOLDER, key)
    print(f"Looking in key folder: {key_folder}")

    if not os.path.exists(key_folder):
        print(f"Key folder does not exist: {key_folder}")
        return None

    # Look for any file with the key as the filename (regardless of extension)
    pattern = os.path.join(key_folder, f"{key}.*")
    matching_files = glob.glob(pattern)
    print(f"Found matching files: {matching_files}")

    # Return the first matching file or None if no matches
    result = matching_files[0] if matching_files else None
    print(f"Returning reference path: {result}")
    return result


def list_reference_images():
    """List all reference images with their keys.

    Returns:
        A list of dictionaries with keys and file paths
    """
    # Ensure the reference directory exists
    if not os.path.exists(REFERENCE_FOLDER):
        return []

    # Get all subdirectories (key folders) in the reference directory
    key_folders = [f for f in os.listdir(REFERENCE_FOLDER)
                  if os.path.isdir(os.path.join(REFERENCE_FOLDER, f))]

    # Extract keys and file paths
    result = []
    for key in key_folders:
        key_folder = os.path.join(REFERENCE_FOLDER, key)
        # Look for the image file with the key name
        pattern = os.path.join(key_folder, f"{key}.*")
        matching_files = glob.glob(pattern)

        if matching_files:  # If we found a matching file
            result.append({"key": key, "path": matching_files[0]})

    return result
