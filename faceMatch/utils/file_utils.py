import os
import shutil
import tempfile
import time
import uuid
from werkzeug.utils import secure_filename
from faceMatch.constant import ALLOWED_FILE_EXTENSIONS

ALLOWED_EXTENSIONS = {ext.lower() for ext in ALLOWED_FILE_EXTENSIONS}
previous_temp_dirs = []


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


def unique_filename(image_name):
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
