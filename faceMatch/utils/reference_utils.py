import os
import glob
import tempfile
from werkzeug.utils import secure_filename
from faceMatch.constant import REFERENCE_FOLDER


def save_reference_image(image_file, key):
    """Save a reference image with the given key.

    Args:
        image_file: The uploaded image file object
        key: The unique key to identify this reference image

    Returns:
        The path to the saved reference image or None if no face detected
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

    # Save the image temporarily to check for faces
    temp_path = os.path.join(tempfile.gettempdir(), reference_filename)
    image_file.save(temp_path)
    print(f"Image saved temporarily at: {temp_path}")

    # Check if a face can be detected in the image
    try:
        from deepface import DeepFace
        # Try to extract faces from the image
        face_objs = DeepFace.extract_faces(
            img_path=temp_path,
            detector_backend='opencv',  # Using opencv as it's faster
            enforce_detection=True  # Enforce face detection
        )

        # If we get here, at least one face was detected
        if len(face_objs) > 0:
            print(f"Face detected in the image. Confidence: {face_objs[0]['confidence']}")

            # Now save the image to the reference folder
            # We need to reopen the file since we've already read it
            image_file.seek(0)
            image_file.save(reference_path)
            print(f"Image saved at: {reference_path}")

            # Clean up the temporary file
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"Error removing temporary file: {str(e)}")

            return reference_path
        else:
            print("No face detected in the image.")
            # Clean up the temporary file
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"Error removing temporary file: {str(e)}")
            return None
    except Exception as e:
        print(f"Error detecting face: {str(e)}")
        # Clean up the temporary file
        try:
            os.remove(temp_path)
        except Exception as e:
            print(f"Error removing temporary file: {str(e)}")
        return None


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


def delete_reference_image(key):
    """Delete a reference image folder with the given key.

    Args:
        key: The unique key identifying the reference image folder to delete

    Returns:
        bool: True if the folder was successfully deleted, False otherwise
    """
    print(f"Attempting to delete reference image with key: {key}")

    # Check if the key folder exists
    key_folder = os.path.join(REFERENCE_FOLDER, key)
    print(f"Looking for key folder: {key_folder}")

    if not os.path.exists(key_folder):
        print(f"Key folder not found: {key_folder}")
        return False

    try:
        # Delete all files in the folder
        for filename in os.listdir(key_folder):
            file_path = os.path.join(key_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")

        # Remove the directory
        os.rmdir(key_folder)
        print(f"Successfully deleted reference folder: {key_folder}")
        return True
    except Exception as e:
        print(f"Error deleting reference folder: {str(e)}")
        return False
