import os
import sqlite3
import numpy as np
from deepface import DeepFace
from werkzeug.utils import secure_filename
from faceMatch.constant import REFERENCE_FOLDER, DATABASE_PATH

def save_reference_image(image_file, key):
    """Save a reference image's face embedding with the given key in the database.

    Args:
        image_file: The uploaded image file object
        key: The unique key to identify this reference embedding

    Returns:
        The key if embedding is saved successfully, None if no face detected
    """
    print(f"Saving reference embedding with key: {key}")
    print(f"Image filename: {image_file.filename}")

    # Save temporarily to check for face and extract embedding
    temp_path = os.path.join(tempfile.gettempdir(), secure_filename(image_file.filename))
    image_file.save(temp_path)
    print(f"Image saved temporarily at: {temp_path}")

    # Face detection and embedding extraction
    try:
        face_objs = DeepFace.extract_faces(
            img_path=temp_path,
            detector_backend='opencv',
            enforce_detection=True
        )

        if not face_objs:
            print("No face detected.")
            os.remove(temp_path)
            return None

        print(f"Face detected. Confidence: {face_objs[0]['confidence']}")

        # Extract embedding
        embedding = DeepFace.represent(
            img_path=temp_path,
            model_name=FACE_MODEL[SELECTED_MODEL_KEY],
            enforce_detection=True
        )[0]['embedding']
        embedding_blob = np.array(embedding).tobytes()  # Convert to BLOB
        filename = secure_filename(image_file.filename)

        # Save embedding to database
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO face_embeddings (key, filename, embedding) VALUES (?, ?, ?)",
            (key, filename, embedding_blob)
        )
        conn.commit()
        conn.close()

        print(f"Embedding saved for key: {key}")
        os.remove(temp_path)
        return key

    except Exception as e:
        print(f"Error processing image or saving embedding: {str(e)}")
        try:
            os.remove(temp_path)
        except Exception as e:
            print(f"Error removing temp file: {e}")
        return None

def get_reference_image_path(key):
    """Get the embedding for a reference image by its key from the database.

    Args:
        key: The unique key of the reference embedding

    Returns:
        The embedding as a numpy array or None if not found
    """
    print(f"Looking for reference embedding with key: {key}")

    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT embedding FROM face_embeddings WHERE key = ?", (key,))
        result = cursor.fetchone()
        conn.close()

        if result:
            embedding_blob = result[0]
            embedding = np.frombuffer(embedding_blob, dtype=np.float64)
            print(f"Embedding found for key: {key}")
            return embedding
        else:
            print(f"No embedding found for key: {key}")
            return None
    except Exception as e:
        print(f"Error retrieving embedding: {str(e)}")
        return None

def list_reference_images():
    """List all reference embeddings with their keys.

    Returns:
        A list of dictionaries with keys and filenames
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT key, filename FROM face_embeddings")
        results = cursor.fetchall()
        conn.close()

        return [{"key": key, "filename": filename} for key, filename in results]
    except Exception as e:
        print(f"Error listing reference embeddings: {str(e)}")
        return []

def delete_reference_image(key):
    """Delete a reference embedding with the given key from the database.

    Args:
        key: The unique key identifying the reference embedding to delete

    Returns:
        bool: True if the embedding was successfully deleted, False otherwise
    """
    print(f"Attempting to delete reference embedding with key: {key}")

    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM face_embeddings WHERE key = ?", (key,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()

        if deleted:
            print(f"Successfully deleted reference embedding for key: {key}")
            return True
        else:
            print(f"No embedding found for key: {key}")
            return False
    except Exception as e:
        print(f"Error deleting reference embedding: {str(e)}")
        return False