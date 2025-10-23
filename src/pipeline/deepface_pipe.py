from deepface import DeepFace
from src.constant import FACE_MODEL, SELECTED_MODEL_KEY, MATRICES, SELECTED_MATRIX_KEY, THRESHOLD
from src.utils.logger import logging
import os 
import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances

logger = logging.getLogger(__name__)

def extract_embedding(image_path: str) -> List[float]:
    embedding = DeepFace.represent(
            img_path=image_path,
            model_name=FACE_MODEL[SELECTED_MODEL_KEY],
            enforce_detection=True
        )[0]['embedding']
    return embedding

def verify_faces(image_path: str, reference_embedding: List[float]):
    """Compare an image with a reference embedding."""
    try:
        # Check if the image file exists
        if not os.path.exists(image_path):
            return {"error": f"Comparison image file not found: {image_path}"}

        # Validate image
        try:
            from PIL import Image
            img = Image.open(image_path)
            img.verify()  # Verify it's a valid image
            logger.info("Image is valid")
        except Exception as img_error:
            logger.error("Invalid image file: %s", str(img_error))
            return {"error": f"Invalid image file: {str(img_error)}"}

        # Extract embedding for uploaded image
        uploaded_embedding = DeepFace.represent(
            img_path=image_path,
            model_name=FACE_MODEL[SELECTED_MODEL_KEY],
            enforce_detection=True
        )[0]['embedding']
        
        # Convert embedding to 2D numpy array
        uploaded_embedding = np.array(uploaded_embedding).reshape(-1,1)
        reference_embedding = np.array(reference_embedding).reshape(-1,1)

        # Compute distance
        if MATRICES[SELECTED_MATRIX_KEY] == "cosine":
            distance = cosine_distances(reference_embedding, uploaded_embedding)[0,0]
           
        elif MATRICES[SELECTED_MATRIX_KEY] == "euclidean_l2":
            distance = euclidean_distances(reference_embedding, uploaded_embedding)[0,0]
        else:
            logger.error(f"Matrix key {SELECTED_MATRIX_KEY} mismatch with matrix checking cosine and euclidean_l2")
            raise Exception("Matrix Key not matching")
        
        # Threshold (adjust based on model and metric)
        verified = bool(distance <= THRESHOLD) 

        result = {
            "verified": verified,
            "distance": round(float(distance), 2),
            "threshold": round(float(THRESHOLD), 2)  
        }
        logger.info("Verification result: %s", result)
        return result
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error("Error in verify_faces: %s", str(e))
        logger.error("Traceback: %s", error_traceback)
        return {"error": f"DeepFace error: {str(e)}"}


def detect_emotion(image_path: str) -> Dict:
    """Detect facial expression (emotion) from an image."""
    try:
        logger.info("Detecting emotion for image: %s", image_path)
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['emotion'],
            enforce_detection=True
        )[0]
        
        # Convert NumPy types to Python types
        emotion_probs = {
            key: round(float(value), 4) for key, value in result['emotion'].items()
        }
        dominant_emotion = result['dominant_emotion']
        
        response = {
            "dominant_emotion": dominant_emotion,
            "emotion_probabilities": emotion_probs
        }
        logger.info("Emotion detection result: %s", response)
        return response
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error("Error in detect_emotion: %s", str(e))
        logger.error("Traceback: %s", error_traceback)
        return {"error": f"Emotion detection error: {str(e)}"}
