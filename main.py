import os
import numpy as np
import logging
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from werkzeug.utils import secure_filename
from deepface import DeepFace
import tempfile
from typing import List, Any
# Configure logging
from faceMatch.constant import (
    UPLOAD_FOLDER, FACE_MODEL, SELECTED_MODEL_KEY, MATRICES, ALLOWED_FILE_EXTENSIONS
)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Face Recognition API", version="1.0.0")

# Enable CORS for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utility Functions
ALLOWED_EXTENSIONS = {ext.lower() for ext in ALLOWED_FILE_EXTENSIONS}

def is_allowed(filename: str) -> bool:
    """Check if the file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def verify_faces(image_path: str, reference_embedding: List[float]):
    """Compare an image with a reference embedding."""
    try:
        logger.info("Verifying face with DeepFace:")
        logger.info("Image path: %s", image_path)
        logger.info("Model: %s", FACE_MODEL[SELECTED_MODEL_KEY])
        logger.info("Distance metric: %s", MATRICES[2])

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
            enforce_detection=False  # Don't enforce face detection
        )[0]['embedding']
        uploaded_embedding = np.array(uploaded_embedding)

        # Convert reference embedding to numpy array
        reference_embedding = np.array(reference_embedding)

        # Compute distance
        if MATRICES[2] == "cosine":
            distance = np.dot(reference_embedding, uploaded_embedding) / (
                np.linalg.norm(reference_embedding) * np.linalg.norm(uploaded_embedding)
            )
            distance = 1 - distance  # Convert similarity to distance
        elif MATRICES[2] == "euclidean":
            distance = np.linalg.norm(reference_embedding - uploaded_embedding)
        elif MATRICES[2] == "euclidean_l2":
            distance = np.sqrt(np.sum((reference_embedding - uploaded_embedding) ** 2))

        # Threshold (adjust based on model and metric)
        threshold = 0.4  # Example for Facenet512 with cosine
        verified = bool(distance <= threshold)  # Convert to Python bool

        result = {
            "verified": verified,
            "distance": round(float(distance), 2),  # Convert to Python float
            "threshold": round(float(threshold), 2)  # Convert to Python float
        }
        logger.info("Verification result: %s", result)
        return result
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error("Error in verify_faces: %s", str(e))
        logger.error("Traceback: %s", error_traceback)
        return {"error": f"DeepFace error: {str(e)}"}

# Pydantic Models
class ReferenceResponse(BaseModel):
    embedding: List[float]

class CompareResponse(BaseModel):
    result: dict

# Endpoints

@app.post("/upload_reference", response_model=ReferenceResponse)
async def upload_reference(image: UploadFile = File(...)):
    """Upload a reference image and return its face embedding."""
    logger.info("Received upload request for image: %s", image.filename)
    
    if not is_allowed(image.filename):
        logger.error("Invalid file extension for %s", image.filename)
        raise HTTPException(status_code=400, detail="Invalid file extension")

    # Save image temporarily
    temp_path = os.path.join(tempfile.gettempdir(), secure_filename(image.filename))
    try:
        content = await image.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        logger.info("Image saved temporarily at: %s", temp_path)

        # Extract embedding
        embedding = DeepFace.represent(
            img_path=temp_path,
            model_name=FACE_MODEL[SELECTED_MODEL_KEY],
            enforce_detection=True
        )[0]['embedding']
        logger.info("Embedding extracted for image: %s", image.filename)

        return ReferenceResponse(embedding=embedding)
    except Exception as e:
        logger.error("Error processing image: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.info("Temporary file deleted: %s", temp_path)
        except Exception as e:
            logger.warning("Error removing temp file: %s", str(e))

@app.post("/compare_with_reference", response_model=CompareResponse)
async def compare_with_reference(
    image: UploadFile = File(...),
    embedding: str = Form(...)
):
    """Compare an uploaded image with a provided reference embedding."""
    logger.info("Received compare request for image: %s", image.filename)

    if not is_allowed(image.filename):
        logger.error("Invalid file extension for %s", image.filename)
        raise HTTPException(status_code=400, detail="Invalid file extension")
    
    # Parse the embedding from JSON string
    try:
        # Try to parse as JSON object with "embedding" key
        embedding_data = json.loads(embedding)
        
        # Check if it's wrapped in {"embedding": [...]} format
        if isinstance(embedding_data, dict) and "embedding" in embedding_data:
            reference_embedding_list = embedding_data["embedding"]
        # Or if it's directly a list
        elif isinstance(embedding_data, list):
            reference_embedding_list = embedding_data
        else:
            raise HTTPException(status_code=400, detail="Invalid embedding format")
        
        # Validate it's a list of numbers
        if not isinstance(reference_embedding_list, list):
            raise HTTPException(status_code=400, detail="Embedding must be a list")
        
        if not all(isinstance(x, (int, float)) for x in reference_embedding_list):
            raise HTTPException(status_code=400, detail="Embedding must contain only numbers")
            
        logger.info("Parsed embedding with %d dimensions", len(reference_embedding_list))
        
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in embedding parameter: %s", str(e))
        raise HTTPException(status_code=400, detail=f"Invalid JSON format for embedding: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error parsing embedding: %s", str(e))
        raise HTTPException(status_code=400, detail=f"Error parsing embedding: {str(e)}")

    # Save uploaded image temporarily
    temp_path = os.path.join(UPLOAD_FOLDER, secure_filename(image.filename))
    try:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        content = await image.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        logger.info("Image saved temporarily at: %s", temp_path)
        
        # Compare using verify_faces - pass the list directly
        result = verify_faces(temp_path, reference_embedding_list)
        
        if isinstance(result, dict) and 'error' in result:
            logger.error("Error in verify_faces: %s", result['error'])
            raise HTTPException(status_code=500, detail=result['error'])
        
        return CompareResponse(result=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing image with DeepFace: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.info("Temporary file deleted: %s", temp_path)
        except Exception as e:
            logger.warning("Error deleting temporary file: %s", str(e))

# Initialize upload folder on startup
@app.on_event("startup")
async def startup_event():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logger.info("Application started, upload folder initialized")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5056, log_level="info")
