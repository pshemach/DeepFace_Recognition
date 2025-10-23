import os
import numpy as np
import logging
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from werkzeug.utils import secure_filename
from deepface import DeepFace
import tempfile
from typing import List
from src.utils.file_utils import is_allowed
from src.pipeline.deepface_pipe import verify_faces, extract_embedding, detect_emotion
from src.utils.logger import logging
from src.constant import (
    UPLOAD_FOLDER
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

# Pydantic Models
class ReferenceResponse(BaseModel):
    embedding: List[float]

class CompareResponse(BaseModel):
    result: dict

@app.post("/upload_reference", response_model=ReferenceResponse)
async def upload_reference(image: UploadFile = File(...)):
    """Upload a reference image and return its face embedding."""
    logger.info("Received upload request for image: %s", image.filename)
    
    if not is_allowed(image.filename):
        logger.error("Invalid file extension for %s", image.filename)
        raise HTTPException(status_code=400, detail="Invalid file extension")

    # Save image temporarily
    temp_image = os.path.join(UPLOAD_FOLDER, secure_filename(image.filename))
    try:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        content = await image.read()
        with open(temp_image, "wb") as f:
            f.write(content)
        logger.info("Image saved temporarily at: %s", temp_image)

        # Extract embedding
        embedding = extract_embedding(image_path=temp_image)
        logger.info("Embedding extracted for image: %s", image.filename)

        return ReferenceResponse(embedding=embedding)
    except Exception as e:
        logger.error("Error processing image: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        try:
            if os.path.exists(temp_image):
                os.remove(temp_image)
                logger.info("Temporary file deleted: %s", temp_image)
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
    
    # Parse the embedding from JSON string - DIRECT LIST
    try:
        reference_embedding_list = json.loads(embedding)
        
        # Validate it's a list of numbers
        if not isinstance(reference_embedding_list, list):
            raise HTTPException(status_code=400, detail="Embedding must be a list")
        
        if not reference_embedding_list:
            raise HTTPException(status_code=400, detail="Embedding list cannot be empty")
        
        if not all(isinstance(x, (int, float)) for x in reference_embedding_list):
            raise HTTPException(status_code=400, detail="Embedding must contain only numbers")
            
        logger.info("Parsed embedding with %d dimensions", len(reference_embedding_list))
        
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in embedding parameter: %s", str(e))
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
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
        matching = verify_faces(temp_path, reference_embedding_list)
        emotions = detect_emotion(temp_path)
        
        result = { "matching": matching, "emotions":emotions}
        
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
