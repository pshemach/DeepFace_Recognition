import os
import uuid
import sqlite3
import numpy as np
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from werkzeug.utils import secure_filename
from deepface import DeepFace
from PIL import Image
import io
import tempfile
import glob
from typing import List
from faceMatch.constant import (
    UPLOAD_FOLDER, REFERENCE_FOLDER, DATABASE_PATH,
    FACE_MODEL, SELECTED_MODEL_KEY, MATRICES, ALLOWED_FILE_EXTENSIONS
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

# Database initialization
def init_db():
    """Initialize the SQLite database and create the embeddings table."""
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            key TEXT PRIMARY_KEY,
            filename TEXT,
            embedding BLOB,
            image BLOB
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialized at %s", DATABASE_PATH)

# Utility Functions
ALLOWED_EXTENSIONS = {ext.lower() for ext in ALLOWED_FILE_EXTENSIONS}

def is_allowed(filename: str) -> bool:
    """Check if the file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def verify_faces(image1, image2):
    try:
        # Print detailed information about the input images
        logger.info("Verifying faces with DeepFace:")
        logger.info("Image 1 path: %s", image1)
        logger.info("Image 2 path: %s", image2)
        logger.info("Model: %s", FACE_MODEL[SELECTED_MODEL_KEY])
        logger.info("Distance metric: %s", MATRICES[2])

        # Check if the image files exist
        if not os.path.exists(image1):
            return {"error": f"Reference image file not found: {image1}"}
        if not os.path.exists(image2):
            return {"error": f"Comparison image file not found: {image2}"}

        # Try to open the images to verify they are valid
        try:
            from PIL import Image
            img1 = Image.open(image1)
            img1.verify()  # Verify that it's a valid image
            logger.info("Image 1 is valid")

            img2 = Image.open(image2)
            img2.verify()  # Verify that it's a valid image
            logger.info("Image 2 is valid")
        except Exception as img_error:
            logger.error("Invalid image file: %s", str(img_error))
            return {"error": f"Invalid image file: {str(img_error)}"}

        # Call DeepFace.verify to get the full result dictionary
        # Set enforce_detection=False to handle images where faces can't be detected
        result = DeepFace.verify(
            image1, image2,
            model_name=FACE_MODEL[SELECTED_MODEL_KEY],
            distance_metric=MATRICES[2],
            enforce_detection=False  # Don't enforce face detection
        )

        # Print the result for debugging
        logger.info("DeepFace verification result: %s", result)

        # Convert NumPy boolean to Python bool
        result["verified"] = bool(result["verified"])
        return result
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error("Error in verify_faces: %s", str(e))
        logger.error("Traceback: %s", error_traceback)
        return {"error": f"DeepFace error: {str(e)}"}

async def save_reference_image(image_file: UploadFile, key: str) -> str:
    """Save a reference image's face embedding and image data with the given key in the database.

    Args:
        image_file: The uploaded image file object
        key: The unique key to identify this reference embedding

    Returns:
        The key if embedding and image are saved successfully, None if no face detected
    """
    logger.info("Saving reference embedding with key: %s, filename: %s", key, image_file.filename)
    
    if not is_allowed(image_file.filename):
        logger.error("Invalid file extension for %s", image_file.filename)
        raise HTTPException(status_code=400, detail="Invalid file extension")

    # Save temporarily to check for face and extract embedding
    temp_path = os.path.join(tempfile.gettempdir(), secure_filename(image_file.filename))
    try:
        content = await image_file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        logger.info("Image saved temporarily at: %s", temp_path)

        # Face detection and embedding extraction
        face_objs = DeepFace.extract_faces(
            img_path=temp_path,
            detector_backend='opencv',
            enforce_detection=True
        )

        if not face_objs:
            logger.warning("No face detected in image: %s", image_file.filename)
            os.remove(temp_path)
            return None

        logger.info("Face detected. Confidence: %s", face_objs[0]['confidence'])

        # Extract embedding
        embedding = DeepFace.represent(
            img_path=temp_path,
            model_name=FACE_MODEL[SELECTED_MODEL_KEY],
            enforce_detection=True
        )[0]['embedding']
        embedding_blob = np.array(embedding).tobytes()  # Convert to BLOB
        filename = secure_filename(image_file.filename)

        # Save embedding and image to database
        conn = sqlite3.connect(DATABASE_PATH)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO face_embeddings (key, filename, embedding, image) VALUES (?, ?, ?, ?)",
                (key, filename, embedding_blob, content)
            )
            conn.commit()
            logger.info("Embedding and image saved for key: %s", key)
        finally:
            conn.close()

        return key
    except Exception as e:
        logger.error("Error processing image or saving embedding: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.info("Temporary file deleted: %s", temp_path)
        except Exception as e:
            logger.warning("Error removing temp file: %s", str(e))

def get_reference_image_path(key: str) -> str:
    """Get the reference image path by saving it temporarily from the database.

    Args:
        key: The unique key of the reference embedding

    Returns:
        The temporary path to the reference image or None if not found
    """
    logger.info("Looking for reference image with key: %s", key)

    try:
        conn = sqlite3.connect(DATABASE_PATH)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT image, filename FROM face_embeddings WHERE key = ?", (key,))
            result = cursor.fetchone()
            if result:
                image_blob, filename = result
                # Save image to temporary file
                temp_path = os.path.join(tempfile.gettempdir(), secure_filename(filename))
                with open(temp_path, "wb") as f:
                    f.write(image_blob)
                logger.info("Reference image saved temporarily at: %s", temp_path)
                return temp_path
            else:
                logger.warning("No reference image found for key: %s", key)
                return None
        finally:
            conn.close()
    except Exception as e:
        logger.error("Error retrieving reference image: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error retrieving reference image: {str(e)}")

def list_reference_images() -> List[dict]:
    """List all reference embeddings with their keys.

    Returns:
        A list of dictionaries with keys and filenames
    """
    logger.info("Listing all reference embeddings")
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT key, filename FROM face_embeddings")
            results = cursor.fetchall()
            return [{"key": key, "filename": filename} for key, filename in results]
        finally:
            conn.close()
    except Exception as e:
        logger.error("Error listing reference embeddings: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error listing embeddings: {str(e)}")

def delete_reference_image(key: str) -> bool:
    """Delete a reference embedding and image with the given key from the database.

    Args:
        key: The unique key identifying the reference embedding to delete

    Returns:
        bool: True if the embedding was successfully deleted, False otherwise
    """
    logger.info("Attempting to delete reference embedding with key: %s", key)
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM face_embeddings WHERE key = ?", (key,))
            deleted = cursor.rowcount > 0
            conn.commit()
            if deleted:
                logger.info("Successfully deleted reference embedding for key: %s", key)
                return True
            else:
                logger.warning("No embedding found for key: %s", key)
                return False
        finally:
            conn.close()
    except Exception as e:
        logger.error("Error deleting reference embedding: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error deleting embedding: {str(e)}")

# Pydantic Models for Request/Response Validation
class ReferenceResponse(BaseModel):
    success: bool
    message: str
    key: str

class CompareResponse(BaseModel):
    reference_key: str
    result: dict

class ReferenceListResponse(BaseModel):
    count: int
    references: List[dict]

class DeleteResponse(BaseModel):
    success: bool
    message: str

# Database Dependency
def get_db():
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        yield conn
    finally:
        conn.close()

# Endpoints
@app.post("/upload_reference", response_model=ReferenceResponse)
async def upload_reference(image: UploadFile = File(...)):
    """Upload a reference image and store its face embedding and image data."""
    logger.info("Received upload request for image: %s", image.filename)
    
    if not is_allowed(image.filename):
        logger.error("Invalid file extension for %s", image.filename)
        raise HTTPException(status_code=400, detail="Invalid file extension")

    # Generate a unique key
    key = str(uuid.uuid4()).replace('-', '')
    
    # Save the reference embedding and image
    result = await save_reference_image(image, key)
    
    if result is None:
        logger.warning("No face detected in uploaded image")
        raise HTTPException(
            status_code=400,
            detail="No face detected in the image. Please upload an image with a clear face."
        )
    
    return ReferenceResponse(
        success=True,
        message=f"Reference embedding saved with key: {key}",
        key=key
    )

@app.post("/compare_with_reference", response_model=CompareResponse)
async def compare_with_reference(
    image: UploadFile = File(...),
    reference_key: str = Form(...)
):
    """Compare an uploaded image with a stored reference image using verify_faces."""
    logger.info("Received compare request with reference_key: %s, image: %s", reference_key, image.filename)
    
    # Validate input
    if not is_allowed(image.filename):
        logger.error("Invalid file extension for %s", image.filename)
        raise HTTPException(status_code=400, detail="Invalid file extension")
    
    if not reference_key.isalnum():
        logger.error("Invalid reference key format: %s", reference_key)
        raise HTTPException(status_code=400, detail="Reference key must be alphanumeric")
    
    # Get reference image path
    reference_image_path = get_reference_image_path(reference_key)
    if reference_image_path is None:
        logger.warning("No reference image found for key: %s", reference_key)
        raise HTTPException(status_code=404, detail=f"No reference image found with key: {reference_key}")
    
    # Save uploaded image temporarily
    temp_path = os.path.join(UPLOAD_FOLDER, secure_filename(image.filename))
    try:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        content = await image.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        logger.info("Image saved temporarily at: %s", temp_path)
        
        # Compare using verify_faces
        result = verify_faces(reference_image_path, temp_path)
        
        # Check for errors in result
        if isinstance(result, dict) and 'error' in result:
            logger.error("Error in verify_faces: %s", result['error'])
            raise HTTPException(status_code=500, detail=result['error'])
        
        # Ensure verified is a Python bool
        result["verified"] = bool(result["verified"])
        result["distance"] = round(float(result["distance"]), 2)
        result["threshold"] = round(float(result["threshold"]), 2)
        
        logger.info("Comparison result: %s", result)
        
        return CompareResponse(
            reference_key=reference_key,
            result=result
        )
    except Exception as e:
        logger.error("Error processing image with DeepFace: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        # Clean up temporary files
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.info("Temporary file deleted: %s", temp_path)
            if os.path.exists(reference_image_path):
                os.remove(reference_image_path)
                logger.info("Reference image file deleted: %s", reference_image_path)
        except Exception as e:
            logger.warning("Error deleting temporary file: %s", str(e))

@app.get("/list_references", response_model=ReferenceListResponse)
async def list_references():
    """List all stored reference embeddings."""
    logger.info("Listing references")
    references = list_reference_images()
    return ReferenceListResponse(
        count=len(references),
        references=[{"key": ref["key"], "filename": ref["filename"]} for ref in references]
    )

@app.delete("/delete_reference/{key}", response_model=DeleteResponse)
async def delete_reference(key: str):
    """Delete a reference embedding and image by its key."""
    logger.info("Received delete request for key: %s", key)
    
    if not key.isalnum():
        logger.error("Invalid reference key format: %s", key)
        raise HTTPException(status_code=400, detail="Invalid reference key")
    
    success = delete_reference_image(key)
    if success:
        return DeleteResponse(
            success=True,
            message=f"Reference embedding with key '{key}' has been deleted"
        )
    else:
        logger.warning("No reference embedding found for key: %s", key)
        raise HTTPException(status_code=404, detail=f"No reference embedding found with key: {key}")

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logger.info("Application started, database and upload folder initialized")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5056, log_level="info")