Face Recognition API Documentation
Overview
The Face Recognition API provides endpoints for managing and comparing face embeddings to perform facial recognition. It allows clients to upload reference images, store their face embeddings and images, compare new images against stored references, list all stored references, and delete references. The API uses the Facenet512 model with cosine distance for face verification, ensuring accurate and efficient face matching.
Base URL: http://<host>:5056 (replace <host> with the production server address)
Authentication: None (add JWT or API key authentication in production if required).
Supported File Formats: JPEG, PNG
Endpoints

1. Upload Reference Image
   Uploads an image, extracts its face embedding, and stores both the embedding and image in the database with a unique key.

Method: POST
Path: /upload_reference
Content-Type: multipart/form-data
Request Body:
image (file, required): The image file containing a single face (JPEG or PNG).

Response:
Status: 200 OK
Body:{
"success": boolean,
"message": string,
"key": string
}

success: Indicates if the upload was successful.
message: Descriptive message (e.g., "Reference embedding saved with key: ").
key: Unique alphanumeric identifier for the stored embedding and image.

Error Responses:
400 Bad Request: Invalid file extension or no face detected.{
"detail": "Invalid file extension"
}

or{
"detail": "No face detected in the image. Please upload an image with a clear face."
}

500 Internal Server Error: Processing or database error.{
"detail": "Error processing image: <error message>"
}

Example:curl -X POST -F "image=@/path/to/image.jpg" http://localhost:5056/upload_reference

Response:{
"success": true,
"message": "Reference embedding saved with key: 80a728a38eb649bdbf91a1c890bbb659",
"key": "80a728a38eb649bdbf91a1c890bbb659"
}

2. Compare with Reference
   Compares an uploaded image with a stored reference image identified by a key, using DeepFace's verification to determine if the faces match.

Method: POST
Path: /compare_with_reference
Content-Type: multipart/form-data
Request Body:
image (file, required): The image file to compare (JPEG or PNG).
reference_key (string, required): The unique key of the stored reference image.

Response:
Status: 200 OK
Body:{
"reference_key": string,
"result": {
"verified": boolean,
"distance": number,
"threshold": number
}
}

reference_key: The provided key for the reference image.
result.verified: Whether the faces match (true if distance ≤ threshold).
result.distance: The computed distance between face embeddings (lower is more similar).
result.threshold: The threshold used for verification (0.4 for Facenet512 with cosine distance).

Error Responses:
400 Bad Request: Invalid file extension or invalid reference key format.{
"detail": "Invalid file extension"
}

or{
"detail": "Reference key must be alphanumeric"
}

404 Not Found: Reference key not found.{
"detail": "No reference image found with key: <reference_key>"
}

500 Internal Server Error: Processing or DeepFace error.{
"detail": "Error processing image: <error message>"
}

Example:curl -X POST -F "image=@/path/to/image.jpg" -F "reference_key=80a728a38eb649bdbf91a1c890bbb659" http://localhost:5056/compare_with_reference

Response:{
"reference_key": "80a728a38eb649bdbf91a1c890bbb659",
"result": {
"verified": true,
"distance": 0.0,
"threshold": 0.4
}
}

3. List References
   Lists all stored reference images with their keys and filenames.

Method: GET
Path: /list_references
Request Parameters: None
Response:
Status: 200 OK
Body:{
"count": integer,
"references": [
{
"key": string,
"filename": string
}
]
}

count: Number of stored references.
references: Array of objects containing the key and original filename for each reference.

Error Responses:
500 Internal Server Error: Database error.{
"detail": "Error listing embeddings: <error message>"
}

Example:curl -X GET http://localhost:5056/list_references

Response:{
"count": 2,
"references": [
{
"key": "80a728a38eb649bdbf91a1c890bbb659",
"filename": "image1.jpg"
},
{
"key": "456e7890e12b34c5a678426614174001",
"filename": "image2.png"
}
]
}

4. Delete Reference
   Deletes a stored reference image and its embedding by its key.

Method: DELETE
Path: /delete_reference/{key}
Path Parameters:
key (string, required): The unique key of the reference to delete.

Response:
Status: 200 OK
Body:{
"success": boolean,
"message": string
}

success: Indicates if the deletion was successful.
message: Descriptive message (e.g., "Reference embedding with key '' has been deleted").

Error Responses:
400 Bad Request: Invalid key format.{
"detail": "Invalid reference key"
}

404 Not Found: Reference key not found.{
"detail": "No reference embedding found with key: <key>"
}

500 Internal Server Error: Database error.{
"detail": "Error deleting embedding: <error message>"
}

Example:curl -X DELETE http://localhost:5056/delete_reference/80a728a38eb649bdbf91a1c890bbb659

Response:{
"success": true,
"message": "Reference embedding with key '80a728a38eb649bdbf91a1c890bbb659' has been deleted"
}
