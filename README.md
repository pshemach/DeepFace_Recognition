# Face Recognition API

## Features

- **Face Embedding Extraction**: Upload an image to extract a 512-dimensional face embedding.
- **Face Comparison**: Compare an uploaded image with a reference embedding to verify if faces match, using a model-specific threshold.
- **Emotion Detection**: Detect facial expressions (e.g., happy, sad, angry) with probability scores.
- **Combined Response**: The `/compare_with_reference` endpoint returns both face matching and emotion detection results in a single JSON response.
- **CORS Support**: Configured for cross-origin requests.
- **Error Handling**: Robust validation for file extensions, JSON parsing, and embedding lengths.

## Prerequisites

- **Python**: 3.8 or higher
- **Linux Server**: Ubuntu or similar (tested on Ubuntu 20.04)
- **Dependencies**:
  - fastapi
  - uvicorn
  - python-multipart
  - deepface
  - numpy
  - pillow
  - pyyaml
  - werkzeug
  - scipy
- **Nginx**: For reverse proxy (optional, recommended for production)
- **Hardware**: Minimum 2GB RAM (DeepFace with Facenet512 is resource-intensive)

## Installation

1. **Clone the Repository** (if using version control):

   ```bash
   git clone <repository-url>
   cd faceapi
   ```

2. **Set Up Virtual Environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:

   ```bash
   echo -e "fastapi\nuvicorn\npython-multipart\ndeepface\nnumpy\npillow\npyyaml\nwerkzeug\nscipy" > requirements.txt
   pip install -r requirements.txt
   ```

4. **Directory Structure**:
   Ensure the following structure in `~/faceapi`:

   ```
   faceapi/
   ├── main.py
   ├── src/
   │   ├── constant.py
   │   ├── pipeline/
   │   │   └── deepface_pipe.py
   │   └── utils/
   │       └── file_utils.py
   ├── static/ (optional, for Nginx)
   └── requirements.txt
   ```

5. **Configure `constant.py`**:
   Ensure `UPLOAD_FOLDER` and other constants (e.g., `FACE_MODEL`, `SELECTED_MODEL_KEY`, `MATRICES`, `ALLOWED_FILE_EXTENSIONS`) are defined in `src/constant.py`.

## Usage

### Endpoints

1. **POST /upload_reference**

   - **Description**: Upload an image to extract a 512-dimensional face embedding.
   - **Request**: `multipart/form-data`
     - `image`: Image file (jpg, jpeg, png)
   - **Response**:
     ```json
     {
       "embedding": [2.255829334259033, 0.04911996051669121, ...]
     }
     ```
   - **Example**:
     ```bash
     curl -X POST -F "image=@/path/to/image.jpg" http://127.0.0.1:5056/upload_reference
     ```

2. **POST /compare_with_reference**
   - **Description**: Compare an uploaded image with a reference embedding and detect facial emotions. Returns a combined result with matching and emotion data.
   - **Request**: `multipart/form-data`
     - `image`: Image file (jpg, jpeg, png)
     - `embedding`: JSON-encoded string of 512 float values
   - **Response**:
     ```json
     {
       "result": {
         "matching": {
           "verified": true,
           "distance": 0.1234,
           "threshold": 0.4
         },
         "emotions": {
           "dominant_emotion": "happy",
           "emotion_probabilities": {
             "angry": 0.05,
             "disgust": 0.01,
             "fear": 0.03,
             "happy": 0.8,
             "sad": 0.1,
             "surprise": 0.01,
             "neutral": 0.05
           }
         }
       }
     }
     ```
   - **Example**:
     Save embedding in `embedding.json`:
     ```json
     [2.255829334259033, 0.04911996051669121, ...]
     ```
     ```bash
     curl -X POST -F "image=@/path/to/test_image.jpg" \
          -F "embedding=$(cat embedding.json)" \
          http://127.0.0.1:5056/compare_with_reference
     ```
