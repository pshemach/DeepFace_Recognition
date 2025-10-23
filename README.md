# Face Recognition API

## Overview

This is a FastAPI-based API for face recognition and emotion detection, built using the [DeepFace](https://github.com/serengil/deepface) library. It allows users to:
- Upload a reference image to extract a 512-dimensional face embedding (`/upload_reference`).
- Compare an uploaded image with a reference embedding and detect facial expressions (`/compare_with_reference`).

The API uses the Facenet512 model for face recognition and supports cosine distance for face matching. It is deployed on a Linux server at `http://178.63.0.159:5056` and includes Swagger UI documentation at `/docs`.

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

## Deployment

### Step 1: Run the API
1. **Start the FastAPI Server**:
   ```bash
   cd ~/faceapi
   source venv/bin/activate
   nohup uvicorn main:app --host 0.0.0.0 --port 5056 > nohup.out 2>&1 &
   ```
   - Check logs:
     ```bash
     tail -f nohup.out
     ```

2. **Verify the API**:
   Access Swagger UI at `http://178.63.0.159:5056/docs`.

### Step 2: Configure Nginx (Optional)
1. **Install Nginx**:
   ```bash
   sudo apt update
   sudo apt install nginx -y
   ```

2. **Create Nginx Configuration**:
   Edit `/etc/nginx/sites-available/faceapi`:
   ```nginx
   server {
       listen 80;
       server_name 178.63.0.159;

       location / {
           proxy_pass http://127.0.0.1:5056;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }

       location /static/ {
           alias /home/youruser/faceapi/static/;
       }
   }
   ```

3. **Enable and Restart Nginx**:
   ```bash
   sudo ln -s /etc/nginx/sites-available/faceapi /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   ```

4. **Open Firewall Port**:
   ```bash
   sudo ufw allow 80
   sudo ufw status
   ```

### Step 3: HTTPS (Recommended)
1. **Install Certbot**:
   ```bash
   sudo apt install certbot python3-certbot-nginx -y
   ```

2. **Obtain SSL Certificate** (if you have a domain):
   ```bash
   sudo certbot --nginx -d your-domain.com
   ```

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
     curl -X POST -F "image=@/path/to/image.jpg" http://178.63.0.159:5056/upload_reference
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
           "threshold": 0.40
         },
         "emotions": {
           "dominant_emotion": "happy",
           "emotion_probabilities": {
             "angry": 0.05,
             "disgust": 0.01,
             "fear": 0.03,
             "happy": 0.80,
             "sad": 0.10,
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
          http://178.63.0.159:5056/compare_with_reference
     ```

### Swagger UI
- Access interactive API documentation at `http://178.63.0.159:5056/docs`.

## Troubleshooting

- **API Not Responding**:
  - Check Uvicorn process: `ps aux | grep uvicorn`
  - View logs: `tail -f ~/faceapi/nohup.out`
  - Ensure port 5056 is open: `sudo ufw allow 5056`
- **Invalid File Extension**:
  - Ensure uploaded images are jpg, jpeg, or png (defined in `ALLOWED_FILE_EXTENSIONS`).
- **JSON Parsing Errors**:
  - Verify `embedding` is a valid JSON array of 512 floats.
- **DeepFace Errors**:
  - Ensure server has 2GB+ RAM for Facenet512 and emotion detection.
  - If no face is detected, try setting `enforce_detection=False` in `src/pipeline/deepface_pipe.py` (may reduce accuracy).
- **Nginx Issues**:
  - Test configuration: `sudo nginx -t`
  - Check status: `sudo systemctl status nginx`

## Notes

- **Performance**: DeepFace is resource-intensive. Monitor server memory and CPU usage.
- **Security**: Add API key authentication for production (not implemented in current code).
- **File Cleanup**: Temporary files are deleted automatically, but monitor `UPLOAD_FOLDER` for disk space.
- **Dependencies**: Ensure `src/utils/file_utils.py`, `src/pipeline/deepface_pipe.py`, and `src/constant.py` are correctly implemented.

## License

This project is licensed under the MIT License.