# DeepFace Recognition System

A facial recognition system built using DeepFace that allows for direct face comparison and reference-based face verification.

## Features

- **Direct Face Comparison**: Compare two uploaded images to determine if they contain the same person.
- **Reference-Based Comparison**: Compare an uploaded image against a stored reference image.
- **Reference Management**: Upload, list, and delete reference images.
- **Face Detection Validation**: Ensures that reference images contain detectable faces.
- **Multiple Face Recognition Models**: Supports various models including VGG-Face, Facenet, Facenet512, and more.
- **Flexible Distance Metrics**: Supports cosine, euclidean, and euclidean_l2 distance metrics.

## Project Structure

```
DeepFace_Recognition/
├── data/
│   └── reference_images/  # Stores reference images in folders by key
├── faceMatch/
│   ├── constant/         # Constants and configuration
│   ├── pipeline/         # Face verification pipeline
│   └── utils/            # Utility functions
│       ├── file_utils.py    # File handling utilities
│       ├── image_utils.py   # Image processing utilities
│       └── reference_utils.py # Reference image management
├── static/              # Static files for the web interface
├── templates/           # HTML templates
├── app.py              # Main application
├── demo.py             # Demo application
└── requirements.txt    # Project dependencies
```

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/DeepFace_Recognition.git
   cd DeepFace_Recognition
   ```

2. Create a virtual environment and activate it:

   ```
   python -m venv face_venv
   source face_venv/bin/activate  # On Windows: face_venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:

   ```
   python app.py
   ```

2. Open your web browser and navigate to:

   ```
   http://localhost:5056
   ```

3. Use the web interface to:
   - Compare two faces directly
   - Upload reference images with unique keys
   - Compare uploaded images with stored references
   - View and manage reference images

## API Endpoints

- `POST /compare_faces`: Compare two uploaded images
- `POST /upload_reference`: Upload a reference image with a key
- `POST /compare_with_reference`: Compare an uploaded image with a stored reference
- `GET /list_references`: List all stored reference images
- `GET /reference_image/<key>`: Get a specific reference image
- `DELETE /delete_reference/<key>`: Delete a reference image

## Reference Image Management

Reference images are stored in the `data/reference_images` directory, organized in folders by their unique keys. Each reference image is validated to ensure it contains a detectable face before being stored.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
