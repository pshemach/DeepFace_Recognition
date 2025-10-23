FACE_MODEL = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
    "GhostFaceNet",
]

MATRICES = ["cosine", "euclidean_l2"]

SELECTED_MODEL_KEY = 2
SELECTED_MATRIX_KEY = 1
THRESHOLD = 1.04

UPLOAD_FOLDER = "static/uploads"
ALLOWED_FILE_EXTENSIONS = {"jpg", "jpeg", "png"}


"""      
"Facenet": {"cosine": 0.40, "euclidean": 10, "euclidean_l2": 0.80},
"Facenet512": {"cosine": 0.30, "euclidean": 23.56, "euclidean_l2": 1.04},
"ArcFace": {"cosine": 0.68, "euclidean": 4.15, "euclidean_l2": 1.13},
"Dlib": {"cosine": 0.07, "euclidean": 0.6, "euclidean_l2": 0.4},
"SFace": {"cosine": 0.593, "euclidean": 10.734, "euclidean_l2": 1.055},
"OpenFace": {"cosine": 0.10, "euclidean": 0.55, "euclidean_l2": 0.55},
"DeepFace": {"cosine": 0.23, "euclidean": 64, "euclidean_l2": 0.64},
"DeepID": {"cosine": 0.015, "euclidean": 45, "euclidean_l2": 0.17},
"GhostFaceNet": {"cosine": 0.65, "euclidean": 35.71, "euclidean_l2": 1.10}    
"""