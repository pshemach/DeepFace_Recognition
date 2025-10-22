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

MATRICES = ["cosine", "euclidean", "euclidean_l2"]

SELECTED_MODEL_KEY = 2

UPLOAD_FOLDER = "static/uploads"
REFERENCE_FOLDER = "data/reference_images"
DATABASE_PATH = "data/face_embeddings.db"

ALLOWED_FILE_EXTENSIONS = {"jpg", "jpeg", "png"}