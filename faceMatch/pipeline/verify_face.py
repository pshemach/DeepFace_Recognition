from deepface import DeepFace
from faceMatch.constant import FACE_MODEL, SELECTED_MODEL_KEY


def verify_faces(image1, image2):
    try:
        result = DeepFace.verify(
            image1, image2, model_name=FACE_MODEL[SELECTED_MODEL_KEY]
        )
        confidence = result["verified"]
        distance = result["distance"]

        return confidence, distance
    except Exception as e:
        return str(e)
