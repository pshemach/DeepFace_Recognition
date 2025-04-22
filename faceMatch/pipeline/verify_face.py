from deepface import DeepFace
from faceMatch.constant import FACE_MODEL, SELECTED_MODEL_KEY, MATRICES


def verify_faces(image1, image2):
    try:
        # Print detailed information about the input images
        print(f"Verifying faces with DeepFace:")
        print(f"Image 1 path: {image1}")
        print(f"Image 2 path: {image2}")
        print(f"Model: {FACE_MODEL[SELECTED_MODEL_KEY]}")
        print(f"Distance metric: {MATRICES[2]}")

        # Check if the image files exist
        import os
        if not os.path.exists(image1):
            return {"error": f"Reference image file not found: {image1}"}
        if not os.path.exists(image2):
            return {"error": f"Comparison image file not found: {image2}"}

        # Try to open the images to verify they are valid
        try:
            from PIL import Image
            img1 = Image.open(image1)
            img1.verify()  # Verify that it's a valid image
            print(f"Image 1 is valid: {img1.format}, size: {img1.size}")

            img2 = Image.open(image2)
            img2.verify()  # Verify that it's a valid image
            print(f"Image 2 is valid: {img2.format}, size: {img2.size}")
        except Exception as img_error:
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
        print(f"DeepFace verification result: {result}")

        # Return the full result dictionary
        return result
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error in verify_faces: {str(e)}")
        print(f"Traceback: {error_traceback}")
        return {"error": f"DeepFace error: {str(e)}"}
