from deepface import DeepFace
import flask
import os
from werkzeug.utils import secure_filename
from faceMatch.pipeline.verify_face import verify_faces
from faceMatch.utils import make_dir
from faceMatch.constant import UPLOAD_FOLDER

app = flask.Flask(__name__)


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/compare_faces", methods=["POST"])
def compare_faces():
    try:
        # Check if both image files are present in the request
        if (
            "img1_path" not in flask.request.files
            or "img2_path" not in flask.request.files
        ):
            return flask.jsonify({"error": "img1_path and img2_path are required"}), 400

        img1 = flask.request.files["img1_path"]
        img2 = flask.request.files["img2_path"]

        # Define paths to save the images
        img1_path = os.path.join(
            app.config["UPLOAD_FOLDER"], secure_filename(img1.filename)
        )
        img2_path = os.path.join(
            app.config["UPLOAD_FOLDER"], secure_filename(img2.filename)
        )

        # Save the images
        try:
            img1.save(img1_path)
            img2.save(img2_path)
        except Exception as e:
            return flask.jsonify({"error": f"Error saving images: {str(e)}"}), 500

        # Verify the images using DeepFace
        try:
            result = verify_faces(image1=img1_path, image2=img2_path)
        except Exception as e:
            return (
                flask.jsonify(
                    {"error": f"Error processing images with DeepFace: {str(e)}"}
                ),
                500,
            )
        finally:
            # Clean up: Delete the uploaded files after processing
            try:
                os.remove(img1_path)
                os.remove(img2_path)
            except Exception as e:
                # Log error but do not fail the request due to cleanup issues
                print(f"Error deleting files: {str(e)}")

        return flask.jsonify(result)

    except Exception as e:
        return flask.jsonify({"error": f"Unexpected error: {str(e)}"}), 500


if __name__ == "__main__":
    make_dir(UPLOAD_FOLDER)
    app.run(host="0.0.0.0", port="5056", debug=True)
