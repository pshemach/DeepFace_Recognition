import flask
import os
from werkzeug.utils import secure_filename
from faceMatch.pipeline.verify_face import verify_faces
from faceMatch.utils import make_dir, save_reference_image, get_reference_image_path, list_reference_images
from faceMatch.constant import UPLOAD_FOLDER, REFERENCE_FOLDER

app = flask.Flask(__name__)


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["REFERENCE_FOLDER"] = REFERENCE_FOLDER


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
            # Print paths for debugging
            print(f"Direct comparison - Image 1 path: {img1_path}")
            print(f"Direct comparison - Image 2 path: {img2_path}")

            # Verify the images
            result = verify_faces(image1=img1_path, image2=img2_path)
            print(f"Direct comparison result: {result}")

            # Check if result contains an error
            if isinstance(result, dict) and 'error' in result:
                error_message = result['error']
                print(f"Error detected in direct comparison result: {error_message}")
                return flask.jsonify({"error": error_message}), 500

            # Verify that the result contains the expected fields
            required_fields = ['verified', 'distance', 'threshold']
            for field in required_fields:
                if field not in result:
                    error_message = f"Missing required field in direct comparison result: {field}"
                    print(error_message)
                    return flask.jsonify({"error": error_message}), 500
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            print(f"Exception in direct comparison: {str(e)}")
            print(f"Traceback: {error_traceback}")
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

        return flask.jsonify({
            "verified": result['verified'],
            "distance": round(result['distance'],2),
            "threshold": round(result['threshold'],2)
        })

    except Exception as e:
        return flask.jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@app.route("/upload_reference", methods=["POST"])
def upload_reference():
    try:
        # Check if image file and key are present in the request
        if "image" not in flask.request.files:
            return flask.jsonify({"error": "Image file is required"}), 400

        if "key" not in flask.request.form:
            return flask.jsonify({"error": "Reference key is required"}), 400

        image = flask.request.files["image"]
        key = flask.request.form["key"]

        # Validate key format (alphanumeric only)
        if not key.isalnum():
            return flask.jsonify({"error": "Key must contain only alphanumeric characters"}), 400

        # Save the reference image
        try:
            reference_path = save_reference_image(image, key)
            return flask.jsonify({
                "success": True,
                "message": f"Reference image saved with key: {key}",
                "key": key,
                "path": reference_path
            })
        except Exception as e:
            return flask.jsonify({"error": f"Error saving reference image: {str(e)}"}), 500

    except Exception as e:
        return flask.jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@app.route("/compare_with_reference", methods=["POST"])
def compare_with_reference():
    try:
        # Check if image file and reference key are present in the request
        if "image" not in flask.request.files and "img2_path" not in flask.request.files:
            return flask.jsonify({"error": "Image file is required (as 'image' or 'img2_path')"}), 400

        if "reference_key" not in flask.request.form:
            return flask.jsonify({"error": "Reference key is required"}), 400

        # Get the image file (either from 'image' or 'img2_path' parameter)
        if "image" in flask.request.files:
            image = flask.request.files["image"]
        else:
            image = flask.request.files["img2_path"]

        reference_key = flask.request.form["reference_key"]
        print(f"Received request with reference_key: {reference_key} and image: {image.filename}")

        # Get the reference image path
        reference_path = get_reference_image_path(reference_key)
        if not reference_path:
            return flask.jsonify({"error": f"No reference image found with key: {reference_key}"}), 404

        # Save the uploaded image temporarily
        img_path = os.path.join(
            app.config["UPLOAD_FOLDER"], secure_filename(image.filename)
        )

        try:
            image.save(img_path)
        except Exception as e:
            return flask.jsonify({"error": f"Error saving image: {str(e)}"}), 500

        # Verify the images using DeepFace
        try:
            # Print paths for debugging
            print(f"Reference path: {reference_path}")
            print(f"Image path: {img_path}")

            # Verify the images
            result = verify_faces(image1=reference_path, image2=img_path)
            print(f"Result: {result}")

            # Check if result contains an error
            if isinstance(result, dict) and 'error' in result:
                error_message = result['error']
                print(f"Error detected in result: {error_message}")
                return flask.jsonify({"error": error_message}), 500

            # Verify that the result contains the expected fields
            required_fields = ['verified', 'distance', 'threshold']
            for field in required_fields:
                if field not in result:
                    error_message = f"Missing required field in result: {field}"
                    print(error_message)
                    return flask.jsonify({"error": error_message}), 500

        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            print(f"Exception in compare_with_reference: {str(e)}")
            print(f"Traceback: {error_traceback}")
            return (
                flask.jsonify(
                    {"error": f"Error processing images with DeepFace: {str(e)}"}
                ),
                500,
            )
        finally:
            # Clean up: Delete the uploaded file after processing
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
                    print(f"Successfully deleted temporary file: {img_path}")
            except Exception as e:
                # Log error but do not fail the request due to cleanup issues
                print(f"Error deleting file: {str(e)}")

        return flask.jsonify({
            "reference_key": reference_key,
            "result": {
                "verified": result['verified'],
                "distance": round(result['distance'],2),
                "threshold": round(result['threshold'],2)
            }
        })

    except Exception as e:
        return flask.jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@app.route("/list_references", methods=["GET"])
def list_references():
    try:
        references = list_reference_images()
        return flask.jsonify({
            "count": len(references),
            "references": [{
                "key": ref["key"],
                "filename": os.path.basename(ref["path"])
            } for ref in references]
        })
    except Exception as e:
        return flask.jsonify({"error": f"Unexpected error: {str(e)}"}), 500


if __name__ == "__main__":
    # Create necessary directories
    make_dir(UPLOAD_FOLDER)

    # For reference folder, we don't want to clear it if it exists
    # Just make sure it exists
    os.makedirs(REFERENCE_FOLDER, exist_ok=True)

    app.run(host="0.0.0.0", port="5056", debug=True)
