from deepface import DeepFace
import flask
import os
from werkzeug.utils import secure_filename

app = flask.Flask(__name__)

models = [
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

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/compare_faces', methods=['POST'])
def compare_faces():
    try:
        if 'img1_path' not in flask.request.files or 'img2_path' not in flask.request.files:
            return flask.jsonify({'error': 'img1_path and img2_path are required'}), 400

        img1 = flask.request.files['img1_path']
        img2 = flask.request.files['img2_path']
        model_name = flask.request.form.get('model_name', models[0])

        if model_name not in models:
            return flask.jsonify({'error': f'Invalid model_name. Available models: {models}'}), 400

        img1_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(img1.filename))
        img2_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(img2.filename))

        img1.save(img1_path)
        img2.save(img2_path)

        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=model_name
        )

        # Optionally delete the uploaded files after processing
        os.remove(img1_path)
        os.remove(img2_path)

        return flask.jsonify(result)
    except Exception as e:
        return flask.jsonify({'error': str(e)}), 500
