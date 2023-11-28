from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__, template_folder='templates')
CORS(app)
# Define the path to the model.h5 file
model_path = "model.h5"

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Define a temporary directory to store uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def preprocess_image(file_path):
    img = tf.keras.preprocessing.image.load_img(
        file_path, target_size=(75, 100))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'imageFile' not in request.files:
        return 'No file part'

    file = request.files['imageFile']

    if file.filename == '':
        return 'No selected file'

    # Save the uploaded file to the temporary directory
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Preprocess the image
    img_array = preprocess_image(file_path)

    # Make predictions
    predictions = model.predict(img_array)
    predictions_as_list = predictions.tolist()

    # Delete the temporary file
    os.remove(file_path)

    response_data = {
        'predictions': predictions_as_list
    }

    return jsonify(response_data)


if __name__ == '__main__':
    app.run()
