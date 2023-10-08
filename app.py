from flask import Flask,jsonify
import tensorflow as tf
import numpy as np
app = Flask(__name__)

# Define the path to the model.h5 file
model_path = "model.h5"

# Load the pre-trained model
model= tf.keras.models.load_model(model_path)

image_path ='input.jpg'
img= tf.keras.preprocessing.image.load_img(image_path, target_size=(75,100))
img_array =tf.keras.preprocessing.image.img_to_array(img)
img_array =np.expand_dims(img_array, axis=0)
img_array =tf.keras.applications.mobilenet_v2.preprocess_input(img_array)


@app.route('/')
def prediction():
    predictions =model.predict(img_array)
    predictions_as_list = predictions.tolist()  # Convert the NumPy ndarray to a list
    response_data = {
        'predictions': predictions_as_list
    }

    # Return the JSON response
    return jsonify(response_data)

if __name__ == '__main__':
    app.run()
