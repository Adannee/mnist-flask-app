from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

app = Flask(__name__)
model = load_model("model/mnist_model.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28)) / 255.0
    image = image.reshape(1, 28, 28)

    prediction = model.predict(image)
    digit = np.argmax(prediction)

    return jsonify({'prediction': int(digit)})

if __name__ == '__main__':
    app.run(debug=True)
