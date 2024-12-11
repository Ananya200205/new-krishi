from flask import Flask, request, jsonify
import joblib
import numpy as np
from PIL import Image
import os
from flask_cors import CORS

# Load the trained model and label encoder
model = joblib.load('random_forest_model.pkl')  # Load the trained model
label_encoder = joblib.load('label_encoder.pkl')  # Load the saved label encoder

# Configure upload folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
CORS(app)

# Preprocess image and make predictions
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((128, 128))  # Resize to match the training size
    image_array = np.array(image).flatten()  # Convert to 1D array
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save the uploaded file temporarily
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Preprocess the image and make a prediction
        image_data = preprocess_image(file_path)
        prediction = model.predict([image_data])

        # Use label_encoder to convert the numeric prediction back to the disease name
        predicted_label = label_encoder.inverse_transform(prediction)[0]  # Decode the label
        
        # Return the prediction as a response
        return jsonify({"predicted_disease": predicted_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
