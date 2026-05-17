from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json
import io
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'best_model_v2.keras')
LABELS_PATH = os.path.join(BASE_DIR, 'class_labels_v2.json')
TREATMENTS_PATH = os.path.join(BASE_DIR, 'treatments.json')

# Download model from Google Drive if not present
if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model from Google Drive...")
    os.makedirs(os.path.join(BASE_DIR, 'model'), exist_ok=True)
    import gdown
    FILE_ID = '1ZiLOnLLzEdu-Yk7bE2gBP8tKyqtv8hYi'
    gdown.download(f'https://drive.google.com/uc?id={FILE_ID}', MODEL_PATH, quiet=False)
    print("✅ Model downloaded")

print("⏳ Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully")

with open(LABELS_PATH) as f:
    LABELS = json.load(f)

with open(TREATMENTS_PATH) as f:
    TREATMENTS = json.load(f)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running', 'model_loaded': True})


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((224, 224))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        preds = model.predict(arr)[0]
        top_idx = int(np.argmax(preds))
        confidence = round(float(preds[top_idx]) * 100, 1)
        label_key = LABELS.get(str(top_idx), 'unknown')
        treatment = TREATMENTS.get(label_key, {
            'disease_name': label_key.replace('_', ' '),
            'crop': label_key.split('___')[0] if '___' in label_key else 'Unknown',
            'description': 'Disease identified. Please consult an agricultural officer.',
            'severity': 'moderate',
            'symptoms': [],
            'chemical_treatment': [],
            'organic_treatment': [],
            'preventive_measures': []
        })

        if confidence < 70.0:
            return jsonify({
                'error': 'Could not identify a crop disease in this image. '
                         'Please upload a clear, close-up photo of a crop leaf '
                         'with good lighting. Supported crops: Tomato, Maize and Cassava.'
            }), 200

        return jsonify({
            'confidence': confidence,
            'label_key': label_key,
            **treatment
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)