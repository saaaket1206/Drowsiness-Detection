from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import os
from predict import predict_drowsiness

app = Flask(__name__)
CORS(app)  # Allow CORS from all origins

# Define the directory to save received images
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'received_images')
os.makedirs(SAVE_DIR, exist_ok=True)

# 🔁 Global state to track latest prediction
latest_status = "0"  # Default to 0 (not drowsy)

@app.route('/predict', methods=['POST'])
def predict():
    global latest_status  # So we can update it

    print("🔥 Received a request to /predict")

    if 'image' not in request.files:
        print("❌ No image in request")
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    print("📷 Image file received")

    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        print("⚠️ Failed to decode image")
        return jsonify({'error': 'Invalid image'}), 400

    # Save image to local folder
    image_path = os.path.join(SAVE_DIR, 'received_image.jpg')
    cv2.imwrite(image_path, img)
    print(f"💾 Image saved at {image_path}")

    result = predict_drowsiness(img)
    print("✅ Prediction:", result)

    latest_status = "1" if result == 1 else "0"

    return jsonify({'drowsy': result})

# 🆕 Status endpoint for ESP32 to poll
@app.route('/status', methods=['GET'])
def status():
    return latest_status

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)