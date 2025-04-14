import torch
import numpy as np
import cv2

# Load your custom YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp23/weights/last.pt', force_reload=True)

# Set the manual confidence threshold
CONFIDENCE_THRESHOLD = 0.001  # 👈 as low as possible

def predict_drowsiness(img):
    try:
        # Resize image to 320x320 (model input size)
        img_resized = cv2.resize(img, (320, 320))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Run inference
        results = model(img_rgb)

        names = results.names
        predictions = results.pred[0]

        if predictions is not None and len(predictions):
            for *box, conf, cls in predictions:
                if conf >= CONFIDENCE_THRESHOLD:
                    class_name = names[int(cls)]
                    print(f"Detected: {class_name} with confidence {conf:.3f}")
                    if class_name.lower() == "drowsy":
                        return 0  # Drowsy
        return 1  # Awake

    except Exception as e:
        print(f"Error during prediction: {e}")
        return -1
