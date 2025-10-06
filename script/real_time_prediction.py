import cv2
import torch
import numpy as np
from torchvision import transforms
from facenet_pytorch import MTCNN
import joblib
from torchvision import models
import torch.nn as nn
from PIL import Image
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..")) 

DATASET_DIR = os.path.join(ROOT_DIR, "DATASET")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")

if not os.path.exists(ARTIFACTS_DIR):
    raise FileNotFoundError(f"Le dossier '{ARTIFACTS_DIR}' n'existe pas. Assurez-vous qu'il est présent au chemin spécifié.")

class_names = {
    0: "Surprise",
    1: "Fear",
    2: "Disgust",
    3: "Happiness",
    4: "Sadness",
    5: "Anger",
    6: "Neutral"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de : {device}")

model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s')

mtcnn = MTCNN(keep_all=False, device=device)

efficientnet_model = models.efficientnet_b0(pretrained=True)
efficientnet_model.classifier[1] = nn.Linear(efficientnet_model.classifier[1].in_features, 7)  # 7 classes

efficientnet_path = os.path.join(ARTIFACTS_DIR, "best_efficientnetNormal_model.pth")
if not os.path.exists(efficientnet_path):
    raise FileNotFoundError(f"Le fichier '{efficientnet_path}' est introuvable dans le dossier 'artifacts'.")
efficientnet_model.load_state_dict(torch.load(efficientnet_path, map_location=torch.device('cpu')))
efficientnet_model.eval()
efficientnet_model = efficientnet_model.to(device)

xgb_model_path = os.path.join(ARTIFACTS_DIR, "xgb_efficientnet_modelNormal_best.pkl")
if not os.path.exists(xgb_model_path):
    raise FileNotFoundError(f"Le fichier '{xgb_model_path}' est introuvable dans le dossier 'artifacts'.")
best_xgb_model = joblib.load(xgb_model_path)

scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Le fichier '{scaler_path}' est introuvable dans le dossier 'artifacts'.")
scaler = joblib.load(scaler_path)

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def detect_faces_and_predict(frame):
    """Détecte les visages dans une frame, affine avec MTCNN, et fait des prédictions."""
    results = model_yolo(frame)  
    boxes = results.xyxy[0].cpu().numpy()  

    predictions = []
    for box in boxes:
        x_min, y_min, x_max, y_max, confidence, class_id = map(int, box[:6])
        if class_id == 0:  
            cropped_face = frame[y_min:y_max, x_min:x_max]

            face_tensor = mtcnn(Image.fromarray(cropped_face))
            if face_tensor is not None:
                face_pil = transforms.ToPILImage()(face_tensor)
                face_transformed = transform_test(face_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    features = efficientnet_model.features(face_transformed)
                    features = efficientnet_model.avgpool(features)
                    features = torch.flatten(features, 1).cpu().numpy()

                features_scaled = scaler.transform(features)
                prediction = best_xgb_model.predict(features_scaled)
                prediction_proba = best_xgb_model.predict_proba(features_scaled)

                predictions.append((prediction[0], prediction_proba, (x_min, y_min, x_max, y_max)))

    return predictions

cap = cv2.VideoCapture(0) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = detect_faces_and_predict(frame_rgb)

    for predicted_label, probabilities, (x_min, y_min, x_max, y_max) in results:
        class_label = class_names[predicted_label]
        confidence = max(probabilities[0]) * 100

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        text = f"{class_label} ({confidence:.1f}%)"
        cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Real-Time Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
