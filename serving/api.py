import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
from facenet_pytorch import MTCNN
from torchvision import transforms, models
import joblib
import torch
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..")) 

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
REPORTING_DIR = os.path.join(BASE_DIR, "reporting")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_efficientnetNormal_model.pth")
XGB_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "xgb_efficientnet_modelNormal_best.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
PROD_DATA_PATH = os.path.join(REPORTING_DIR, "prod_data.csv")
MODEL_YOLO_PATH = os.path.join(ARTIFACTS_DIR, "yolov5s.pt")

REQUIRED_FILES = [
    (MODEL_PATH, "modèle EfficientNet"),
    (XGB_MODEL_PATH, "modèle XGBoost"),
    (SCALER_PATH, "scaler"),
    (MODEL_YOLO_PATH, "modèle YOLOv5"),
]
for path, description in REQUIRED_FILES:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Le fichier {description} est introuvable : {path}")

model_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_YOLO_PATH, force_reload=True, device='cpu')

mtcnn = MTCNN(keep_all=False, device="cpu")

efficientnet_model = models.efficientnet_b0(pretrained=True)
efficientnet_model.classifier[1] = torch.nn.Linear(efficientnet_model.classifier[1].in_features, 7)
efficientnet_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
efficientnet_model.eval()

best_xgb_model = joblib.load(XGB_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

class_names = {
    0: "Surprise",
    1: "Fear",
    2: "Disgust",
    3: "Happiness",
    4: "Sadness",
    5: "Anger",
    6: "Neutral",
}

transform_test = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        image_np = np.array(image)

        results = model_yolo(image_np)
        boxes = results.xyxy[0].cpu().numpy()

        predictions = []
        for box in boxes:
            x_min, y_min, x_max, y_max, confidence, class_id = map(int, box[:6])
            if class_id == 0:  
                cropped_face = image_np[y_min:y_max, x_min:x_max]
                face_tensor = mtcnn(Image.fromarray(cropped_face))

                if face_tensor is not None:
                    face_pil = transforms.ToPILImage()(face_tensor)
                    face_transformed = transform_test(face_pil).unsqueeze(0)

                    with torch.no_grad():
                        features = efficientnet_model.features(face_transformed)
                        features = efficientnet_model.avgpool(features)
                        features = torch.flatten(features, 1).numpy()

                    features_scaled = scaler.transform(features)
                    prediction = best_xgb_model.predict(features_scaled)
                    prediction_proba = best_xgb_model.predict_proba(features_scaled)

                    predictions.append(
                        {
                            "label": class_names[prediction[0]],
                            "confidence": float(max(prediction_proba[0])),
                            "box": [x_min, y_min, x_max, y_max],
                        }
                    )

        return {"predictions": predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")

@app.post("/feedback")
async def feedback(file: UploadFile = File(...), prediction: str = Form(...), target: str = Form(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        image_np = np.array(image)

        results = model_yolo(image_np)
        boxes = results.xyxy[0].cpu().numpy()

        features = []
        if len(boxes) > 0:
            x_min, y_min, x_max, y_max, _, _ = map(int, boxes[0])
            cropped_face = image_np[y_min:y_max, x_min:x_max]
            face_tensor = mtcnn(Image.fromarray(cropped_face))

            if face_tensor is not None:
                face_pil = transforms.ToPILImage()(face_tensor)
                face_transformed = transform_test(face_pil).unsqueeze(0)

                with torch.no_grad():
                    extracted_features = efficientnet_model.features(face_transformed)
                    extracted_features = efficientnet_model.avgpool(extracted_features)
                    extracted_features = torch.flatten(extracted_features, 1).numpy()
                    features = extracted_features.tolist()
        else:
            raise HTTPException(status_code=400, detail="Aucun visage détecté pour le feedback.")

        if not features:
            raise HTTPException(status_code=500, detail="Impossible d'extraire les caractéristiques.")

        new_entry_df = pd.DataFrame(
            {
                "features": [features[0]],
                "prediction": [prediction],
                "target": [target],
            }
        )

        if os.path.exists(PROD_DATA_PATH):
            df = pd.read_csv(PROD_DATA_PATH)
        else:
            df = pd.DataFrame(columns=["features", "prediction", "target"])

        logging.info(f"Nouvelle entrée : prediction={prediction}, target={target}")

        df = pd.concat([df, new_entry_df], ignore_index=True)
        df.to_csv(PROD_DATA_PATH, index=False)

        return {"message": "Feedback enregistré avec succès."}

    except Exception as e:
        logging.error(f"Erreur lors de l'enregistrement du feedback : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur : {str(e)}")
