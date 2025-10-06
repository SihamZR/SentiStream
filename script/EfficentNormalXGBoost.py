import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de : {device}")

# -----------------------------
# 1. Dataset personnalisé
# -----------------------------
class CustomDataset(Dataset):
    def __init__(self, labels_file, images_dir, transform=None):
        self.data = pd.read_csv(labels_file)
        self.images_dir = images_dir
        self.transform = transform
        self.data['label'] = self.data['label'] - 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        label = int(self.data.iloc[idx, 1])
        img_path = os.path.join(self.images_dir, str(label + 1), img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"L'image {img_path} est introuvable. Vérifiez votre dataset.")

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement de l'image {img_path}. Détails : {e}")

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                raise RuntimeError(f"Erreur lors des transformations sur l'image {img_path}. Détails : {e}")

        return image, label

# -----------------------------
# 2. Chargement des données
# -----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
train_csv = os.path.join(script_dir, "DATASET/train_labels.csv")
test_csv = os.path.join(script_dir, "DATASET/test_labels.csv")
train_images_dir = os.path.join(script_dir, "DATASET/train")
test_images_dir = os.path.join(script_dir, "DATASET/test")

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = CustomDataset(train_csv, train_images_dir, transform=transform_train)
test_dataset = CustomDataset(test_csv, test_images_dir, transform=transform_test)

indices = list(range(len(train_dataset)))
train_indices, val_indices = train_test_split(
    indices,
    test_size=0.2,
    stratify=[train_dataset.data.iloc[i, 1] for i in indices],
    random_state=42
)

train_subset = torch.utils.data.Subset(train_dataset, train_indices)
val_subset = torch.utils.data.Subset(train_dataset, val_indices)

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# -----------------------------
# Sauvegarde des images et labels dans des fichiers CSV
# -----------------------------
def save_images_and_labels(dataset, output_file):
    """Sauvegarde les chemins des images et leurs labels dans un fichier CSV."""
    records = []
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        img_name = dataset.data.iloc[idx, 0]
        records.append({"image": img_name, "label": label})

    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    print(f"Données sauvegardées dans {output_file}")

# Sauvegarder pour le dataset d'entraînement et de test
save_images_and_labels(train_dataset, os.path.join(script_dir, "DATASET/train_images_labels.csv"))
save_images_and_labels(test_dataset, os.path.join(script_dir, "DATASET/test_images_labels.csv"))

# -----------------------------
# 3. Fine-tuning EfficientNet
# -----------------------------
efficientnet_model = models.efficientnet_b0(pretrained=True)
num_classes = len(train_dataset.data['label'].unique())
efficientnet_model.classifier[1] = nn.Linear(efficientnet_model.classifier[1].in_features, num_classes)

for name, param in efficientnet_model.named_parameters():
    if "features.0" in name or "features.1" in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

efficientnet_model = efficientnet_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, efficientnet_model.parameters()), lr=1e-4)

def train_efficientnet(model, train_ld, val_ld, criterion, optimizer, device, epochs=40, patience=3):
    best_loss = float("inf")
    patience_count = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, lbls in train_ld:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_ld)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, lbls in val_ld:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                val_loss += loss.item()
        val_loss /= len(val_ld)
        val_losses.append(val_loss)

        print(f"[{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(), "artifacts/best_efficientnetNormal_model.pth")
            print("  -> Model improved. Saved.")
        else:
            patience_count += 1
            if patience_count >= patience:
                print("Early stopping triggered.")
                break

    # Graphique des pertes
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("train_val_loss_curve.png")
    plt.show()

print("==== Entraînement EfficientNet ====")
train_efficientnet(efficientnet_model, train_loader, val_loader, criterion, optimizer, device, epochs=40, patience=3)
efficientnet_model.load_state_dict(torch.load("best_efficientnetNormal_model.pth"))
efficientnet_model.eval()

# -----------------------------
# 4. Extraction de caractéristiques
# -----------------------------
def extract_features(model, loader, device):
    features_list = []
    labels_list = []
    model.eval()
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            x = model.features(imgs)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            features_list.append(x.cpu().numpy())
            labels_list.extend(lbls.numpy())
    return np.concatenate(features_list, axis=0), np.array(labels_list)

print("==== Extraction des caractéristiques (EfficientNet) ====")
X_train, y_train = extract_features(efficientnet_model, train_loader, device)
X_val, y_val = extract_features(efficientnet_model, val_loader, device)
X_test, y_test = extract_features(efficientnet_model, test_loader, device)

X_train_full = np.vstack((X_train, X_val))
y_train_full = np.hstack((y_train, y_val))

# -----------------------------
# 5. Prétraitement et XGBoost
# -----------------------------

adasyn = ADASYN(random_state=42)
X_train_res, y_train_res = adasyn.fit_resample(X_train_full, y_train_full)

# Génération et normalisation avec StandardScaler
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# Sauvegarde du scaler
joblib.dump(scaler, "scaler.pkl")
print("Scaler sauvegardé sous scaler.pkl")

# Paramètres pour XGBoost
params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(xgb_model, params, cv=3, scoring='accuracy', verbose=1)
grid_search.fit(X_train_res, y_train_res)

best_xgb_model = grid_search.best_estimator_

# Sauvegarder le modèle entraîné
joblib.dump(best_xgb_model, "xgb_efficientnet_modelNormal_best.pkl")
print("Modèle XGBoost sauvegardé sous xgb_efficientnet_modelNormal_best.pkl")

# -----------------------------
# Charger le scaler pour prédictions
# -----------------------------
scaler = joblib.load("scaler.pkl")  # Charger le scaler sauvegardé

# -----------------------------
# 6. Évaluation du modèle XGBoost
# -----------------------------
y_pred = best_xgb_model.predict(X_test_scaled)

print("=== Résultats XGBoost ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred, average='weighted'):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# 7. Sauvegarde du modèle XGBoost
# -----------------------------
joblib.dump(best_xgb_model, "xgb_efficientnet_modelNormal_best.pkl")
print("Modèle XGBoost sauvegardé sous xgb_efficientnet_modelNormal_best.pkl")





# import os
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
# from imblearn.over_sampling import ADASYN
# from sklearn.preprocessing import StandardScaler
# import xgboost as xgb
# import joblib
# from PIL import Image

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Utilisation de : {device}")

# # -----------------------------
# # 1. Dataset personnalisé
# # -----------------------------
# class CustomDataset(Dataset):
#     def __init__(self, labels_file, images_dir, transform=None):
#         self.data = pd.read_csv(labels_file)
#         self.images_dir = images_dir
#         self.transform = transform
#         self.data['label'] = self.data['label'] - 1  # Convertir les étiquettes 1-7 en 0-6

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         img_name = self.data.iloc[idx, 0]
#         label = int(self.data.iloc[idx, 1])
#         img_path = os.path.join(self.images_dir, str(label + 1), img_name)

#         if not os.path.exists(img_path):
#             raise FileNotFoundError(f"L'image {img_path} est introuvable. Vérifiez votre dataset.")

#         try:
#             image = Image.open(img_path).convert("RGB")
#         except Exception as e:
#             raise RuntimeError(f"Erreur lors du chargement de l'image {img_path}. Détails : {e}")

#         if self.transform:
#             try:
#                 image = self.transform(image)
#             except Exception as e:
#                 raise RuntimeError(f"Erreur lors des transformations sur l'image {img_path}. Détails : {e}")

#         return image, label

# # -----------------------------
# # 2. Chargement des données
# # -----------------------------
# script_dir = os.path.dirname(os.path.abspath(__file__))
# train_csv = os.path.join(script_dir, "train_labels.csv")
# test_csv = os.path.join(script_dir, "test_labels.csv")
# train_images_dir = os.path.join(script_dir, "DATASET/train")
# test_images_dir = os.path.join(script_dir, "DATASET/test")

# transform_train = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomRotation(degrees=30),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     transforms.ToTensor(),
#     transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# transform_test = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# train_dataset = CustomDataset(train_csv, train_images_dir, transform=transform_train)
# test_dataset = CustomDataset(test_csv, test_images_dir, transform=transform_test)

# indices = list(range(len(train_dataset)))
# train_indices, val_indices = train_test_split(
#     indices,
#     test_size=0.2,
#     stratify=[train_dataset.data.iloc[i, 1] for i in indices],
#     random_state=42
# )

# train_subset = torch.utils.data.Subset(train_dataset, train_indices)
# val_subset = torch.utils.data.Subset(train_dataset, val_indices)

# train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# # -----------------------------
# # 3. Fine-tuning EfficientNet
# # -----------------------------
# efficientnet_model = models.efficientnet_b0(pretrained=True)
# num_classes = len(train_dataset.data['label'].unique())
# efficientnet_model.classifier[1] = nn.Linear(efficientnet_model.classifier[1].in_features, num_classes)

# for name, param in efficientnet_model.named_parameters():
#     if "features.0" in name or "features.1" in name:
#         param.requires_grad = False
#     else:
#         param.requires_grad = True

# efficientnet_model = efficientnet_model.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, efficientnet_model.parameters()), lr=1e-4)

# def train_efficientnet(model, train_ld, val_ld, criterion, optimizer, device, epochs=40, patience=3):
#     best_loss = float("inf")
#     patience_count = 0

#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0
#         for imgs, lbls in train_ld:
#             imgs, lbls = imgs.to(device), lbls.to(device)
#             optimizer.zero_grad()
#             outputs = model(imgs)
#             loss = criterion(outputs, lbls)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#         train_loss = running_loss / len(train_ld)

#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for imgs, lbls in val_ld:
#                 imgs, lbls = imgs.to(device), lbls.to(device)
#                 outputs = model(imgs)
#                 loss = criterion(outputs, lbls)
#                 val_loss += loss.item()
#         val_loss /= len(val_ld)

#         print(f"[{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

#         if val_loss < best_loss:
#             best_loss = val_loss
#             patience_count = 0
#             torch.save(model.state_dict(), "best_efficientnetNormal_model.pth")
#             print("  -> Model improved. Saved.")
#         else:
#             patience_count += 1
#             if patience_count >= patience:
#                 print("Early stopping triggered.")
#                 break

# print("==== Entraînement EfficientNet ====")
# train_efficientnet(efficientnet_model, train_loader, val_loader, criterion, optimizer, device, epochs=40, patience=3)
# efficientnet_model.load_state_dict(torch.load("best_efficientnetNormal_model.pth"))
# efficientnet_model.eval()

# # -----------------------------
# # 4. Extraction de features
# # -----------------------------
# def extract_features(model, loader, device):
#     features_list = []
#     labels_list = []
#     model.eval()
#     with torch.no_grad():
#         for imgs, lbls in loader:
#             imgs = imgs.to(device)
#             x = model.features(imgs)
#             x = model.avgpool(x)
#             x = torch.flatten(x, 1)
#             features_list.append(x.cpu().numpy())
#             labels_list.extend(lbls.numpy())
#     return np.concatenate(features_list, axis=0), np.array(labels_list)

# print("==== Extraction features (EfficientNet) ====")
# X_train, y_train = extract_features(efficientnet_model, train_loader, device)
# X_val, y_val = extract_features(efficientnet_model, val_loader, device)
# X_test, y_test = extract_features(efficientnet_model, test_loader, device)

# X_train_full = np.vstack((X_train, X_val))
# y_train_full = np.hstack((y_train, y_val))

# # -----------------------------
# # 5. Prétraitement et XGBoost
# # -----------------------------
# adasyn = ADASYN(random_state=42)
# X_train_res, y_train_res = adasyn.fit_resample(X_train_full, y_train_full)

# scaler = StandardScaler()
# X_train_res = scaler.fit_transform(X_train_res)
# X_test_scaled = scaler.transform(X_test)

# params = {
#     'n_estimators': [100, 200],
#     'max_depth': [3, 5],
#     'learning_rate': [0.01, 0.1],
#     'subsample': [0.8, 1.0],
#     'colsample_bytree': [0.8, 1.0]
# }

# xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
# grid_search = GridSearchCV(xgb_model, params, cv=3, scoring='accuracy', verbose=1)
# grid_search.fit(X_train_res, y_train_res)

# best_xgb_model = grid_search.best_estimator_

# # -----------------------------
# # 6. Évaluer XGBoost
# # -----------------------------
# y_pred = best_xgb_model.predict(X_test_scaled)

# print("=== Résultats XGBoost ===")
# print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
# print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
# print(f"Recall:    {recall_score(y_test, y_pred, average='weighted'):.4f}")
# print(f"F1-score:  {f1_score(y_test, y_pred, average='weighted'):.4f}")
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# # -----------------------------
# # 7. Sauvegarder le modèle XGBoost
# # -----------------------------
# joblib.dump(best_xgb_model, "xgb_efficientnet_modelNormal_best.pkl")
# print("Modèle XGBoost sauvegardé sous xgb_efficientnet_modelNormal_best.pkl")
