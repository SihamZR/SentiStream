# SentiStream — Analyse & Détection des Sentiments en Temps Réel (ML)

**SentiStream** met en place un pipeline **temps réel** pour la classification de sentiments :
ingestion de flux, prétraitement NLP, entraînement de modèles (classiques et Transformers),
inférence en streaming et **visualisation dynamique** des résultats.

---

## Objectif du projet

Développer un pipeline temps réel capable de :
- Collecter et traiter des données de flux (streaming),
- Entraîner automatiquement des modèles de Machine Learning et Deep Learning,
- Déployer un service d’inférence en ligne via une API,
- Visualiser dynamiquement les prédictions des sentiments.

---

## Architecture du projet

```

Projet_Mlops/
│
├── script/               # Entraînement et extraction des caractéristiques
├── serving/              # API FastAPI pour les prédictions en ligne
├── webapp/               # Interface Streamlit pour la visualisation temps réel
├── artifacts/            # Modèles sauvegardés (XGBoost, EfficientNet, etc.)
├── DATASET/              # Données d'entraînement et de test
├── reporting/            # Résultats, graphiques et rapports
│
├── requirements.txt      # Dépendances
├── README.md             # Documentation du projet
├── .gitignore
└── Architecture_et_Composants_du_Modele.pdf

````

---

## Technologies utilisées

| Domaine | Outils et bibliothèques |
|----------|-------------------------|
| **Machine Learning** | XGBoost, LightGBM, Random Forest, ADASYN |
| **Deep Learning** | EfficientNet (PyTorch) |
| **Backend API** | FastAPI |
| **Déploiement** | Docker |
| **Interface Web** | Streamlit |
| **Suivi & Analyse** | Matplotlib, Pandas |
| **MLOps & Streaming** | Architecture temps réel, pipeline automatisé |

---

## Points techniques clés

- Extraction des caractéristiques d’images avec **EfficientNet** fine-tunée.  
- Classification supervisée via **XGBoost**, optimisée par **GridSearchCV**.  
- Rééquilibrage des classes avec **ADASYN**.  
- Service REST pour l’inférence en temps réel via **FastAPI**.  
- Interface utilisateur dynamique sous **Streamlit**.  
- Conteneurisation complète avec **Docker** pour le déploiement reproductible.

---

## Installation et exécution

### 1️⃣ Cloner le dépôt
```bash
git clone https://github.com/SihamZR/SentiStream.git
cd SentiStream
````

### 2️⃣ Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3️⃣ Lancer l’API FastAPI

```bash
cd serving
uvicorn main:app --reload
```

### 4️⃣ Lancer l’interface Streamlit

```bash
cd webapp
streamlit run app.py
```

---

## Documentation technique

Pour une présentation complète de l’architecture, du fine-tuning **EfficientNet** et de l’intégration **XGBoost**, consultez le document :
[Architecture et Composants du Modèle](./Architecture_et_Composants_du_Modele.pdf)

---

## Auteur

**Siham Zarmoum**
Master 2 Data Science – Centrale Lyon
[sihamzarmoum@gmail.com](mailto:sihamzarmoum@gmail.com)
[LinkedIn](https://linkedin.com/in/siham-zarmoum/)
[GitHub](https://github.com/SihamZR)

---

## Mots-clés

`machine-learning` `deep-learning` `fastapi` `docker` `streamlit` `sentiment-analysis` `real-time` `mlops`

````
