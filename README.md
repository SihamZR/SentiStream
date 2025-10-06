# SentiStream — Analyse & Détection des Sentiments en Temps Réel (ML)

**SentiStream** met en place un pipeline **temps réel** pour la classification de sentiments :
ingestion de flux, prétraitement NLP, entraînement de modèles (classiques et Transformers),
inférence en streaming et **visualisation dynamique** des résultats.

---

## Objectifs
- Ingestion continue de textes (API, Kafka ou WebSocket).
- Prétraitement & vectorisation (TF-IDF / embeddings Transformers).
- Entraînement de modèles (LogReg/SVM/XGBoost) et/ou **Transformers**.
- Service d’inférence **FastAPI**.
- **Dashboard Streamlit** pour la visualisation en temps réel.

---

## Stack
- **Python**, **scikit-learn**, **NLTK/TextBlob**
- **Transformers** (optionnel) — `bert-base-multilingual-cased` ou équivalent
- **FastAPI + Uvicorn** (API), **WebSockets** (stream)
- **Streamlit + Plotly** (dashboard)
- **Kafka** (optionnel) pour le transport des messages

---

## Documentation Technique

Pour une description détaillée de l’architecture du modèle, des étapes de fine-tuning,
et des composants utilisés (EfficientNet, XGBoost, ADASYN, GridSearchCV), consultez le document ci-dessous :

[Architecture et Composants du Modèle](./Architecture_et_Composants_du_Modele.pdf)
