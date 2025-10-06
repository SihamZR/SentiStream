import streamlit as st
import requests
from PIL import Image
import cv2
import numpy as np
import io

st.set_page_config(
    page_title="Détecteur d'Émotions",
    page_icon="😊",
    layout="wide"
)

API_URL = "http://serving-api-container:8080"

st.markdown("""
    <style>
        .main-title {
            text-align: center;
            padding: 1rem;
            color: #1E88E5;
        }
        .success-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #E8F5E9;
            border-left: 5px solid #4CAF50;
        }
        .stButton>button {
            background-color: #1E88E5;
            color: white;
            border-radius: 20px;
            padding: 0.5rem 2rem;
        }
        .emotion-info {
            background-color: #F3F4F6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🎭 Détecteur d'Émotions IA</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ℹ️ À propos")
    st.info("""
    Cette application utilise l'intelligence artificielle pour détecter les émotions sur les visages.
    
    **Technologies utilisées:**
    - FastAPI
    - EfficientNet
    - YOLOv5
    - MTCNN
    - XGBoost
    """)
    
    st.markdown("### 🖍 Guide des émotions")
    emotions_guide = {
        "😊 Happiness": "Joie, sourire",
        "😲 Surprise": "Étonnement",
        "😐 Neutral": "Expression neutre",
        "😨 Fear": "Peur, appréhension",
        "😢 Sadness": "Tristesse",
        "😡 Anger": "Colère"
    }
    for emotion, description in emotions_guide.items():
        st.markdown(f"**{emotion}**: {description}")

main_container = st.container()

with main_container:
    tab1, tab2 = st.tabs(["📸 Mode Photo", "🎥 Mode Vidéo"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 💄 Téléchargement d'image")
            uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])
            
        if uploaded_file is not None:
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Image téléchargée")

            img_bytes = io.BytesIO()
            image.save(img_bytes, format="JPEG")
            img_bytes = img_bytes.getvalue()

            with col2:
                st.markdown("### 🔍 Résultats de l'analyse")
                with st.spinner("🤖 Analyse en cours..."):
                    response = requests.post(
                        f"{API_URL}/predict",
                        files={"file": ("image.jpg", img_bytes, "image/jpeg")}
                    )

                if response.status_code == 200:
                    results = response.json()
                    predictions = results.get("predictions", [])
                    
                    if predictions:
                        for i, pred in enumerate(predictions, 1):
                            with st.container():
                                st.markdown(f"<div class='emotion-info'>", unsafe_allow_html=True)
                                st.markdown(f"**Visage #{i}**")
                                st.markdown(f"🎭 **Émotion**: {pred['label']}")
                                st.progress(pred['confidence'])
                                st.markdown(f"Confiance: {pred['confidence'] * 100:.1f}%")
                                st.markdown("</div>", unsafe_allow_html=True)

                        st.markdown("### ✍️ Feedback")
                        true_label = st.selectbox(
                            "Quelle est la véritable émotion ?",
                            ["", "😊 Happiness", "😲 Surprise", "😐 Neutral", 
                             "😨 Fear", "😢 Sadness", "😡 Anger"]
                        )

                        if st.button("📤 Envoyer le feedback"):
                            if true_label:
                                with st.spinner("Envoi du feedback..."):
                                    feedback_response = requests.post(
                                         f"{API_URL}/feedback",
                                        files={"file": ("image.jpg", img_bytes, "image/jpeg")},
                                        data={
                                            "prediction": predictions[0]["label"],
                                            "target": true_label.split()[1] if len(true_label.split()) > 1 else true_label
                                        }
                                    )

                                if feedback_response.status_code == 200:
                                    st.markdown(
                                        "<div class='success-box'>✅ Feedback envoyé avec succès!</div>",
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.error("❌ Erreur lors de l'envoi du feedback")
                    else:
                        st.warning("😕 Aucun visage détecté dans l'image")
                else:
                    st.error("❌ Erreur lors de l'analyse de l'image")

    with tab2:
        st.markdown("### 🎥 Mode Vidéo en Direct")
        if st.checkbox("📹 Activer la caméra"):
            cap = cv2.VideoCapture(0)
            frame_placeholder = st.empty()
            stop_button = st.button("⏹️ Arrêter la caméra")
            
            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.error("Erreur d'accès à la caméra")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _, encoded_img = cv2.imencode(".jpg", frame_rgb)
                img_bytes = encoded_img.tobytes()

                response = requests.post(
                    f"{API_URL}/predict",  
                    files={"file": ("frame.jpg", img_bytes, "image/jpeg")}
                )

                if response.status_code == 200:
                    results = response.json()
                    for pred in results.get("predictions", []):
                        x_min, y_min, x_max, y_max = pred["box"]
                        label = f"{pred['label']} ({pred['confidence'] * 100:.1f}%)"
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (46, 139, 87), 2)
                        cv2.putText(frame, label, (x_min, y_min - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (46, 139, 87), 2)

                frame_placeholder.image(frame, channels="BGR")

            cap.release()