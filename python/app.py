import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="NuageLab Explorer", page_icon="☁️", layout="centered")

# Style CSS pour une interface moderne
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stAlert { border-radius: 15px; }
    .prediction-box {
        padding: 20px;
        background-color: white;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CHARGEMENT DES RESSOURCES ---
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model('modele_nuages.h5')
    with open('classes.json', 'r') as f:
        # Charger les classes (clés en string dans JSON, on les remet en int)
        labels = {int(k): v for k, v in json.load(f).items()}
    return model, labels

# --- DESCRIPTIONS MÉTIER ---
DESCRIPTIONS = {
    "cirrus": "Fins et fibreux. Signale souvent un changement de temps.",
    "cumulonimbus": "Nuage d'orage massif en forme d'enclume. Risque de grêle.",
    "cumulus": "Aspect de coton blanc. Beau temps si petit développement.",
    "stratus": "Nappe grise uniforme, peut provoquer de la bruine.",
    "altocumulus": "Galets ou rouleaux blancs/gris à moyenne altitude.",
    # Ajoutez les autres ici...
}

# --- INTERFACE ---
st.title("☁️ NuageLab Explorer Pro")
st.write("Identifiez les nuages instantanément grâce à l'Intelligence Artificielle.")

uploaded_file = st.file_uploader("Téléchargez une photo du ciel", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Affichage de l'image
    image = Image.open(uploaded_file)
    st.image(image, caption="Votre spécimen", use_container_width=True)
    
    if st.button("🔍 ANALYSER LE CIEL"):
        try:
            model, labels = load_model_and_classes()
            
            # Prétraitement de l'image
            img = image.resize((224, 224))
            img_array = np.array(img).astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Prédiction
            with st.spinner('Analyse des motifs nuageux...'):
                preds = model.predict(img_array)
                idx = np.argmax(preds)
                label = labels[idx]
                confiance = preds[0][idx] * 100
            
            # Affichage des résultats
            st.markdown(f"""
                <div class="prediction-box">
                    <h2 style='color: #0288d1;'>{label.upper()}</h2>
                    <p>Indice de confiance : <b>{confiance:.1f}%</b></p>
                </div>
            """, unsafe_allow_html=True)
            
            st.progress(confiance / 100)
            
            # Infos complémentaires
            st.info(f"**Description :** {DESCRIPTIONS.get(label.lower(), 'Information non disponible pour ce type.')}")

        except Exception as e:
            st.error(f"Erreur lors de l'analyse : {e}")
            st.info("Vérifiez que 'modele_nuages.h5' et 'classes.json' sont bien dans le dossier.")

else:
    st.info("En attente d'une image pour commencer l'analyse.")