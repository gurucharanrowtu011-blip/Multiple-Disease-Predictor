import streamlit as st
import numpy as np
import pandas as pd
import joblib

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="AI Virtual Doctor", layout="centered")
st.title("🩺 AI Virtual Doctor")

# =========================
# LOAD MODEL + ENCODER
# =========================
model = joblib.load("disease_predictor_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("training.csv")
df.columns = df.columns.str.strip()

features = df.columns[:-2]  # exclude prognosis + medicine

# =========================
# SYMPTOM CATEGORIES
# =========================
categories = {
    "General": ["fatigue", "lethargy", "malaise", "restlessness", "anxiety", "loss_of_appetite"],
    "Fever": ["high_fever", "mild_fever", "chills", "sweating"],
    "Respiratory": ["cough", "breathlessness", "chest_pain", "runny_nose"],
    "Digestive": ["stomach_pain", "vomiting", "nausea", "diarrhoea"],
    "Neurological": ["headache", "dizziness", "loss_of_balance", "altered_sensorium"],
    "Skin": ["itching", "skin_rash", "blister", "red_spots_over_body"],
    "Urinary": ["burning_micturition", "polyuria", "dark_urine"],
    "Cardio": ["chest_pain", "palpitations"],
    "Musculoskeletal": ["joint_pain", "back_pain", "muscle_pain"],
    "Metabolic": ["obesity", "weight_gain", "weight_loss"]
}

# =========================
# BUILD VECTOR
# =========================
def build_vector(symptoms):
    vec = np.zeros(len(features))

    for s in symptoms:
        if s in features:
            idx = list(features).index(s)
            vec[idx] = 1

    return vec.reshape(1, -1)

# =========================
# FIXED PREDICTION (IMPORTANT)
# =========================
def predict(symptoms):
    X = build_vector(symptoms)

    pred_index = model.predict(X)[0]

    # 🔥 FIX: force correct label decoding
    disease = encoder.inverse_transform([int(pred_index)])[0]

    medicine = df[df["prognosis"] == disease]["medicine"].values
    medicine = medicine[0] if len(medicine) > 0 else "Not Available"

    return disease, medicine

# =========================
# SESSION STATE
# =========================
if "symptoms" not in st.session_state:
    st.session_state.symptoms = []

# =========================
# UI
# =========================
st.subheader("Select symptoms")

category = st.selectbox("Choose category", list(categories.keys()))
selected = st.multiselect("Symptoms", categories[category])

if st.button("➕ Add Symptoms"):
    for s in selected:
        if s not in st.session_state.symptoms:
            st.session_state.symptoms.append(s)
    st.success("Added!")

st.write("### Selected Symptoms")
st.write(st.session_state.symptoms)

# =========================
# PREDICT
# =========================
if st.button("🩺 Predict Disease"):

    if len(st.session_state.symptoms) == 0:
        st.error("Please select symptoms first")
    else:
        disease, medicine = predict(st.session_state.symptoms)

        st.success(f"🧾 Disease: {disease}")
        st.info(f"💊 Medicine: {medicine}")

# =========================
# RESET
# =========================
if st.button("🔄 Reset"):
    st.session_state.symptoms = []
