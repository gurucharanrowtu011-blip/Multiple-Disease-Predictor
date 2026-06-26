import streamlit as st
import numpy as np
import joblib
import pandas as pd
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Virtual Doctor", layout="centered")

st.title("🩺 AI Virtual Doctor")
st.write("Select your symptoms like a real consultation with a doctor.")

# =========================
# LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "disease_predictor_model.pkl"))

# Load dataset only for symptom columns + medicine mapping
df = pd.read_csv(os.path.join(BASE_DIR, "training.csv"))

symptoms = list(df.columns[:-2])  # last 2 = prognosis, medicine

# =========================
# CATEGORY MAPPING
# =========================
categories = {
    "🔥 General": [
        "fatigue", "lethargy", "malaise", "weight_gain", "weight_loss",
        "loss_of_appetite", "dehydration", "restlessness"
    ],
    "🤒 Fever / Infection": [
        "high_fever", "mild_fever", "chills", "shivering", "sweating"
    ],
    "🫁 Respiratory": [
        "cough", "breathlessness", "chest_pain", "phlegm",
        "runny_nose", "congestion", "throat_irritation"
    ],
    "🍽️ Digestive": [
        "stomach_pain", "abdominal_pain", "vomiting", "nausea",
        "diarrhoea", "constipation", "acidity", "indigestion"
    ],
    "🧠 Neurological": [
        "headache", "dizziness", "loss_of_balance", "blurred_and_distorted_vision"
    ],
    "🧴 Skin": [
        "itching", "skin_rash", "blister", "pus_filled_pimples",
        "skin_peeling", "red_spots_over_body"
    ],
    "💧 Urinary": [
        "burning_micturition", "polyuria", "dark_urine", "bladder_discomfort"
    ],
    "🦴 Musculoskeletal": [
        "joint_pain", "back_pain", "neck_pain", "muscle_pain"
    ],
    "❤️ Cardiovascular": [
        "chest_pain", "palpitations", "fast_heart_rate"
    ]
}

# =========================
# SESSION STATE
# =========================
if "selected_symptoms" not in st.session_state:
    st.session_state.selected_symptoms = []

# =========================
# CATEGORY SELECT
# =========================
st.subheader("Select Symptom Category")

selected_category = st.multiselect(
    "Choose categories",
    list(categories.keys())
)

# =========================
# SHOW SYMPTOMS BASED ON CATEGORY
# =========================
st.subheader("Select Symptoms")

available_symptoms = []
for cat in selected_category:
    available_symptoms.extend(categories[cat])

available_symptoms = list(set(available_symptoms))

selected = st.multiselect(
    "Your symptoms",
    available_symptoms
)

st.session_state.selected_symptoms = selected

# =========================
# PREDICTION FUNCTION
# =========================
def predict_disease(symptoms_selected):
    input_vector = [0] * len(symptoms)

    for i, sym in enumerate(symptoms):
        if sym in symptoms_selected:
            input_vector[i] = 1

    input_array = np.array([input_vector])

    pred = model.predict(input_array)[0]

    # medicine lookup
    medicine = df[df["prognosis"] == pred]["medicine"].values[0]
    disease = pred

    return disease, medicine

# =========================
# PREDICT BUTTON
# =========================
if st.button("🔍 Predict Disease"):

    if len(st.session_state.selected_symptoms) == 0:
        st.warning("Please select at least one symptom")
    else:
        disease, medicine = predict_disease(st.session_state.selected_symptoms)

        st.success("Diagnosis Complete")

        st.markdown("### 🧾 Disease")
        st.markdown(f"**{disease}**")

        st.markdown("### 💊 Recommended Medicine")
        st.write(medicine)

        st.markdown("### 🛡️ Advice")
        st.write("Consult a doctor if symptoms persist or worsen.")
