import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="AI Virtual Doctor", layout="centered")

st.title("🩺 AI Virtual Doctor")
st.write("Enter your details like a real consultation.")

# =========================
# LOAD FILES
# =========================
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "disease_predictor_model.pkl"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
df = pd.read_csv(os.path.join(BASE_DIR, "training.csv"))

symptoms = list(df.columns[:-2])  # last two = prognosis, medicine

# =========================
# SESSION STATE
# =========================
if "age" not in st.session_state:
    st.session_state.age = None
if "gender" not in st.session_state:
    st.session_state.gender = None

# =========================
# USER INPUTS
# =========================
st.subheader("Personal Details")

age = st.number_input("Enter Age", min_value=0, max_value=120, value=25)
gender = st.selectbox("Select Gender", ["Male", "Female"])

st.session_state.age = age
st.session_state.gender = gender

# =========================
# CATEGORY SYSTEM
# =========================
categories = {
    "General": ["fatigue", "lethargy", "malaise", "weight_gain", "weight_loss", "loss_of_appetite"],
    "Fever": ["high_fever", "mild_fever", "chills", "shivering", "sweating"],
    "Respiratory": ["cough", "breathlessness", "chest_pain", "phlegm", "runny_nose", "congestion"],
    "Digestive": ["stomach_pain", "abdominal_pain", "vomiting", "nausea", "diarrhoea", "constipation"],
    "Skin": ["itching", "skin_rash", "blister", "pus_filled_pimples", "skin_peeling"],
    "Neuro": ["headache", "dizziness", "loss_of_balance", "blurred_and_distorted_vision"],
    "Urinary": ["burning_micturition", "polyuria", "dark_urine", "bladder_discomfort"],
    "Muscle": ["joint_pain", "back_pain", "neck_pain", "muscle_pain"]
}

st.subheader("Select Symptom Category")

selected_categories = st.multiselect(
    "Choose category",
    list(categories.keys())
)

# collect symptoms
available_symptoms = []
for c in selected_categories:
    available_symptoms.extend(categories[c])

available_symptoms = list(set(available_symptoms))

st.subheader("Select Symptoms")

selected_symptoms = st.multiselect(
    "Choose your symptoms",
    available_symptoms
)

# =========================
# PREDICTION FUNCTION
# =========================
def predict(symptoms_selected):
    vector = [0] * len(symptoms)

    for i, s in enumerate(symptoms):
        if s in symptoms_selected:
            vector[i] = 1

    input_data = np.array([vector])

    pred = model.predict(input_data)[0]

    # decode label safely
    try:
        disease = label_encoder.inverse_transform([pred])[0]
    except:
        disease = str(pred)

    # medicine lookup (SAFE)
    med_row = df[df["prognosis"].astype(str).str.strip() == str(disease)]

    if len(med_row) > 0:
        medicine = med_row["medicine"].values[0]
    else:
        medicine = "Consult a doctor for proper medication."

    return disease, medicine

# =========================
# BUTTON
# =========================
if st.button("🔍 Get Diagnosis"):

    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom")
    else:
        disease, medicine = predict(selected_symptoms)

        st.success("Diagnosis Complete")

        st.markdown("### 🧾 Disease Detected")
        st.markdown(f"## {disease}")

        st.markdown("### 💊 Medicine")
        st.write(medicine)

        st.markdown("### 🧠 Advice")
        st.write("This is an AI suggestion. Please consult a doctor for confirmation.")
