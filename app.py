import streamlit as st
import numpy as np
import joblib

# =========================
# LOAD MODEL + ENCODER
# =========================
model = joblib.load("disease_predictor_model.pkl")
encoder = joblib.load("label_encoder.pkl")
symptom_dict = joblib.load("symptom_dict.pkl")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 AI Health Assistant")
st.write("Select your symptoms and get instant prediction")

# =========================
# SYMPTOM CATEGORIES
# =========================
category_map = {
    "General": ["itching", "fatigue", "weight_loss", "high_fever", "chills"],
    "Respiratory": ["cough", "breathlessness", "chest_pain", "sore_throat"],
    "Digestive": ["nausea", "vomiting", "abdominal_pain", "acidity"],
    "Skin": ["skin_rash", "itching", "blister"],
    "Neurological": ["headache", "dizziness", "loss_of_balance"],
    "Urinary": ["burning_micturition", "urination_frequent"]
}

# =========================
# FORM UI
# =========================
with st.form("predict_form"):

    age = st.number_input("Age", 0, 120, 25)
    gender = st.radio("Gender", ["Male", "Female"])

    category = st.selectbox("Choose Symptom Category", list(category_map.keys()))
    symptoms = st.multiselect("Select Symptoms", category_map[category])

    submitted = st.form_submit_button("Predict")

# =========================
# MEDICINE MAP (BASIC)
# =========================
medicine_map = {
    "Fungal infection": "Antifungal cream / Fluconazole",
    "Common Cold": "Paracetamol, Rest, Fluids",
    "Dengue": "Paracetamol, Hydration",
    "Malaria": "Antimalarial treatment (consult doctor)",
    "Pneumonia": "Antibiotics (doctor prescribed)"
}

# =========================
# PREDICTION
# =========================
if submitted:

    if len(symptoms) == 0:
        st.warning("Please select at least one symptom")
        st.stop()

    features = np.zeros(len(symptom_dict))

    for s in symptoms:
        if s in symptom_dict:
            features[symptom_dict[s]] = 1

    prediction = model.predict([features])[0]
    disease = encoder.inverse_transform([prediction])[0]

    try:
        confidence = np.max(model.predict_proba([features])) * 100
    except:
        confidence = None

    medicine = medicine_map.get(disease, "Consult a doctor for proper medication")

    # =========================
    # OUTPUT (CLEAN)
    # =========================
    st.success("Prediction Complete")

    st.markdown("## 🩺 Disease Identified")
    st.markdown(f"### {disease}")

    if confidence:
        st.write(f"Confidence: **{confidence:.2f}%**")

    st.markdown("## 💊 Medicine")
    st.write(medicine)

    st.markdown("## 🛡️ Precautions")
    st.write("""
    - Take proper rest  
    - Drink fluids  
    - Avoid self medication  
    - Maintain hygiene  
    - Consult doctor if symptoms persist  
    """)
