import streamlit as st
import pickle
import numpy as np

# ---- Load the trained model ----
with open("astro_model.pkl", "rb") as f:
    model = pickle.load(f)

# ---- Load the scaler ----
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---- App Title ----
st.set_page_config(page_title="Astro Object Classifier", layout="centered")
st.title("🌌 Astro Object Classifier")
st.write("Enter the features of an astronomical object to classify it as Star, Quasar, or Galaxy.")

# ---- User Inputs ----
st.subheader("Input Features")
u = st.number_input("Ultraviolet magnitude (u)", value=0.0)
g = st.number_input("Green magnitude (g)", value=0.0)
r = st.number_input("Red magnitude (r)", value=0.0)
i = st.number_input("Near-infrared magnitude (i)", value=0.0)
z = st.number_input("Infrared magnitude (z)", value=0.0)
redshift = st.number_input("Redshift", value=0.0)

input_values = [u, g, r, i, z, redshift]

# ---- Prediction ----
if st.button("Predict"):
    input_array = np.array(input_values).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)  # Scale input using the trained scaler
    
    prediction_num = model.predict(input_array_scaled)[0]
    
    if prediction_num == 0:
        prediction = "Star"
    elif prediction_num == 1:
        prediction = "Quasar"
    else:
        prediction = "Galaxy"

    st.success(f"✅ This object is predicted as: **{prediction}**")

    # Optional: Show prediction probabilities if available
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_array_scaled)[0]
        st.subheader("Prediction Probabilities")
        for cls, prob in zip(model.classes_, probs):
            st.write(f"{cls}: {prob:.2f}")
