import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_iris

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# ---------------------------
# Title
# ---------------------------
st.markdown("<h1 style='text-align: center; color: #6C3483;'>🌸 Iris Flower Prediction App 🌸</h1>", unsafe_allow_html=True)
st.write("---")

# ---------------------------
# Load Model
# ---------------------------
MODEL_PATH = "iris_model.pkl"

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

model = load_model(MODEL_PATH)
target_names = load_iris().target_names

# ---------------------------
# User Input
# ---------------------------
st.subheader("Enter flower measurements:")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8)
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.35)
with col2:
    sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
    petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.3)

features = [[sepal_length, sepal_width, petal_length, petal_width]]

# ---------------------------
# Prediction Button
# ---------------------------
if st.button("🔍 Predict Flower Type"):
    prediction = model.predict(features)
    predicted_class = target_names[prediction[0]]
    st.success(f"🌼 **Predicted Class:** {predicted_class.capitalize()}")

    # Optional: Show details
    st.write("#### 🌱 Feature Summary:")
    summary_df = pd.DataFrame(features, columns=[
        "Sepal Length", "Sepal Width", "Petal Length", "Petal Width"
    ])
    st.dataframe(summary_df.style.set_precision(2), use_container_width=True)
else:
    st.info("Adjust the sliders and click **Predict Flower Type** to get a result.")
