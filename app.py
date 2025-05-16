import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_iris

st.title("Iris Flower Prediction App")

# Load model
model = joblib.load("iris_model.pkl")

# User input
sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.0)

features = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(features)
target_names = load_iris().target_names

st.write(f"### Predicted class: {target_names[prediction[0]]}")
