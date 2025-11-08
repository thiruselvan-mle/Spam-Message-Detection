import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

Base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))

model_path = (os.path.join(Base_dir, "models","spam_model.pkl"))
vectorizer_path = (os.path.join(Base_dir,"models","spam_vectorizer.pkl"))

@st.cache_resource
def load_artifacts(_model_path, _vectorizer_path):
    if not os.path.exists(_model_path) or not os.path.exists(_vectorizer_path):
        st.error("Model or Vectorizer path not found")
        return None, None
    
    model = joblib.load(_model_path)
    vectorizer = joblib.load(_vectorizer_path)
    return model, vectorizer

model, vectorizer = load_artifacts(model_path, vectorizer_path)

st.set_page_config(page_title="Spam-Message-Classifier", page_icon="📩", layout="centered")

st.markdown("""
    <style>
    /* Background gradient */
    body {
        background: linear-gradient(135deg, #74ebd5 0%, #ACB6E5 100%);
    }

    /* Main content area */
    .main {
        background-color: #ffffff;
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    /* Title style */
    .title {
        text-align: center;
        font-size: 38px;
        font-weight: 800;
        color: #2c3e50;
        margin-bottom: 5px;
    }

    /* Subtitle style */
    .subtitle {
        text-align: center;
        font-size: 16px;
        color: #555;
        margin-bottom: 30px;
    }

    /* Text area style */
    textarea {
        border-radius: 12px !important;
        border: 1px solid #ccc;
    }

    /* Button style */
    div.stButton > button {
        background-color: black ;
        color: white;
        font-weight: 600;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: blue;
        color : white;
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)


st.markdown('<div class="title">📩 Spam Message Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Machine Learning model to classify SMS messages as Spam or Ham (Not Spam)</div>', unsafe_allow_html=True)


user_input = st.text_area("Enter Your Message Below:", height=150)

if st.button("Check Message"):
    if  user_input.strip() == "":
        st.warning("Please enter a message to analysis")
    
    else:
        data=[user_input]
        data_vector = vectorizer.transform(data).toarray()
        prediction = model.predict(data_vector)[0]
        prediction_proba = model.predict_proba(data_vector)[0]

        confidence =np.max(prediction_proba)*100

        if prediction == 1:
            st.error("This message is **Spam**")
            st.write(f"Confidence {confidence:.2f}%")
            st.info("This message might contain promotional or fraudulent content.")
        else:
            st.success("This message is **Ham(Not Spam)**")
            st.write(f"Confidence {confidence:.2f}%")
            st.info("This seems like a normal, safe message.")
        
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>"
    "Built with ❤️ by <b>Thiruselvan M</b> | Spam Detection using Machine Learning"
    "</p>",
    unsafe_allow_html=True
)