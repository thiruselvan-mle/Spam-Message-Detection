from flask import Flask, render_template, url_for, redirect, request
import joblib
import numpy as np
import os

app = Flask(__name__)

Base_path = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
model_path = os.path.join(Base_path, "models", "spam_model.pkl")
vectorizer_path = os.path.join(Base_path, "models", "spam_vectorizer.pkl")

def load_model(model_path, vectorizer_path):
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Model or Vectorizer not loaded")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

model, vectorizer = load_model(model_path, vectorizer_path)

@app.route('/')
def home():
    return redirect(url_for('index'))

@app.route('/index', methods = ['GET', 'POST'])
def index():
    if request.method == 'POST':
        message = request.form.get('message', '').strip()

        if message =="":
            return render_template('result.html', prediction="Please Enter a Message to Analysis", msg_type='none', confidence=0)
        else:
            data = [message]
            vector = vectorizer.transform(data)
            prediction = model.predict(vector).tolist()[0]
            prediction_proba = model.predict_proba(vector)[0]
            confidence = np.max(prediction_proba)*100
            
            if prediction == 1:
                prediction_text ="Spam Message Detected!"
                msg_type = "spam"
            else:
                prediction_text ="Ham (Not Spam) Message"
                msg_type = "ham"
            return render_template('result.html', prediction = prediction_text, msg_type = msg_type, confidence = confidence)
    return render_template("index.html")


if __name__ =='__main__':
    app.run(debug=True)