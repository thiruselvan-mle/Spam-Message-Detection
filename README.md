# Spam / Not Spam Message Classifier

A Machine Learning project to classify SMS messages as **Spam** or **Ham (Not Spam)** using NLP (Natural Language Processing).  
This project is built with **Python**, **Scikit-learn**, **Streamlit**, and **Flask**, designed for both educational and practical learning purposes.

---

## Project Overview

Spam messages are one of the most common nuisances in modern communication.  
This project aims to automatically detect spam messages using machine learning techniques such as **TF-IDF Vectorization** and **Naive Bayes Classification**.

---

## Project Structure
```bash
spam-classifier/
│
├── data/
│ ├── raw/ # Original dataset (spam.csv)
│ └── processed/ # Cleaned dataset
│
├── notebooks/ # Jupyter notebooks for step-by-step workflow
│ ├── 01-data-exploration.ipynb
│ ├── 02-data-cleaning.ipynb
│ ├── 03-eda.ipynb
│ ├── 04-model-training.ipynb
│ └── 05-model-evaluation.ipynb
│
├── src/ # Source code for data processing & training
├── models/ # Trained model and vectorizer files (.pkl)
├── app.py # Streamlit or Flask app
├── requirements.txt # Dependencies list
└── README.md
```
---

## Tech Stack

- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **Frameworks:** Streamlit & Flask  
- **Model:** Multinomial Naive Bayes  
- **Vectorization:** TF-IDF (Term Frequency - Inverse Document Frequency)

---

## Features

 - Classifies SMS messages as **Spam** or **Ham (Not Spam)**  
 - Provides **confidence percentage** of prediction  
 - Simple **Streamlit web interface**  
 - Optional **Flask web app** for deployment  
 - Clean and modular project structure  

 ---

 ## How It Works

1. The model is trained on the **SMS Spam Collection Dataset**.  
2. Each message is converted into numerical features using **TF-IDF Vectorizer**.  
3. The trained **Naive Bayes classifier** predicts whether a message is spam or ham.  
4. You can input a message through the web app to see live predictions.

---

## Run Locally (Streamlit App)

```bash
# Clone the repository
git clone https://github.com/thiruselvan-mle/Spam-Message-Detection.git
cd Spam-Message-Detection

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app or Run Flask app
streamlit run app/streamlit_app.py # Then Visit localhost:8501
python -m app/flask_app.py # Then visit localhost:5000
```
---


## Model Performance
```bash
| Metric    | Score |
| --------- | ----- |
| Accuracy  | 98%   |
| Precision | 97%   |
| Recall    | 96%   |
| F1-Score  | 96.5% |
```

---

## License

**This project is licensed under the MIT License**

---

## Author

Thiruselvan Muthuraman
 - Machine Learning Enthusiast | Passionate about AI and Automation
 - Contact: thiruselvan.muthuraman@gamil.com
 - GitHub: https://github.com/thiruselvan-mle