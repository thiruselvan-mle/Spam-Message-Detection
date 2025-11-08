import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score


def split_data(df):
    x = df['message']
    y = df['label']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y) 
    return x_train, x_test, y_train, y_test

def vectorizer_text(x_train, x_test):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)

    print("\nShape of TF-IDF Vectors:")
    print('Train',x_train.shape)
    print('Test',x_train.shape)
    return x_train, x_test, vectorizer

def compare_algorithm(x_train, y_train):
    models=[]

    models.append(('LR',LogisticRegression()))
    models.append(('MNB',MultinomialNB()))
    models.append(('RF',RandomForestClassifier(n_estimators=100, random_state=42)))
    models.append(('DT',DecisionTreeClassifier()))
    models.append(('SVC',SVC()))
    models.append(('KNN',KNeighborsClassifier()))

    results=[]
    names=[]
    res=[]
    results_df=[]

    for name,model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=None)
        cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        res.append(cv_results.mean())
        results_df.append({'Model':name, 'Accuracy':cv_results.mean()*100})

    results_df = pd.DataFrame(results_df).sort_values(by='Accuracy', ascending=False)
    display(results_df)
    return results_df, results, names, res

def compare_acuu(names,res):
    plt.figure(figsize=(6,3))
    sns.barplot(x=names, y=res, palette='Set2')
    plt.title('Algorithm Comparison')
    plt.xlabel('Models name')
    plt.ylabel('Accuracy')
    plt.ylim(.900,1)
    plt.show()

def train_model(x_train,y_train):
    model = MultinomialNB(alpha =0.5)
    model.fit(x_train,y_train)
    return model

def save_model_vector(path,model,vectorizer):
    if not os.path.exists(path):
        os.makedirs(path)

    path_model = os.path.join(path,"spam_model.pkl")
    path_vector = os.path.join(path,"spam_vectorizer.pkl")
    joblib.dump(model,path_model)
    joblib.dump(vectorizer,path_vector)

    print("Model and Vectorizer Saved Successfully")

def load_model_vector(path):
    loaded = joblib.load(path)
    print(f"Loaded successfully from {path}")
    return loaded

def transform_test(x_test, vectorizer):
    x_test=vectorizer.transform(x_test)
    return x_test 
 
def make_predict(x_test, model):
    y_pred = model.predict(x_test)
    return y_pred

def evaluate_metrics(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Model Evaluation Results:")
    print(f"Accuracy: {acc*100:.2f}")
    print(f"Precision: {prec*100:.2f}")
    print(f"Recall: {rec*100:.2f}")
    print(f"f1_score: {f1*100:.2f}")
    return acc, prec, rec, f1

def cls_report(y_pred,y_test):
    print(classification_report(y_pred,y_test, target_names=['Ham','Spam']))

def plot_cm(y_pred,y_test):
    cm = confusion_matrix(y_pred,y_test)

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, fmt='d', annot=True, cmap='Blues', xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def test_custom_msg(vectorizer, model, msg):
    test_message = [msg]
    test_vectorizer = vectorizer.transform(test_message)
    prediction = model.predict(test_vectorizer)

    if prediction == 1:
        print(f"\nMessage '{test_message[0]}' - predicted as Spam")
    else:
        print(f"\nmessage '{test_message[0]}' - predicted as Ham(Not Spam)")