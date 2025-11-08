import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os

def distribute_labels(df):
    plt.figure(figsize=(6,4))
    sns.countplot(x=df['label'].map({0 : 'Ham', 1 : 'Spam'}), data=df, palette='viridis')
    plt.title('Distribution of Spam and Ham Message')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.show()

def add_msg_length(df):
    df['message_length'] = df['message'].apply(len)
    return df

def compare_avg_length(df):
    avg_length = df.groupby('label')['message_length'].mean()
    print(f'Average essage length of Ham(0) and Spam(1)\n\n{avg_length}')

    plt.figure(figsize=(6,4))
    sns.barplot(x=avg_length.index, y=avg_length.values, palette='cool')
    plt.title('Average Message Length by Label')
    plt.xlabel('labels')
    plt.ylabel('count')
    plt.show()
    return avg_length

def msg_length_distribute(df):
    plt.figure(figsize=(8,5))
    sns.histplot(df[df['label']==0]['message_length'], bins=50, color='green', label='Ham', kde=True)
    sns.histplot(df[df['label']==1]['message_length'], bins=50, color='red', label='Spam', kde=True)
    plt.legend()
    plt.title('Message Length Distribution (Ham vs Spam)')
    plt.xlabel('Message Length')
    plt.ylabel('Frequency')
    plt.show()

def word_count(df):
    df['word_count'] = df['message'].apply(lambda x: len(x.split()))

    plt.figure(figsize=(6,4))
    sns.boxplot(x='label', y='word_count', data=df, palette='Set2')
    plt.title('word count by label')
    plt.xlabel('labels (Ham=0, Spam=1)')
    plt.ylabel('word_count')
    plt.show()
    return df

def corr_heatmap(df):
    plt.figure(figsize=(7,5))
    sns.heatmap(df[['message_length', 'word_count', 'label']].corr(), annot=True, cmap='Blues')
    plt.title('Features Correlation Heatmap')
    plt.show()

def save_eda_summary(df, avg_length, path, filename):
    eda_summary = {
    'Total message':len(df),
    'Spam message':df['label'].sum(),
    'Ham message':len(df) - df['label'].sum(),
    'Average Length (Spam)': avg_length[1],
    'Average Length (Ham)': avg_length[0]
    }

    eda_summary=pd.DataFrame([eda_summary])
    if not os.path.exists(path):
        os.makedirs(path)

    save_path = os.path.join(path,filename)
    eda_summary.to_csv(save_path,index=False)
    print(f"EDA summary successfully saved into {save_path}")