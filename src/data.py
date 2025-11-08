import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

def load_data(path,encoding='utf-8'):
    try:
        df = pd.read_csv(path,encoding=encoding)
        display('Dataset successfully loaded', df.head())
        return df
    except FileNotFoundError:
        print(f"Error: FileNOtFound at {path}")
        return None

def data_shape(df):
    print('data shape(Rows,Columns):',df.shape)

def col_names(df):
    print('columns name:',df.columns)

def data_info(df):
    print(df.info())

def check_mis_val(df):
    print(df.isnull().sum())

def check_duplicate(df):
    duplicate = df.duplicated().sum()
    print(f"\n Number of duplicate rows:{duplicate}")

def unique_labels(df):
    print("\n unique labels in V1 column:",df['v1'].unique())

def remove_cols(df):
    df=df[['v1','v2']]
    return df

def rename_cols(df):
    df.columns=['label','message']
    df.head()
    return df

def remove_duplicate_rows(df):
    duplicate = df.duplicated().sum()
    print(f"Duplicate rows:{duplicate}")

    df.drop_duplicates(inplace=True)
    print(f"\nAfter_drop Duplicate rows:{df.duplicated().sum()}")
    return df

def encode_labels(df):
    df['label'] = df['label'].map({'ham': 0 , 'spam' : 1})
    return df

def save_clean_data(df , path ,filename):
    if not os.path.exists(path):    
        os.makedirs(path)

    save_path = os.path.join(path,filename)
    df.to_csv(save_path,index=False)
    print(f"Cleaned Dataset save into {save_path}")