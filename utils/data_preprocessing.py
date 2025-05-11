# utils/data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath):
    """
    CSV dosyasını oku ve sd_ale sütununu kaldır.
    """
    df = pd.read_csv(filepath)
    if 'sd_ale' in df.columns:
        df = df.drop(columns=['sd_ale'])
    return df

def explore_data(df):
    """
    Veri seti hakkında temel bilgi ver.
    """
    print("Veri seti boyutu:", df.shape)
    print("\nİlk 5 satır:\n", df.head())
    print("\nVeri seti istatistikleri:\n", df.describe())

def split_data(df):
    """
    Giriş (X) ve hedef (y) değişkenlerini ayır.
    """
    X = df.iloc[:, :-1].values  # İlk 4 sütun: anchor_ratio, trans_range, node_density, iterations
    y = df.iloc[:, -1].values   # Son sütun: ALE
    return X, y

def normalize_data(X):
    """
    Giriş verilerini [0, 1] aralığına normalize et.
    """
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    return X_normalized, scaler

