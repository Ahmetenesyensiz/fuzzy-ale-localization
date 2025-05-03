import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path = "fuzzy_node_localization_project/data/mcs_ds_edited_iter_shuffled.csv"):
    """Veri dosyasını yükler ve gerekli ön işlemeleri yapar"""
    data = pd.read_csv(file_path)
    # Sütun isimlerini düzenleme
    data.columns = ['anchor_ratio', 'transmission_range', 'node_density', 'iteration_count', 'ale', 'std_dev']
    # Son sütunu atmamız isteniyor (standart sapma)
    data = data.drop('std_dev', axis=1)
    return data

def explore_data(data):
    """Veri seti hakkında temel istatistikler ve grafikler oluşturur"""
    # Temel istatistikler
    print("Veri Seti Boyutu:", data.shape)
    print("\nİlk 5 Satır:")
    print(data.head())
    print("\nİstatistiksel Özet:")
    print(data.describe())
    
    # Korelasyon matrisi
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Özellikler Arası Korelasyon')
    plt.savefig('results/correlation_matrix.png')
    
    # Giriş değişkenlerinin dağılımları
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(data['anchor_ratio'], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Çapa Oranı Dağılımı')
    
    sns.histplot(data['transmission_range'], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('İletim Aralığı Dağılımı')
    
    sns.histplot(data['node_density'], kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Düğüm Yoğunluğu Dağılımı')
    
    sns.histplot(data['iteration_count'], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Yineleme Sayısı Dağılımı')
    
    plt.tight_layout()
    plt.savefig('results/feature_distributions.png')
    
    # Çıkış (ALE) dağılımı
    plt.figure(figsize=(10, 6))
    sns.histplot(data['ale'], kde=True)
    plt.title('ALE (Ortalama Lokalizasyon Hatası) Dağılımı')
    plt.savefig('results/ale_distribution.png')
    
    # Giriş-çıkış ilişkileri
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.scatterplot(x='anchor_ratio', y='ale', data=data, ax=axes[0, 0])
    axes[0, 0].set_title('Çapa Oranı vs ALE')
    
    sns.scatterplot(x='transmission_range', y='ale', data=data, ax=axes[0, 1])
    axes[0, 1].set_title('İletim Aralığı vs ALE')
    
    sns.scatterplot(x='node_density', y='ale', data=data, ax=axes[1, 0])
    axes[1, 0].set_title('Düğüm Yoğunluğu vs ALE')
    
    sns.scatterplot(x='iteration_count', y='ale', data=data, ax=axes[1, 1])
    axes[1, 1].set_title('Yineleme Sayısı vs ALE')
    
    plt.tight_layout()
    plt.savefig('results/input_output_relationships.png')
    
    return data

def split_data(data, test_size=0.2, random_state=42):
    """Veriyi eğitim ve test kümelerine ayırır"""
    from sklearn.model_selection import train_test_split
    X = data.drop('ale', axis=1)
    y = data['ale']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

def normalize_data(X_train, X_test):
    """Giriş verilerini 0-1 aralığına normalize eder"""
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler