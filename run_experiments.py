import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_preprocessing import load_data, explore_data, split_data
from utils.membership_functions import generate_triangular_mfs, generate_gaussian_mfs
from utils.rule_generator import generate_rule_base
from fuzzy_models.inference.mamdani_engine import MamdaniEngine
from evaluation.error_metrics import evaluate_model, compare_models
from evaluation.visualization import plot_predictions_vs_actual, plot_error_distribution

def run_all_experiments():
    """
    Tüm bulanık mantık kombinasyonlarını çalıştırır ve sonuçları karşılaştırır
    """
    print("Tüm deneyler çalıştırılıyor...")
    
    # Veriyi yükle
    data_path = 'data/sensor_localization_data.csv'
    print(f"Veri dosyası yükleniyor: {data_path}")
    data = load_data(data_path)
    
    # Veriyi analiz et
    data = explore_data(data)
    
    # Veriyi böl
    X_train, X_test, y_train, y_test = split_data(data)
    print(f"Eğitim veri boyutu: {X_train.shape}")
    print(f"Test veri boyutu: {X_test.shape}")
    
    # Veri aralıklarını belirle
    data_ranges = {}
    for col in X_train.columns:
        min_val = min(data[col].min(), 0)  # 0'dan küçük değerler varsa 0 kabul et
        max_val = data[col].max()
        data_ranges[col] = (min_val, max_val)
        print(f"{col} aralığı: ({min_val:.4f}, {max_val:.4f})")
    
    output_range = (data['ale'].min(), data['ale'].max())
    print(f"ALE aralığı: ({output_range[0]:.4f}, {output_range[1]:.4f})")
    
    # Kombinasyon parametreleri
    mf_types = ['triangular', 'gaussian']
    defuzz_methods = ['center_of_sums', 'weighted_average']
    
    # Sonuçları depolamak için
    results = []
    predictions = {}
    
    # Her kombinasyon için
    for mf_type in mf_types:
        for defuzz_method in defuzz_methods:
            model_name = f"{mf_type}_{defuzz_method}"
            print(f"\n{'-'*40}")
            print(f"Çalıştırılıyor: {model_name}")
            print(f"{'-'*40}")
            
            # Bulanık çıkarım sistemi oluştur
            fis = MamdaniEngine(mf_type=mf_type)
            
            # Giriş değişkenleri için bulanık kümeleri tanımla
            num_sets = 5  # Her değişken için bulanık küme sayısı
            labels = ['Very_Low', 'Low', 'Medium', 'High', 'Very_High']
            
            for var_name in X_train.columns:
                if mf_type == 'triangular':
                    mfs = generate_triangular_mfs(data_ranges[var_name], num_sets)
                else:  # gaussian
                    mfs = generate_gaussian_mfs(data_ranges[var_name], num_sets)
                
                fis.add_input_variable(var_name, mfs, labels)
            
            # Çıkış değişkeni için bulanık kümeleri tanımla
            if mf_type == 'triangular':
                output_mfs = generate_triangular_mfs(output_range, num_sets)
            else:  # gaussian
                output_mfs = generate_gaussian_mfs(output_range, num_sets)
            
            fis.add_output_variable