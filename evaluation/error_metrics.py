import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_mae(y_true, y_pred):
    """
    Ortalama Mutlak Hata (Mean Absolute Error - MAE) hesaplar
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin edilen değerler
        
    Returns:
        float: MAE değeri
    """
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def calculate_rmse(y_true, y_pred):
    """
    Kök Ortalama Kare Hata (Root Mean Squared Error - RMSE) hesaplar
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin edilen değerler
        
    Returns:
        float: RMSE değeri
    """
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

def evaluate_model(model_name, y_true, y_pred):
    """
    Modeli değerlendirir ve sonuçları döndürür
    
    Args:
        model_name: Model adı
        y_true: Gerçek değerler
        y_pred: Tahmin edilen değerler
        
    Returns:
        dict: Değerlendirme metrikleri
    """
    mae = calculate_mae(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    
    return {
        'model': model_name,
        'mae': mae,
        'rmse': rmse
    }

def compare_models(results_df):
    """
    Modelleri karşılaştırır ve görselleştirir
    
    Args:
        results_df: Model sonuçlarını içeren veri çerçevesi
    """
    # MAE karşılaştırması
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    bars = plt.bar(results_df['model'], results_df['mae'])
    plt.title('Model Karşılaştırması - MAE')
    plt.xlabel('Model')
    plt.ylabel('MAE (Ortalama Mutlak Hata)')
    plt.xticks(rotation=45, ha='right')
    
    # Değerleri çubukların üzerine yazdır
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')
    
    # RMSE karşılaştırması
    plt.subplot(1, 2, 2)
    bars = plt.bar(results_df['model'], results_df['rmse'])
    plt.title('Model Karşılaştırması - RMSE')
    plt.xlabel('Model')
    plt.ylabel('RMSE (Kök Ortalama Kare Hata)')
    plt.xticks(rotation=45, ha='right')
    
    # Değerleri çubukların üzerine yazdır
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png')
    plt.close()
    
    # En iyi modeli bulma
    best_mae_idx = results_df['mae'].idxmin()
    best_rmse_idx = results_df['rmse'].idxmin()
    
    print(f"En düşük MAE: {results_df.loc[best_mae_idx, 'model']} - {results_df.loc[best_mae_idx, 'mae']:.4f}")
    print(f"En düşük RMSE: {results_df.loc[best_rmse_idx, 'model']} - {results_df.loc[best_rmse_idx, 'rmse']:.4f}")
    
    return results_df