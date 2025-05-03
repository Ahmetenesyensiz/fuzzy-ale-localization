import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_predictions_vs_actual(model_name, y_true, y_pred):
    """
    Tahmin vs Gerçek değerleri görselleştirir
    
    Args:
        model_name: Model adı
        y_true: Gerçek değerler
        y_pred: Tahmin edilen değerler
    """
    plt.figure(figsize=(10, 6))
    
    # Mükemmel tahmin çizgisi (y=x)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    # Tahmin vs Gerçek
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.title(f'{model_name} - Tahmin vs Gerçek Değerler')
    plt.xlabel('Gerçek ALE')
    plt.ylabel('Tahmin Edilen ALE')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(f'results/{model_name}_predictions.png')
    plt.close()

def plot_error_distribution(model_name, y_true, y_pred):
    """
    Hata dağılımını görselleştirir
    
    Args:
        model_name: Model adı
        y_true: Gerçek değerler
        y_pred: Tahmin edilen değerler
    """
    errors = np.array(y_true) - np.array(y_pred)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title(f'{model_name} - Hata Dağılımı')
    plt.xlabel('Hata (Gerçek - Tahmin)')
    plt.ylabel('Frekans')
    plt.grid(True)
    
    plt.savefig(f'results/{model_name}_error_distribution.png')
    plt.close()

def plot_fuzzy_inference_example(model, input_values, output_range):
    """
    Bulanık çıkarım sürecini görselleştirir
    
    Args:
        model: Bulanık çıkarım modeli
        input_values: Örnek giriş değerleri
        output_range: Çıkış değişkeni aralığı (min, max)
    """
    # Bulanıklaştırma
    fuzzified_inputs = model.fuzzify(input_values)
    
    # Her bir giriş değişkeni için bulanıklaştırma grafiği
    for var_name, memberships in fuzzified_inputs.items():
        plt.figure(figsize=(10, 4))
        
        # Üyelik fonksiyonlarını çiz
        x = np.linspace(0, 1, 1000)  # Normalize edilmiş değerler için
        
        for i, (label, degree) in enumerate(memberships.items()):
            if model.mf_type == 'triangular':
                a, b, c = model.input_mfs[var_name]['mfs'][i]
                y = [triangular_mf(val, a, b, c) for val in x]
            else:  # gaussian
                mean, sigma = model.input_mfs[var_name]['mfs'][i]
                y = [gaussian_mf(val, mean, sigma) for val in x]
            
            plt.plot(x, y, label=f'{label} ({degree:.2f})')
            
            # Giriş değerini vurgula
            plt.axvline(x=input_values[var_name], color='r', linestyle='--')
            plt.text(input_values[var_name], 0.5, f'Giriş: {input_values[var_name]:.2f}', 
                    rotation=90, verticalalignment='center')
        
        plt.title(f'{var_name} Bulanıklaştırma')
        plt.xlabel('Değer')
        plt.ylabel('Üyelik Derecesi')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(f'results/fuzzification_{var_name}.png')
        plt.close()
    
    # Bulanık çıkarım
    rule_activations = model.inference(fuzzified_inputs)
    
    # Çıkış değişkeni için berraklaştırma grafiği
    plt.figure(figsize=(10, 4))
    output_var = list(model.output_mfs.keys())[0]
    output_mfs = model.output_mfs[output_var]['mfs']
    output_labels = model.output_mfs[output_var]['labels']
    
    min_val, max_val = output_range
    x = np.linspace(min_val, max_val, 1000)
    
    # Her bulanık küme için üyelik fonksiyonunu çiz
    for i, label in enumerate(output_labels):
        if model.mf_type == 'triangular':
            a, b, c = output_mfs[i]
            y = [min(triangular_mf(val, a, b, c), rule_activations[label]) for val in x]
        else:  # gaussian
            mean, sigma = output_mfs[i]
            y = [min(gaussian_mf(val, mean, sigma), rule_activations[label]) for val in x]
        
        plt.plot(x, y, label=f'{label} ({rule_activations[label]:.2f})')
    
    # Berraklaştırılmış değeri vurgula
    defuzzified_center = model.defuzzify_center_of_sums(rule_activations, output_range)
    defuzzified_weighted = model.defuzzify_weighted_average(rule_activations)
    
    plt.axvline(x=defuzzified_center, color='r', linestyle='--')
    plt.text(defuzzified_center, 0.5, f'Center of Sums: {defuzzified_center:.2f}', 
            rotation=90, verticalalignment='center')
    
    plt.axvline(x=defuzzified_weighted, color='g', linestyle='--')
    plt.text(defuzzified_weighted, 0.3, f'Weighted Average: {defuzzified_weighted:.2f}', 
            rotation=90, verticalalignment='center')
    
    plt.title(f'{output_var} Berraklaştırma')
    plt.xlabel('Değer')
    plt.ylabel('Üyelik Derecesi')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'results/defuzzification.png')
    plt.close()