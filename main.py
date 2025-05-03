import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.data_preprocessing import load_data, explore_data, split_data, normalize_data
from utils.membership_functions import generate_triangular_mfs, generate_gaussian_mfs, plot_membership_functions
from utils.rule_generator import generate_rule_base
from fuzzy_models.inference.mamdani_engine import MamdaniEngine
from evaluation.error_metrics import evaluate_model, compare_models
from evaluation.visualization import plot_predictions_vs_actual, plot_error_distribution, plot_fuzzy_inference_example

def create_directories():
    """Gerekli dizinleri oluşturur"""
    dirs = ['data', 'results', 'fuzzy_models', 'evaluation', 'utils', 'report', 'presentation']
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def run_experiment(mf_type, defuzz_method, X_train, X_test, y_train, y_test, data_ranges, output_range):
    """
    Belirli bir kombinasyon için bulanık çıkarım deneyini çalıştırır
    
    Args:
        mf_type: 'triangular' veya 'gaussian'
        defuzz_method: 'center_of_sums' veya 'weighted_average'
        X_train, X_test, y_train, y_test: Eğitim ve test verileri
        data_ranges: Her değişken için (min, max) değerlerini içeren sözlük
        output_range: Çıkış değişkeni aralığı (min, max)
        
    Returns:
        dict: Model sonuçları
    """
    model_name = f"{mf_type.capitalize()}_{defuzz_method.replace('_', '_')}"
    print(f"\nÇalıştırılıyor: {model_name}")
    
    # Bulanık çıkarım sistemi oluştur
    fis = MamdaniEngine(mf_type=mf_type)
    
    # Değişken isimlerini al
    variable_names = X_train.columns
    
    # Giriş değişkenleri için bulanık kümeleri tanımla
    num_sets = 5  # Her değişken için bulanık küme sayısı
    labels = ['Very_Low', 'Low', 'Medium', 'High', 'Very_High']
    
    for var_name in variable_names:
        if mf_type == 'triangular':
            mfs = generate_triangular_mfs(data_ranges[var_name], num_sets)
        else:  # gaussian
            mfs = generate_gaussian_mfs(data_ranges[var_name], num_sets)
        
        fis.add_input_variable(var_name, mfs, labels)
        
        # Üyelik fonksiyonlarını görselleştir
        plot_membership_functions(
            mf_type, mfs, data_ranges[var_name], labels,
            f"{var_name} {mf_type.capitalize()} Üyelik Fonksiyonları",
            f"{var_name}_{mf_type}_mfs"
        )
    
    # Çıkış değişkeni için bulanık kümeleri tanımla
    if mf_type == 'triangular':
        output_mfs = generate_triangular_mfs(output_range, num_sets)
    else:  # gaussian
        output_mfs = generate_gaussian_mfs(output_range, num_sets)
    
    fis.add_output_variable('ale', output_mfs, labels)
    
    # Üyelik fonksiyonlarını görselleştir
    plot_membership_functions(
        mf_type, output_mfs, output_range, labels,
        f"ALE {mf_type.capitalize()} Üyelik Fonksiyonları",
        f"ale_{mf_type}_mfs"
    )
    
    # Kural tabanını oluştur
    rules = generate_rule_base(num_sets, num_sets, num_sets, num_sets, num_sets)
    fis.add_rules(rules)
    
    # Eğitim verisinde tahminler yap
    y_train_pred = []
    for _, row in X_train.iterrows():
        input_values = {col: row[col] for col in X_train.columns}
        pred = fis.predict(input_values, defuzzification=defuzz_method, output_range=output_range)
        y_train_pred.append(pred)
    
    # Test verisinde tahminler yap
    y_test_pred = []
    for _, row in X_test.iterrows():
        input_values = {col: row[col] for col in X_test.columns}
        pred = fis.predict(input_values, defuzzification=defuzz_method, output_range=output_range)
        y_test_pred.append(pred)
    
    # Değerlendirme
    train_metrics = evaluate_model(f"{model_name}_Train", y_train, y_train_pred)
    test_metrics = evaluate_model(f"{model_name}_Test", y_test, y_test_pred)
    
    print(f"Eğitim MAE: {train_metrics['mae']:.4f}, RMSE: {train_metrics['rmse']:.4f}")
    print(f"Test MAE: {test_metrics['mae']:.4f}, RMSE: {test_metrics['rmse']:.4f}")
    
    # Tahminleri görselleştir
    plot_predictions_vs_actual(f"{model_name}_Train", y_train, y_train_pred)
    plot_predictions_vs_actual(f"{model_name}_Test", y_test, y_test_pred)
    
    # Hata dağılımını görselleştir
    plot_error_distribution(f"{model_name}_Train", y_train, y_train_pred)
    plot_error_distribution(f"{model_name}_Test", y_test, y_test_pred)
    
    # Örnek çıkarım sürecini görselleştir (test veri setinin ilk örneği)
    if len(X_test) > 0:
        sample_input = {col: X_test.iloc[0][col] for col in X_test.columns}
        plot_fuzzy_inference_example(fis, sample_input, output_range)
    
    # Sonuçları kaydet
    predictions_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_test_pred,
        'error': np.array(y_test) - np.array(y_test_pred)
    })
    predictions_df.to_csv(f'results/{model_name}_predictions.csv', index=False)
    
    return test_metrics

def main():
    """Ana uygulama fonksiyonu"""
    print("Kablosuz Sensör Ağlarında Bulanık Mantık ile Düğüm Lokalizasyonu")
    print("=" * 70)
    
    # Dizinleri oluştur
    create_directories()
    
    # Veriyi yükle ve analiz et
    data_path = 'data/sensor_localization_data.csv'
    data = load_data(data_path)
    data = explore_data(data)
    
    # Veriyi böl
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Veri aralıklarını belirle
    data_ranges = {}
    for col in X_train.columns:
        min_val = min(X_train[col].min(), X_test[col].min())
        max_val = max(X_train[col].max(), X_test[col].max())
        data_ranges[col] = (min_val, max_val)
    
    output_range = (min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max()))
    
    # Tüm kombinasyonları dene
    results = []
    
    for mf_type in ['triangular', 'gaussian']:
        for defuzz_method in ['center_of_sums', 'weighted_average']:
            test_metrics = run_experiment(
                mf_type, defuzz_method,
                X_train, X_test, y_train, y_test,
                data_ranges, output_range
            )
            results.append(test_metrics)
    
    # Sonuçları karşılaştır
    results_df = pd.DataFrame(results)
    compare_models(results_df)
    results_df.to_csv('results/model_comparison.csv', index=False)
    
    print("\nUygulama tamamlandı! Sonuçlar 'results/' klasöründe.")

if __name__ == "__main__":
    main()