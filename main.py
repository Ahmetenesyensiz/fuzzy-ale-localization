# main.py (GÜNCELLENMİŞ)

import numpy as np
import pandas as pd
import os
from utils.data_preprocessing import load_data, split_data, normalize_data
from utils.rule_generator import generate_rule_base
from fuzzy_models.membership.triangular import triangular_membership
from fuzzy_models.membership.gaussian import gaussian_membership
from fuzzy_models.inference.mamdani_engine import MamdaniEngine
from fuzzy_models.defuzzification.center_of_sums import center_of_sums
from fuzzy_models.defuzzification.weighted_average import weighted_average
from evaluation.error_metrics import mean_absolute_error, root_mean_squared_error
from evaluation.visualization import plot_predictions_vs_actual, plot_comparison_metrics

def create_input_mfs(mf_type, feature_ranges):
    input_mfs = {}
    for feature, (min_val, max_val) in feature_ranges.items():
        if mf_type == 'triangular':
            input_mfs[feature] = {
                'low': lambda x, a=min_val, b=(min_val+max_val)/2, c=max_val: triangular_membership(x, a, a, b),
                'medium': lambda x, a=min_val, b=(min_val+max_val)/2, c=max_val: triangular_membership(x, a, b, c),
                'high': lambda x, a=min_val, b=(min_val+max_val)/2, c=max_val: triangular_membership(x, b, c, c)
            }
        elif mf_type == 'gaussian':
            center = (min_val + max_val) / 2
            spread = (max_val - min_val) / 4
            input_mfs[feature] = {
                'low': lambda x, m=min_val, s=spread: gaussian_membership(x, m, s),
                'medium': lambda x, m=center, s=spread: gaussian_membership(x, m, s),
                'high': lambda x, m=max_val, s=spread: gaussian_membership(x, m, s)
            }
    return input_mfs

def create_output_mfs(mf_type, output_range):
    min_val, max_val = np.min(output_range), np.max(output_range)
    if mf_type == 'triangular':
        return {
            'low_ALE': lambda x: triangular_membership(x, min_val, min_val, (min_val+max_val)/2),
            'medium_ALE': lambda x: triangular_membership(x, min_val, (min_val+max_val)/2, max_val),
            'high_ALE': lambda x: triangular_membership(x, (min_val+max_val)/2, max_val, max_val)
        }
    elif mf_type == 'gaussian':
        center = (min_val + max_val) / 2
        spread = (max_val - min_val) / 4
        return {
            'low_ALE': lambda x: gaussian_membership(x, min_val, spread),
            'medium_ALE': lambda x: gaussian_membership(x, center, spread),
            'high_ALE': lambda x: gaussian_membership(x, max_val, spread)
        }

def main():
    os.makedirs('results/plots', exist_ok=True)

    df = load_data('data/sensor_localization_data.csv')
    X, y = split_data(df)
    X, _ = normalize_data(X)

    feature_ranges = {
        'anchor_ratio': (np.min(X[:,0]), np.max(X[:,0])),
        'trans_range': (np.min(X[:,1]), np.max(X[:,1])),
        'node_density': (np.min(X[:,2]), np.max(X[:,2])),
        'iterations': (np.min(X[:,3]), np.max(X[:,3]))
    }

    output_range = np.linspace(min(y), max(y), 100)

    mf_types = ['triangular', 'gaussian']
    defuzz_methods = ['COS', 'WA']
    results = []
    mae_dict = {}
    rmse_dict = {}   # ✅ YENİ EKLENDİ

    for mf_type in mf_types:
        for defuzz_method in defuzz_methods:
            model_name = f"{mf_type} + {defuzz_method}"
            print(f"\nModel: {model_name}")

            input_mfs = create_input_mfs(mf_type, feature_ranges)
            output_mfs = create_output_mfs(mf_type, output_range)
            rules = generate_rule_base()
            engine = MamdaniEngine(input_mfs, output_mfs, rules)

            y_pred = []
            for sample in X:
                inputs = {
                    'anchor_ratio': sample[0],
                    'trans_range': sample[1],
                    'node_density': sample[2],
                    'iterations': sample[3]
                }

                if defuzz_method == 'COS':
                    aggregated = engine.infer(inputs, output_range)
                    prediction = center_of_sums(output_range, aggregated)
                elif defuzz_method == 'WA':
                    fuzzified = engine.fuzzify(inputs)
                    activations = engine.apply_rules(fuzzified)
                    output_centers = {
                        'low_ALE': np.min(y),
                        'medium_ALE': (np.min(y) + np.max(y)) / 2,
                        'high_ALE': np.max(y)
                    }
                    prediction = weighted_average(activations, output_centers)

                y_pred.append(prediction)

            mae = mean_absolute_error(y, y_pred)
            rmse = root_mean_squared_error(y, y_pred)
            print(f"{model_name} → MAE: {mae:.4f}, RMSE: {rmse:.4f}")

            results.append({'Model': model_name, 'MAE': mae, 'RMSE': rmse})
            mae_dict[model_name] = mae
            rmse_dict[model_name] = rmse    # ✅ YENİ EKLENDİ

            plot_path = f"results/plots/{mf_type}_{defuzz_method}.png"
            plot_predictions_vs_actual(y, y_pred, title=f"{model_name} - Gerçek vs Tahmin", save_path=plot_path)

    results_df = pd.DataFrame(results)
    results_df.to_csv('results/comparison_metrics.csv', index=False)
    print("\nTüm metrikler 'results/comparison_metrics.csv' dosyasına kaydedildi.")

    plot_comparison_metrics(mae_dict, metric_name='MAE', save_path="results/overall_comparison_mae.png")
    plot_comparison_metrics(rmse_dict, metric_name='RMSE', save_path="results/overall_comparison_rmse.png")   # ✅ YENİ EKLENDİ

if __name__ == "__main__":
    main()
