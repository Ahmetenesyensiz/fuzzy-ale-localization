# evaluation/visualization.py

import matplotlib.pyplot as plt
import numpy as np

def plot_predictions_vs_actual(y_true, y_pred, title='Gerçek vs Tahmin Edilen ALE', save_path=None):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, color='blue', alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', label='Doğru Tahmin')
    plt.xlabel('Gerçek ALE')
    plt.ylabel('Tahmin ALE')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()   # Ekrana çizmesin, kapatsın (en önemli ekleme!)
    else:
        plt.show()

def plot_comparison_metrics(metrics_dict, metric_name='MAE', save_path=None):
    labels = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color='skyblue')
    plt.xlabel('Model Kombinasyonu')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Karşılaştırması')
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.3f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()   # Yine kapatsın, ekranı açmasın
    else:
        plt.show()
