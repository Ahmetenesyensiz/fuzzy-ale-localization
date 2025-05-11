# fuzzy_models/defuzzification/center_of_sums.py

import numpy as np

def center_of_sums(output_range, aggregated_output):
    """
    Center of Sums (COS) berraklaştırma yöntemi.
    """
    numerator = np.sum(output_range * aggregated_output)
    denominator = np.sum(aggregated_output) + 1e-6  # Sıfır bölmeyi önle
    return numerator / denominator
