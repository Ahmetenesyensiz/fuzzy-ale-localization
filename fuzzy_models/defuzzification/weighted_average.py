# fuzzy_models/defuzzification/weighted_average.py

import numpy as np

def weighted_average(output_activations, output_centers):
    """
    Weighted Average (WA) berraklaştırma yöntemi.
    
    output_activations: {label: activation_degree}
    output_centers: {label: center_value of output MF}
    """
    numerator = 0.0
    denominator = 0.0
    for label, activation in output_activations.items():
        numerator += activation * output_centers[label]
        denominator += activation
    if denominator == 0:
        return 0
    return numerator / denominator
