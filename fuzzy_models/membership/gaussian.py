# fuzzy_models/membership/gaussian.py

import numpy as np

def gaussian_membership(x, mean, sigma):
    """
    Gauss üyelik fonksiyonu.
    
    mean: ortalama (tepe noktası)
    sigma: standart sapma (yayılma)
    """
    return np.exp(-0.5 * ((x - mean) / (sigma + 1e-6)) ** 2)
