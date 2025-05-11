# fuzzy_models/membership/triangular.py

import numpy as np

def triangular_membership(x, a, b, c):
    """
    Üçgen üyelik fonksiyonu.
    
    a: sol uç nokta
    b: tepe noktası
    c: sağ uç nokta
    """
    return np.maximum(np.minimum((x - a) / (b - a + 1e-6), (c - x) / (c - b + 1e-6)), 0)
