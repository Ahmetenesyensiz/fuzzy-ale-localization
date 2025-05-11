# utils/rule_generator.py

def generate_rule_base():
    """
    Basit örnek kural tabanı oluşturur.
    
    Her rule: ({giriş değişkenleri ve dilsel etiketleri}, çıkış etiketi)
    """
    rules = [
        # Örnek kurallar
        ({'anchor_ratio': 'low', 'trans_range': 'low', 'node_density': 'low', 'iterations': 'low'}, 'high_ALE'),
        ({'anchor_ratio': 'high', 'trans_range': 'high', 'node_density': 'high', 'iterations': 'high'}, 'low_ALE'),
        ({'anchor_ratio': 'medium', 'trans_range': 'medium', 'node_density': 'medium', 'iterations': 'medium'}, 'medium_ALE'),

        # Ekstra kurallar
        ({'anchor_ratio': 'low', 'trans_range': 'high', 'node_density': 'high', 'iterations': 'low'}, 'medium_ALE'),
        ({'anchor_ratio': 'high', 'trans_range': 'low', 'node_density': 'low', 'iterations': 'high'}, 'medium_ALE'),
        ({'anchor_ratio': 'low', 'trans_range': 'medium', 'node_density': 'medium', 'iterations': 'medium'}, 'high_ALE'),
        ({'anchor_ratio': 'high', 'trans_range': 'high', 'node_density': 'low', 'iterations': 'medium'}, 'low_ALE'),
    ]

    return rules
