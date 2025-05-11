# fuzzy_models/inference/mamdani_engine.py

import numpy as np

class MamdaniEngine:
    def __init__(self, input_membership_functions, output_membership_functions, rules):
        """
        Mamdani inference engine.

        input_membership_functions: dict, her giriş için üyelik fonksiyonları
        output_membership_functions: dict, çıkış için üyelik fonksiyonları
        rules: list of tuples, (input_condition_dict, output_label)
        """
        self.input_mfs = input_membership_functions
        self.output_mfs = output_membership_functions
        self.rules = rules

    def fuzzify(self, inputs):
        """
        Giriş değerlerini bulanıklaştır.
        """
        fuzzified = {}
        for var_name, value in inputs.items():
            fuzzified[var_name] = {}
            for label, mf in self.input_mfs[var_name].items():
                fuzzified[var_name][label] = mf(value)
        return fuzzified

    def apply_rules(self, fuzzified_inputs):
        """
        Kuralları uygula ve çıktı derecelerini hesapla.
        """
        output_activations = {}
        for rule in self.rules:
            input_conditions, output_label = rule
            firing_strengths = []
            for var_name, label in input_conditions.items():
                strength = fuzzified_inputs[var_name][label]
                firing_strengths.append(strength)
            rule_strength = min(firing_strengths)  # Mamdani: min operator
            if output_label in output_activations:
                output_activations[output_label] = max(output_activations[output_label], rule_strength)
            else:
                output_activations[output_label] = rule_strength
        return output_activations

    def aggregate(self, output_activations, output_range):
        """
        Çıkış bulanık kümesini oluştur (max aggregation).
        """
        aggregated = np.zeros_like(output_range, dtype=float)
        for label, activation in output_activations.items():
            mf_values = np.array([self.output_mfs[label](x) for x in output_range])
            aggregated = np.maximum(aggregated, np.minimum(activation, mf_values))
        return aggregated

    def infer(self, inputs, output_range):
        """
        Tam inference işlemi: fuzzification → rule evaluation → aggregation.
        """
        fuzzified = self.fuzzify(inputs)
        output_activations = self.apply_rules(fuzzified)
        aggregated_output = self.aggregate(output_activations, output_range)
        return aggregated_output
