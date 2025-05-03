import numpy as np
from utils.membership_functions import triangular_mf, gaussian_mf

class MamdaniEngine:
    def __init__(self, mf_type='triangular'):
        """Mamdani bulanık çıkarım sistemi
        
        Args:
            mf_type: 'triangular' veya 'gaussian'
        """
        self.mf_type = mf_type
        self.input_mfs = {}  # Her bir giriş değişkeni için üyelik fonksiyonları
        self.output_mfs = {}  # Çıkış değişkeni için üyelik fonksiyonları
        self.rules = []  # Bulanık kurallar
        
    def add_input_variable(self, name, mfs, labels):
        """Giriş değişkeni ekleme
        
        Args:
            name: Değişken adı
            mfs: Üyelik fonk. parametreleri listesi
            labels: Bulanık küme etiketleri
        """
        self.input_mfs[name] = {'mfs': mfs, 'labels': labels}
        
    def add_output_variable(self, name, mfs, labels):
        """Çıkış değişkeni ekleme
        
        Args:
            name: Değişken adı
            mfs: Üyelik fonk. parametreleri listesi
            labels: Bulanık küme etiketleri
        """
        self.output_mfs[name] = {'mfs': mfs, 'labels': labels}
        
    def add_rules(self, rules):
        """Kural tabanı ekleme
        
        Args:
            rules: Kural listesi
        """
        self.rules = rules
        
    def fuzzify(self, input_values):
        """Giriş değerlerini bulanıklaştırma
        
        Args:
            input_values: Giriş değerleri sözlüğü {değişken_adı: değer}
            
        Returns:
            dict: Her değişken için bulanık üyelik dereceleri
        """
        fuzzified = {}
        
        for var_name, var_value in input_values.items():
            if var_name not in self.input_mfs:
                continue
                
            var_fuzzified = {}
            mfs = self.input_mfs[var_name]['mfs']
            labels = self.input_mfs[var_name]['labels']
            
            for i, label in enumerate(labels):
                if self.mf_type == 'triangular':
                    a, b, c = mfs[i]
                    degree = triangular_mf(var_value, a, b, c)
                else:  # gaussian
                    mean, sigma = mfs[i]
                    degree = gaussian_mf(var_value, mean, sigma)
                    
                var_fuzzified[label] = degree
                
            fuzzified[var_name] = var_fuzzified
            
        return fuzzified
        
    def inference(self, fuzzified_inputs):
        """Bulanık çıkarım aşaması
        
        Args:
            fuzzified_inputs: Bulanıklaştırılmış giriş değerleri
            
        Returns:
            dict: Çıkış değişkeni bulanık kümelerinin aktivasyon dereceleri
        """
        # Çıkış değişkeni adını al
        output_var = list(self.output_mfs.keys())[0]
        output_labels = self.output_mfs[output_var]['labels']
        
        # Her çıkış bulanık kümesi için aktivasyon derecesi
        rule_activations = {label: 0.0 for label in output_labels}
        
        # Her kural için
        for rule in self.rules:
            # Kural formatı: (anchor_label, range_label, density_label, iteration_label, ale_label)
            # None değeri "don't care" anlamına gelir
            
            input_vars = list(self.input_mfs.keys())
            min_activation = 1.0  # AND operatörü için başlangıç değeri
            
            # Kuralın giriş kısmını değerlendir
            for i, var_name in enumerate(input_vars):
                if rule[i] is not None:  # Bu değişken kural içinde belirtilmiş
                    activation = fuzzified_inputs[var_name][rule[i]]
                    min_activation = min(min_activation, activation)
            
            # Çıkış bulanık kümesindeki aktivasyonu güncelle
            output_label = rule[-1]
            # MAX operatörü - aynı çıkışı üreten farklı kuralların maksimum aktivasyonu
            rule_activations[output_label] = max(rule_activations[output_label], min_activation)
        
        return rule_activations
    
    def defuzzify_center_of_sums(self, rule_activations, output_range, num_points=1000):
        """Toplamların Merkezi berraklaştırma yöntemi
        
        Args:
            rule_activations: Kural aktivasyon dereceleri
            output_range: Çıkış değişkeni aralığı (min, max)
            num_points: Ayrıklaştırma nokta sayısı
            
        Returns:
            float: Berraklaştırılmış çıkış değeri
        """
        output_var = list(self.output_mfs.keys())[0]
        output_mfs = self.output_mfs[output_var]['mfs']
        output_labels = self.output_mfs[output_var]['labels']
        
        min_val, max_val = output_range
        x = np.linspace(min_val, max_val, num_points)
        
        # Her bir çıkış değeri için toplam alan hesapla
        numerator = 0.0
        denominator = 0.0
        
        for i in x:
            sum_mu = 0.0
            
            # Her bulanık küme için üyelik değerlerini topla
            for j, label in enumerate(output_labels):
                if self.mf_type == 'triangular':
                    a, b, c = output_mfs[j]
                    mu = triangular_mf(i, a, b, c)
                else:  # gaussian
                    mean, sigma = output_mfs[j]
                    mu = gaussian_mf(i, mean, sigma)
                
                # Kural aktivasyonu ile ağırlıklandır
                mu = min(mu, rule_activations[label])
                sum_mu += mu
            
            numerator += i * sum_mu
            denominator += sum_mu
        
        if denominator == 0:
            return (min_val + max_val) / 2  # Varsayılan değer
        
        return numerator / denominator
    
    def defuzzify_weighted_average(self, rule_activations):
        """Ağırlıklı Ortalama berraklaştırma yöntemi
        
        Args:
            rule_activations: Kural aktivasyon dereceleri
            
        Returns:
            float: Berraklaştırılmış çıkış değeri
        """
        output_var = list(self.output_mfs.keys())[0]
        output_mfs = self.output_mfs[output_var]['mfs']
        output_labels = self.output_mfs[output_var]['labels']
        
        numerator = 0.0
        denominator = 0.0
        
        for i, label in enumerate(output_labels):
            activation = rule_activations[label]
            
            if activation > 0:
                if self.mf_type == 'triangular':
                    # Üçgen üyelik fonksiyonu için merkez noktası
                    a, b, c = output_mfs[i]
                    center = b  # Tepe noktası
                else:  # gaussian
                    # Gauss üyelik fonksiyonu için merkez
                    mean, _ = output_mfs[i]
                    center = mean
                
                numerator += center * activation
                denominator += activation
        
        if denominator == 0:
            # Varsayılan değer (çıkış aralığının ortası)
            if self.mf_type == 'triangular':
                return output_mfs[len(output_mfs)//2][1]  # Orta kümenin merkezi
            else:
                return output_mfs[len(output_mfs)//2][0]  # Orta kümenin merkezi
        
        return numerator / denominator
    
    def predict(self, input_values, defuzzification='center_of_sums', output_range=None):
        """Giriş değerleri için çıkış tahmini yapar
        
        Args:
            input_values: Giriş değerleri sözlüğü {değişken_adı: değer}
            defuzzification: Berraklaştırma yöntemi ('center_of_sums' veya 'weighted_average')
            output_range: Çıkış değişkeni aralığı (min, max), yalnızca 'center_of_sums' için gerekli
            
        Returns:
            float: Berraklaştırılmış çıkış değeri
        """
        # Bulanıklaştırma
        fuzzified_inputs = self.fuzzify(input_values)
        
        # Bulanık çıkarım
        rule_activations = self.inference(fuzzified_inputs)
        
        # Berraklaştırma
        if defuzzification == 'center_of_sums':
            if output_range is None:
                raise ValueError("output_range gereklidir")
            return self.defuzzify_center_of_sums(rule_activations, output_range)
        else:  # weighted_average
            return self.defuzzify_weighted_average(rule_activations)