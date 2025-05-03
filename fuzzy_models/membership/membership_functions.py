import numpy as np
import matplotlib.pyplot as plt

def triangular_mf(x, a, b, c):
    """Üçgen üyelik fonksiyonu
    
    Args:
        x: Değerlendirilen girdi değeri
        a: Sol uç
        b: Tepe nokta
        c: Sağ uç
    
    Returns:
        float: Üyelik derecesi [0, 1]
    """
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    else:  # b < x < c
        return (c - x) / (c - b)

def gaussian_mf(x, mean, sigma):
    """Gauss üyelik fonksiyonu
    
    Args:
        x: Değerlendirilen girdi değeri
        mean: Ortalama (tepe noktası)
        sigma: Standart sapma (genişlik)
    
    Returns:
        float: Üyelik derecesi [0, 1]
    """
    return np.exp(-((x - mean)**2) / (2 * sigma**2))

def generate_triangular_mfs(data_range, num_sets):
    """Verilen aralık için üçgen üyelik fonksiyonları oluşturur
    
    Args:
        data_range: (min, max) değerleri
        num_sets: Oluşturulacak bulanık küme sayısı
        
    Returns:
        list: (a, b, c) parametrelerini içeren üçgen üyelik fonk. listesi
    """
    min_val, max_val = data_range
    width = (max_val - min_val) / (num_sets - 1)
    
    mfs = []
    for i in range(num_sets):
        # Üçgen üyelik fonksiyonu parametreleri
        a = max(min_val, min_val + (i - 1) * width)
        b = min_val + i * width
        c = min(max_val, min_val + (i + 1) * width)
        mfs.append((a, b, c))
    
    return mfs

def generate_gaussian_mfs(data_range, num_sets):
    """Verilen aralık için Gauss üyelik fonksiyonları oluşturur
    
    Args:
        data_range: (min, max) değerleri
        num_sets: Oluşturulacak bulanık küme sayısı
        
    Returns:
        list: (mean, sigma) parametrelerini içeren Gauss üyelik fonk. listesi
    """
    min_val, max_val = data_range
    width = (max_val - min_val) / (num_sets - 1)
    
    mfs = []
    for i in range(num_sets):
        mean = min_val + i * width
        sigma = width / 2.5  # Sigma değeri örtüşmeyi kontrol eder
        mfs.append((mean, sigma))
    
    return mfs

def plot_membership_functions(mf_type, mfs, data_range, labels, title, filename):
    """Üyelik fonksiyonlarını görselleştirir
    
    Args:
        mf_type: 'triangular' veya 'gaussian'
        mfs: Üyelik fonksiyonu parametrelerinin listesi
        data_range: (min, max) veri aralığı
        labels: Bulanık küme etiketleri
        title: Grafik başlığı
        filename: Kaydedilecek dosya adı
    """
    min_val, max_val = data_range
    x = np.linspace(min_val, max_val, 1000)
    
    plt.figure(figsize=(10, 6))
    
    for i, params in enumerate(mfs):
        if mf_type == 'triangular':
            a, b, c = params
            y = [triangular_mf(val, a, b, c) for val in x]
        else:  # gaussian
            mean, sigma = params
            y = [gaussian_mf(val, mean, sigma) for val in x]
        
        plt.plot(x, y, label=labels[i])
    
    plt.title(title)
    plt.xlabel('Değer')
    plt.ylabel('Üyelik Derecesi')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/{filename}.png')
    plt.close()