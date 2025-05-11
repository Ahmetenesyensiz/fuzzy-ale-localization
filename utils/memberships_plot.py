import os
import numpy as np
import matplotlib.pyplot as plt
from fuzzy_models.membership.triangular import triangular_membership
from fuzzy_models.membership.gaussian import gaussian_membership

def plot_membership_functions():
    save_path = "results/membership_functions.png"
    
    # Eğer dosya zaten varsa çizme
    if os.path.exists(save_path):
        print(f"{save_path} zaten mevcut. Çizim yapılmadı.")
        return

    x = np.linspace(0, 1, 500)

    # Triangular üyelik fonksiyonları
    tri_low = [triangular_membership(val, 0.0, 0.0, 0.5) for val in x]
    tri_medium = [triangular_membership(val, 0.0, 0.5, 1.0) for val in x]
    tri_high = [triangular_membership(val, 0.5, 1.0, 1.0) for val in x]

    # Gaussian üyelik fonksiyonları
    gau_low = [gaussian_membership(val, 0.0, 0.1) for val in x]
    gau_medium = [gaussian_membership(val, 0.5, 0.1) for val in x]
    gau_high = [gaussian_membership(val, 1.0, 0.1) for val in x]

    plt.figure(figsize=(10, 6))

    # Triangular plot
    plt.subplot(2, 1, 1)
    plt.plot(x, tri_low, label="Low")
    plt.plot(x, tri_medium, label="Medium")
    plt.plot(x, tri_high, label="High")
    plt.title("Triangular Membership Functions")
    plt.legend()
    plt.grid(True)

    # Gaussian plot
    plt.subplot(2, 1, 2)
    plt.plot(x, gau_low, label="Low")
    plt.plot(x, gau_medium, label="Medium")
    plt.plot(x, gau_high, label="High")
    plt.title("Gaussian Membership Functions")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"{save_path} dosyası başarıyla kaydedildi.")

if __name__ == "__main__":
    plot_membership_functions()
