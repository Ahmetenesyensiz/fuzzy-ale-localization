def generate_rule_base(num_anchor_sets, num_range_sets, num_density_sets, num_iteration_sets, num_ale_sets):
    """Bulanık kural tabanını oluşturur
    
    Args:
        num_anchor_sets: Çapa oranı için bulanık küme sayısı
        num_range_sets: İletim aralığı için bulanık küme sayısı
        num_density_sets: Düğüm yoğunluğu için bulanık küme sayısı
        num_iteration_sets: Yineleme sayısı için bulanık küme sayısı
        num_ale_sets: ALE için bulanık küme sayısı
        
    Returns:
        list: Kural listesi
    """
    # Bulanık küme etiketleri
    anchor_labels = ['Very_Low', 'Low', 'Medium', 'High', 'Very_High'][:num_anchor_sets]
    range_labels = ['Very_Low', 'Low', 'Medium', 'High', 'Very_High'][:num_range_sets]
    density_labels = ['Very_Low', 'Low', 'Medium', 'High', 'Very_High'][:num_density_sets]
    iteration_labels = ['Very_Low', 'Low', 'Medium', 'High', 'Very_High'][:num_iteration_sets]
    ale_labels = ['Very_Low', 'Low', 'Medium', 'High', 'Very_High'][:num_ale_sets]
    
    # Manuel olarak uzman bilgisine dayalı kurallar oluşturabiliriz
    # Örnek: Düşük çapa oranı ve düşük iletim aralığı -> Yüksek ALE
    
    rules = []
    
    # Örnek kurallar - gerçek veriye göre iyileştirilmeli
    # 1. Düşük çapa oranı -> Yüksek ALE
    rules.append((anchor_labels[0], None, None, None, ale_labels[-1]))
    
    # 2. Yüksek çapa oranı -> Düşük ALE
    rules.append((anchor_labels[-1], None, None, None, ale_labels[0]))
    
    # 3. Düşük iletim aralığı -> Yüksek ALE
    rules.append((None, range_labels[0], None, None, ale_labels[-1]))
    
    # 4. Yüksek iletim aralığı -> Düşük ALE
    rules.append((None, range_labels[-1], None, None, ale_labels[0]))
    
    # 5. Yüksek düğüm yoğunluğu ve düşük çapa oranı -> Yüksek ALE
    rules.append((anchor_labels[0], None, density_labels[-1], None, ale_labels[-1]))
    
    # 6. Yüksek düğüm yoğunluğu ve yüksek çapa oranı -> Düşük ALE
    rules.append((anchor_labels[-1], None, density_labels[-1], None, ale_labels[0]))
    
    # 7. Düşük yineleme sayısı -> Yüksek ALE
    rules.append((None, None, None, iteration_labels[0], ale_labels[-1]))
    
    # 8. Yüksek yineleme sayısı -> Orta ALE
    rules.append((None, None, None, iteration_labels[-1], ale_labels[2]))
    
    # Veri analizi sonuçlarına göre daha fazla kural eklenebilir
    
    return rules

def extract_rules_from_data(data, num_clusters=5):
    """Verideki ilişkileri analiz ederek otomatik kural çıkarma
    
    Args:
        data: Veri çerçevesi
        num_clusters: Kümeleme sayısı
        
    Returns:
        list: Otomatik oluşturulan kurallar
    """
    from sklearn.cluster import KMeans
    
    # Veriyi kümeleme
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(data)
    
    rules = []
    for cluster_id in range(num_clusters):
        cluster_data = data[data['cluster'] == cluster_id]
        
        # Her küme için ortalama değerleri al
        anchor_avg = cluster_data['anchor_ratio'].mean()
        range_avg = cluster_data['transmission_range'].mean()
        density_avg = cluster_data['node_density'].mean()
        iteration_avg = cluster_data['iteration_count'].mean()
        ale_avg = cluster_data['ale'].mean()
        
        # Ortalamalara göre dilsel değişkenleri belirle
        # Bu kısım veriye göre özelleştirilmeli
        
        # Örneğin:
        anchor_idx = min(int(anchor_avg * 5), 4)  # 0-4 arası indeks
        range_idx = min(int(range_avg * 5), 4)
        density_idx = min(int(density_avg * 5), 4)
        iteration_idx = min(int(iteration_avg * 5), 4)
        ale_idx = min(int(ale_avg * 5), 4)
        
        # Bulanık küme etiketleri
        anchor_labels = ['Very_Low', 'Low', 'Medium', 'High', 'Very_High']
        range_labels = ['Very_Low', 'Low', 'Medium', 'High', 'Very_High']
        density_labels = ['Very_Low', 'Low', 'Medium', 'High', 'Very_High']
        iteration_labels = ['Very_Low', 'Low', 'Medium', 'High', 'Very_High']
        ale_labels = ['Very_Low', 'Low', 'Medium', 'High', 'Very_High']
        
        # Kural oluşturma
        rule = (
            anchor_labels[anchor_idx],
            range_labels[range_idx],
            density_labels[density_idx],
            iteration_labels[iteration_idx],
            ale_labels[ale_idx]
        )
        
        rules.append(rule)
    
    return rules