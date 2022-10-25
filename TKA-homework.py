
elma = [1, 0] # 1. örnek
B1 = 0 # 1. örnek için beklenen çıktı
portakal = [0, 1] # 2. örnek
B2 = 1 # 2. örnek için beklenen çıktı

w1 = 1  # 1. ağırlık
w2 = 2 # 2. ağırlık

λ = 0.5 # öğrenme katsayısı
Ф = -1 # eşik değeri

iterasyon = 0 # iterasyon sayısı

while iterasyon<14: # iterasyon sayısı 14'den küçük olduğu sürece
    iterasyon += 1 # iterasyon sayısını 1 artır
    print("{}. iterasyon".format(iterasyon)) # iterasyon sayısını yazdır
    print("elma için: {}".format(elma)) # 1. örnek için girdileri yazdır
    net = w1 * elma[0] + w2 * elma[1] # 1. örnek için net değerini hesapla
    if net > Ф: # net değeri eşik değerinden büyükse
        c = 1 # çıktıyı 1 yap
    else: # net değeri eşik değerinden küçükse
        c = 0 # çıktıyı 0 yap
    if c != B1: # çıktı beklenen çıktıdan farklıysa
        w1 = w1 - λ * elma[0] # 1. ağırlığı güncelle
        w2 = w2 - λ * elma[1] # 2. ağırlığı güncelle
        print("elma için ağırlıklar değiştirildi") # ağırlıkların değiştirildiğini yazdır
    else: # çıktı beklenen çıktıya eşitse
        print("elma için ağırlıklar değiştirilmedi") # ağırlıkların değiştirilmediğini yazdır
    print("portakal için: {}".format(portakal)) # 2. örnek için girdileri yazdır
    net = w1 * portakal[0] + w2 * portakal[1] # 2. örnek için net değerini hesapla
    if net > Ф: # net değeri eşik değerinden büyükse
        c = 1 # çıktıyı 1 yap
    else: # net değeri eşik değerinden küçükse
        c = 0 # çıktıyı 0 yap
    if c != B2: # çıktı beklenen çıktıdan farklıysa
        w1 = w1 - λ * portakal[0] # 1. ağırlığı güncelle
        w2 = w2 - λ * portakal[1] # 2. ağırlığı güncelle
        print("portakal için ağırlıklar değiştirildi") # ağırlıkların değiştirildiğini yazdır
    else: print("portakal için ağırlıklar değiştirilmedi") # ağırlıkların değiştirilmediğini yazdır
    if c == B1 and c == B2: # çıktılar beklenen çıktılara eşitse
        print("ağ öğrenmeyi tamamladı") # öğrenmeyi tamamladığını yazdır
        break # döngüyü sonlandır
    else: # çıktılar beklenen çıktılara eşit değilse
        print("ağ öğrenmeyi tamamlamadı") # öğrenmeyi tamamlamadığını yazdır
    print("ağın ağırlıkları: w1 = {} ve w2 = {}".format(w1, w2)) # ağırlıkları yazdır
    print("")