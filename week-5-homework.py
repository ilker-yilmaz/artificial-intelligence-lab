# TKA Öğrenme Algoritması:
# 1.Adım: Ağa girdi setini ve ona karşılık olarak beklenen çıktı gösterilir (X,B).
# Burada birden fazla girdi değeri olabilir. Yani X= (X1, X2, X3,…, XN) demektir.
# Çıktı değeri ise 1 ve 0 değerlerinden birisini alır.
# 2.Adım: Perseptron ünitesine gelen net girdi şu şekilde hesaplanır:
# Net = Σ (Xi * Wi) + B
# Burada Xi, girdi değerlerini, Wi ise ağırlıkları, B ise bias değerini temsil eder.
# 3.Adım: Perseptron ünitesinin çıktısı hesaplanır. Net girdinin eşik (Φ)
# değerinden büyük veya küçük olmasına göre çıktı değeri 0 ve 1 değerlerinden
# birisini alır:
# Y = 1, Net > Φ
# Y = 0, Net < Φ
# Eğer gerçekleşen çıktı ile beklenen çıktı aynı olursa ağırlıklarda herhangi bir
# değişiklik olmaz. Ağ, beklenmeyen bir çıktı üretmiş ise o zaman iki durum söz
# konusudur:
# i. Ağın beklenen çıktısı 0 değeridir. Fakat NET girdi eşik değerinin
# üstündedir. Yani ağın gerçekleşen çıktısı 1 değeridir. Bu durumda ağırlık
# değerleri azaltılmaktadır. Ağırlıkların değişim oranı girdi değerlerinin
# belirli bir oranı kadardır. Yani;
# Wn = Wo-λX
# olur. Burada λ öğrenme katsayısıdır. Ağırlıkların değişim miktarlarını
# belirlemekte ve sabit bir değer olarak alınmaktadır
# ii. Beklenen çıktının 1 olması ve ağın gerçek çıktısının 0 olması durumudur.
# Yani NET girdi eşik değerinin altındadır. Bu durumda ağırlıkların değerinin
# artırılması gerekmektedir. Yani;
# Wn = Wo +λX
# olacaktır.
# 4.Adım: Yukarıdaki adımları bütün girdi setindeki örnekler için doğru sınıflandırmalar yapılıncaya kadar ilk üç adımdaki işlemler tekrarlanır
#
#

# Örnek: Elma ve Portakal örüntüsünü ayıran bir TKA ağı.
# 1. örnek : X1= (x1, x2)= (1,0); B1= 1 (Portakal)
# 2. örnek : X2= (x1,x2)= (0,1); B2=0 (Elma)
# W = (w1, w2)= (1,2), Ф= -1, λ =0.5 olarak seçilmiştir

# 1. iterasyon - 1. örnek ağa gösterilir:
# NET= w1*x1 +w2*x2 = 1*1+2*0= 1
# NET > Ф olduğundan gerçekleşen çıktı Ç= 1 olacaktır. Ç=B1 olduğundan
# ağırlıklar değiştirilmez.

# 2. iterasyon - 2. örnek ağa gösterilir:
# NET= w1*x1 +w2*x2 = 1*0+2*1= 2
# NET > Ф olduğundan gerçekleşen çıktı Ç= 1 ve Ç≠B2 olduğundan
# ağırlıklar:
# Wn = Wo- λX formülü kullanılarak değiştirilir. Yeni değerler:
# w1= w1 – λ.x1 =1-0.5*0=1
# w2= w2 – λ.x2 =2-0.5*1=1.5

# 3. iterasyon - 1. örnek tekrar gösterilir:
# NET= w1*x1 +w2*x2 = 1*1+1.5*0=1
# NET > Ф olduğundan gerçekleşen çıktı Ç = 1 ve Ç=B1 olduğundan
# ağırlıklar değiştirilmez

# 4. iterasyon - 2. örnek tekrar gösterilir:
# NET= w1*x1 +w2*x2 = 1*0+1.5*1=1.5
# NET > Ф olduğundan gerçekleşen çıktı Ç= 1 ve Ç≠B2 olduğundan
# ağırlıklar;
# Wn = Wo- λX
# formülü kullanılarak değiştirilir. Burada önceki iterasyonlarda değiştirilen
# ağırlıkların kullanıldığına dikkat ediniz;
# w1= w1 – λ.x1 = 1-0.5*0=1
# w2= w2 – λ.x2 =1.5-0.5* 1 = 1

# 5. iterasyon - 1. örnek tekrar gösterilir:
# NET= w1*x1 +w2*x2 = 1*1+1*0= 1
# NET > Ф olduğundan gerçekleşen çıktı Ç= 1 ve Ç=B1 olduğundan
# ağırlıklar değiştirilmez.

# 6. iterasyon - 2. örnek tekrar gösterilir:
# NET= 1*0+1*1= 1
# NET > Ф olduğundan Ç= 1 ve Ç≠B2 olduğundan ağırlıklar;
# w1= 1-0.5*0=1
# w2= 1-0.5*1=0.5

# 7. iterasyon - 1. örnek tekrar gösterilir:
# NET= 1*1+0.5*0= 1
# NET > Ф olduğundan Ç=1 ve Ç=B1 olduğundan ağırlıklar değiştirilmez.

# 8. iterasyon - 2. örnek tekrar gösterilir:
# NET= 1*0+0.5*1=0.5
# NET > Ф olduğundan Ç= 1 ve Ç≠B2 olduğundan ağırlıklar;
# w1= 1 - 0.5* 0 =1
# w2= 0.5 - 0.5*1 = 0

# 9. iterasyon - 1. örnek tekrar gösterilir:
# NET= 1*1+0*0=1
# NET > Ф olduğundan Ç=1 ve Ç=B1 olduğundan ağırlıklar değiştirilmez.

# 10. iterasyon - 2. örnek tekrar gösterilir:
# NET= 1*0+0.0*1= 0
# NET > Ф olduğundan Ç= 1 ve Ç≠B2 olduğundan ağırlıklar;
# w1= 1-0.5*0=1
# w2= 0-0.5*1=-0.5

# 11. iterasyon - 1. örnek tekrar gösterilir:
# NET = 1*1+ (-0.5)*0= 1
# NET > Ф olduğundan Ç=1 ve Ç=B1 olduğundan ağırlıklar değiştirilmez.

# 12. iterasyon - 2. örnek tekrar gösterilir:
# NET= 1*0+(-0.5)*1= -0.5.
# NET > Ф olduğundan Ç= 1 ve Ç≠B2 olduğundan ağırlıklar;
# w1= 1-0.5*0=1
# w2=-0.5-0.5*1=-1

# 13. iterasyon - 1. örnek tekrar gösterilir:
# NET= 1*1+ (-1)*0= 1
# NET > Ф olduğundan Ç=1 ve Ç=B1 olduğundan ağırlıklar değiştirilmez.

# 14. iterasyon - 2. örnek tekrar gösterilir:
# NET = 1*0+(-1)*1= -1
# NET = Ф olduğundan Ç= 0 ve Ç=B2 olduğundan ağırlıklar değiştirilmez.
# Bundan sonra her iki örnekte doğru olarak sınıflandırılır. Öğrenme
# sonunda ağırlıklar:
# w1 = 1 ve w2 = -1
# değerlerini alınca örnekler doğru sınıflandırılabilir demektir. Bu ağırlık
# değerleri kullanılarak, iki örnek tekrar ağa tekrar gösterilirse, ağın çıktıları
# şöyle olur

# 1. örnek için: NET= w1*x1 +w2*x2 = 1*1+ (-1)*0= 1 > Ф ve Ç=B1=1 olur.
# 2. örnek için: NET= w1*x1 +w2*x2 = 1*0+(-1)*1= -1= Ф ve Ç= B2=0 olur.
# Görüldüğü gibi her iki örnek içinde ağ tarafından doğru sınıflandırma
# yapılmaktadır. O nedenle, ağ öğrenmeyi tamamlamıştır denilebilir

# elma ile portakal TKA ağı

# Aşağıdaki örnekte, elma ve portakal sınıflarının ayrılması için TKA ağı
# kullanılmıştır. Ağın öğrenme süreci aşağıdaki gibi gerçekleşmiştir. python kodunu yazalım:


# Girdi değerleri:

# çözüm - 1
# elma = [1, 0]
# portakal = [0, 1]
# # Ağırlık değerleri:
# w1 = 1
# w2 = 2
# # Ağın öğrenme katsayısı:
# λ = 0.5
# # Ağın eşik değeri:
# Ф = -1
# # Ağın iterasyon sayısı:
# iterasyon = 0
# # Ağın öğrenme süreci:
# while iterasyon<14:
#     iterasyon += 1
#     print("{}. iterasyon".format(iterasyon))
#     print("elma için: {}".format(elma))
#     net = w1 * elma[0] + w2 * elma[1]
#     if net > Ф:
#         c = 1
#     else:
#         c = 0
#     if c != elma[0]:
#         w1 = w1 - λ * elma[0]
#         w2 = w2 - λ * elma[1]
#         print("elma için ağırlıklar değiştirildi")
#     else:
#         print("elma için ağırlıklar değiştirilmedi")
#     print("portakal için: {}".format(portakal))
#     net = w1 * portakal[0] + w2 * portakal[1]
#     if net > Ф:
#         c = 1
#     else:
#         c = 0
#     if c != portakal[1]:
#         w1 = w1 - λ * portakal[0]
#         w2 = w2 - λ * portakal[1]
#         print("portakal için ağırlıklar değiştirildi")
#     else:
#         print("portakal için ağırlıklar değiştirilmedi")
#     if c == elma[0] and c == portakal[1]:
#         print("ağ öğrenmeyi tamamladı")
#         break
#     else:
#         print("ağ öğrenmeyi tamamlamadı")
#     print("ağın ağırlıkları: w1 = {} ve w2 = {}".format(w1, w2))
#     print("")

# Programın sonuçları şöyle olur:
# 1. iterasyon
# elma için: [1, 0]
# elma için ağırlıklar değiştirildi
# portakal için: [0, 1]
# portakal için ağırlıklar değiştirildi
# ağ öğrenmeyi tamamlamadı
# ağın ağırlıkları: w1 = 0.5 ve w2 = 0.5


# çözüm - 2: 

elma = [1, 0]
B1 = 0
portakal = [0, 1]
B2 = 1

w1 = 1
w2 = 2

λ = 0.5
Ф = -1

iterasyon = 0

while iterasyon<14:
    iterasyon += 1
    print("{}. iterasyon".format(iterasyon))
    print("elma için: {}".format(elma))
    net = w1 * elma[0] + w2 * elma[1]
    if net > Ф:
        c = 1
    else:
        c = 0
    if c != B1:
        w1 = w1 - λ * elma[0]
        w2 = w2 - λ * elma[1]
        print("elma için ağırlıklar değiştirildi")
    else:
        print("elma için ağırlıklar değiştirilmedi")
    print("portakal için: {}".format(portakal))
    net = w1 * portakal[0] + w2 * portakal[1]
    if net > Ф:
        c = 1
    else:
        c = 0
    if c != B2:
        w1 = w1 - λ * portakal[0]
        w2 = w2 - λ * portakal[1]
        print("portakal için ağırlıklar değiştirildi")
    else:
        print("portakal için ağırlıklar değiştirilmedi")
    if c == B1 and c == B2:
        print("ağ öğrenmeyi tamamladı")
        break
    else:
        print("ağ öğrenmeyi tamamlamadı")
    print("ağın ağırlıkları: w1 = {} ve w2 = {}".format(w1, w2))
    print("")

# Programın sonuçları şöyle olur:

