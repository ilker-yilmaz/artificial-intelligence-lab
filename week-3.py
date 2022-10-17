# print("######################### SORU - 1 #########################")
# # Kullanıcıdan integer türünde bir değer isteyiniz. İstemiş olduğunuz bu değerin çarpım tablosu
# # değerlerini gösteren kodu for döngüsü ile gerçekleştiriniz
# # Örnek: Kullanıcı 5 değerini girerse, çıktı olarak 5 10 15 20 … … 50 (10’a kadar)
#
# # Kullanıcıdan integer türünde bir değer isteme
# sayi = int(input("Sayı: "))
# # İstemiş olduğunuz bu değerin çarpım tablosu değerlerini gösteren kodu for döngüsü ile gerçekleştirme
# for i in range(1,11):
#     print("{} x {} = {}".format(sayi,i,sayi*i))
#
# print("######################### SORU - 2 #########################")
# # Girilen bir sayının kaç basamaklı olduğunu belirleyen programı while döngüsü ile gerçekleştiriniz
#
# # Girilen bir sayının kaç basamaklı olduğunu belirleme
# sayi = int(input("Sayı: "))
# basamak = 0
# while sayi > 0:
#     sayi = sayi // 10
#     basamak += 1
# print("Basamak sayısı: {}".format(basamak))
#
# print("######################### SORU - 3 #########################")
# # Aşağıda bir listeye ait sayısal değerler verilmiştir.
# # sayısalDeğerler = [12, 15, 32, 42, 55, 75, 122, 132, 150, 180, 200]
# # Bu listedeki 5’e bölünen sayıları çıktı olarak veren programı hem for hem de while döngüsü ile
# # gerçekleştiriniz. 150’den büyük değerleri dikkate almayınız (20p).
# # Çıktı: 15, 55, 75, 150
#
# # Aşağıda bir listeye ait sayısal değerler verilmiştir.
# sayısalDeğerler = [12, 15, 32, 42, 55, 75, 122, 132, 150, 180, 200]
#
# # for döngüsü ile gerçekleştirme
# for i in sayısalDeğerler:
#     if i > 150:
#         break
#     if i % 5 == 0:
#         print(i, end=", ")
# print()
# # while döngüsü ile gerçekleştirme
# i = 0
# while i < len(sayısalDeğerler):
#     if sayısalDeğerler[i] > 150:
#         break
#     if sayısalDeğerler[i] % 5 == 0:
#         print(sayısalDeğerler[i], end=", ")
#     i += 1
# print()
#
# print("######################### SORU - 4 #########################")
# # Kullanıcıdan 3 adet (a, b ve c) değer alınız. a (dahil) ve b (dahil) arasında kaç sayının c’ye
# # bölünebildiğini belirleyen programı yazınız (20p).
# # Örnek: a = 20, b = 40, c = 5 ise Çıktı: 5
#
# # Kullanıcıdan 3 adet (a, b ve c) değer alma
# a = int(input("a: "))
# b = int(input("b: "))
# c = int(input("c: "))
#
# # a (dahil) ve b (dahil) arasında kaç sayının c’ye bölünebildiğini belirleme
# sayac = 0
# for i in range(a,b+1):
#     if i % c == 0:
#         sayac += 1
# print("Bölünebilen sayı sayısı: {}".format(sayac))
#
# print("######################### SORU - 5 #########################")
# # Aşağıdaki çıktıyı veren programı yazınız (10p).
# # 1 – 99
# # 2 – 98
# # 3 – 97
# # ..
# # ..
# # ..
# # 98 – 2
# # 99 – 1
#
# # Aşağıdaki çıktıyı veren programı yazma
# for i in range(1,100):
#     print("{} - {}".format(i,100-i))

print("######################### SORU - 6 #########################")
# Kullanıcıdan bir IP adresi isteyiniz. İstediğiniz bu IP adresinden sonraki 5 değeri çıktı olarak
# veren programı yazınız (30p).
# Örnek: 192 168 255 252
# Çıktı: 192 168 255 253
# 192 168 255 254
# 192 168 255 255
# 192 169 0 0
# 192 169 0 1

# Kullanıcıdan bir IP adresi isteme
ip = input("IP: ")
# İstediğiniz bu IP adresinden sonraki 5 değeri çıktı olarak verme
ip = ip.split(".")
ip = [int(i) for i in ip]
for i in range(5):
    ip[3] += 1
    if ip[3] == 256:
        ip[3] = 0
        ip[2] += 1
    if ip[2] == 256:
        ip[2] = 0
        ip[1] += 1
    if ip[1] == 256:
        ip[1] = 0
        ip[0] += 1
    print("{}.{}.{}.{}".format(ip[0],ip[1],ip[2],ip[3]))

