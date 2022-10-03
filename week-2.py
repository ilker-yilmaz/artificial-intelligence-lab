# SORU - 1
# Kullanıcıdan 3 adet integer türünde değer alınız. Almış olduğunuz bu değerler bir üçgenin
# açılarını ifade edecektir. Bu açı değerlerine göre bu üçgenin dik, geniş ya da dar üçgen olup
# olmadığını belirleyen programı yazınız

print("################### SORU - 1 ###################")

# Kullanıcıdan 3 adet integer türünde değer alma
a = int(input("a: "))
b = int(input("b: "))
c = int(input("c: "))

# Bu açı değerlerine göre bu üçgenin dik, geniş ya da dar üçgen olup olmadığını belirleme

if a == 90 or b == 90 or c == 90:
    print("Bu üçgen bir dik üçgendir.")
elif a < 90 and b < 90 and c < 90:
    print("Bu üçgen bir dar üçgendir.")
elif a > 90 or b > 90 or c > 90:
    print("Bu üçgen bir geniş üçgendir.")
else:
    print("Bu üçgen bir üçgen değildir.")

print("################### SORU - 2 ###################")

# SORU - 2
# İçinde uzaylı olan bir oyun geliştirdiğinizi düşünün. uzaylı_rengi isminde bir değişken oluşturun
# ve bu değişken string türünde değerler alsın. Bu değişkene kırmızı, yeşil ya da sarı
# değerlerinden birini klavyeden veriniz. Eğer uzaylının rengi yeşilse “Tebrikler, yeşil uzaylıya ateş
# ettiğiniz için 5 puan kazandınız” şeklinde bir çıktı veriniz. Eğer rengi yeşil değilse "Tebrikler, yeşil
# olmayan uzaylıya ateş ettiğiniz için 10 puan kazandınız" şeklinde çıktı veriniz. Senaryoya ait
# programı yazınız

# string türünde değerler alan uzaylı_rengi değişkeni oluşturma
uzayli_rengi = input("Uzaylı rengi: ")

# Eğer uzaylının rengi yeşilse “Tebrikler, yeşil uzaylıya ateş ettiğiniz için 5 puan kazandınız” şeklinde bir çıktı verme
if uzayli_rengi == "yeşil":
    print("Tebrikler, yeşil uzaylıya ateş ettiğiniz için 5 puan kazandınız")
else:
    print("Tebrikler, yeşil olmayan uzaylıya ateş ettiğiniz için 10 puan kazandınız")

print("################### SORU - 3 ###################")

# SORU - 3
# a) Eğer uzaylı rengi yeşil ise "Tebrikler, yeşil uzaylıya ateş ettiğiniz için 5 puan kazandınız"
# b) Eğer uzaylı rengi sarı ise "Tebrikler, sarı uzaylıya ateş ettiğiniz için 10 puan kazandınız"
# c) Eğer uzaylı rengi kırmız ise "Tebrikler, kırmızı uzaylıya ateş ettiğiniz için 15 puan kazandınız"

if uzayli_rengi == "yeşil":
    print("Tebrikler, yeşil uzaylıya ateş ettiğiniz için 5 puan kazandınız")
elif uzayli_rengi == "sarı":
    print("Tebrikler, sarı uzaylıya ateş ettiğiniz için 10 puan kazandınız")
elif uzayli_rengi == "kırmızı":
    print("Tebrikler, kırmızı uzaylıya ateş ettiğiniz için 15 puan kazandınız")

print("################### SORU - 4 ###################")

# SORU - 4
# if-elif-else yapılarını kullanarak bir insanın yaşam evreleri ile ilgili programı oluşturunuz. int
# türünde, yaş isminde bir değişken oluşturup, bu değişken için gerekli olan değeri kullanıcıdan
# isteyiniz (20p).
# a) Eğer bir insanın yaşı 2 yaşından küçük ise, "Bu kişi bebektir",
# b) Eğer bir insanın yaşı 2 ile 4 arasındaysa (2 dâhil) "Bu kişi yeni yürümeye başlayan çocuktur",
# c) Eğer bir insanın yaşı 4 ile 13 arasındaysa (4 dâhil) "Bu kişi çocuktur",
# d) Eğer bir insanın yaşı 13 ile 20 arasındaysa (13 dâhil) "Bu kişi ergendir",
# e) Eğer bir insanın yaşı 20 ile 65 arasındaysa (20 dâhil) "Bu kişi yetişkindir",
# f) Eğer bir insanın yaşı 65 ve üstü ise (65 dâhil) "Bu kişi yaşlıdır" şeklinde çıktı veriniz.

# int türünde, yaş isminde bir değişken oluşturma
yas = int(input("Yaş: "))
# Yaşa göre if-elif-else yapısı
if yas < 2:
    print("Bu kişi bebektir")
elif yas < 4:
    print("Bu kişi yeni yürümeye başlayan çocuktur")
elif yas < 13:
    print("Bu kişi çocuktur")
elif yas < 20:
    print("Bu kişi ergendir")
elif yas < 65:
    print("Bu kişi yetişkindir")
else:
    print("Bu kişi yaşlıdır")

print("################### SORU - 5 ###################")

# SORU - 5
# Favori meyvelerinizin olduğu bir liste oluşturunuz ve bu listede 5 adet meyveniz bulunsun.
# Listenin adı favori_meyveler şeklinde tanımlansın. if-else yapısını kullanarak örnekte verilen
# meyvelerin favori listenizde olup olmadığını kontrol ediniz. Örnek meyveler; elma, armut,
# karpuz, kavun, muz, portakal, çilek, vişne, kiraz ve mandalina

# Favori meyvelerinizin olduğu bir liste oluşturma
favori_meyveler = ["elma", "armut", "karpuz", "kavun", "muz"]

# Örnek meyveler
meyveler = ["elma", "armut", "karpuz", "kavun", "muz", "portakal", "çilek", "vişne", "kiraz", "mandalina"]

# Örnek meyveler ile favori meyveler karşılaştırma
for meyve in meyveler:
    if meyve in favori_meyveler:
        print(meyve, "favori meyvelerimden biridir.")
    else:
        print(meyve, "favori meyvelerimden değildir.")

