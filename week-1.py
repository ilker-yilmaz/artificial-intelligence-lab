# .csv formatında verilen dosyayı pandas kütüphanesi kullanarak açınız
import pandas as pd

df = pd.read_csv("datasets/kitap1.csv")

# Pandas kütüphanesi aracılığıyla tablodan “Sıcaklık” ve “Nem” değerlerini siliniz
#df = df.drop(["Sıcaklık", "Nem"], axis=1)

# Pandas kütüphanesinin metodu olan DataFrame() ile yukarıda verilen tabloyu oluşturunuz ve
# tablo hakkında betimleyici istatiksel bilgiler veriniz
df = pd.DataFrame(df)
print("df describe:\n",df.describe())

# (3,4) boyutunda bir dizi oluşturunuz.
# Oluşturduğunuz bu dizinin boyutunu (6,2) olacak şekilde değiştiriniz
import numpy as np
dizi = np.array([[1,2,3,4],[4,5,6,7],[7,8,9,10]])
print("dizi:\n",dizi)

yeni_dizi = dizi.reshape(6,2)
print("yeni_dizi:\n",yeni_dizi)

# İki tane (3,3) boyutunda rastgele sayılardan meydana bir dizi oluşturunuz. Oluşturulan bu diziyi
# hem yatay hem de dikey olacak şekilde istif (stack) ediniz

dizi1 = np.random.randint(0,10,(3,3))
dizi2 = np.random.randint(0,10,(3,3))

print("dizi1:\n",dizi1)
print("dizi2:\n",dizi2)

yatay = np.hstack((dizi1,dizi2))
dikey = np.vstack((dizi1,dizi2))

print("yatay:\n",yatay)
print("dikey:\n",dikey)

