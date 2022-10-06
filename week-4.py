# Örnek Uygulama (YSA ile cep telefonu fiyatlarının artışının belirlenmesi)

# Kütüphanelerin yüklenmesi
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Veri setinin yüklenmesi
veriler = pd.read_csv("datasets/telefon_fiyatlari.csv")

# sınıf sayısının belirlenmesi
label_encoder = LabelEncoder().fit(veriler.price_range)
labels = label_encoder.transform(veriler.price_range)
classes = list(label_encoder.classes_)

# girdi ve çıktı verilerinin hazırlanması
X = veriler.drop(["price_range"], axis=1)
y = labels

# verilerin standartlaştırılması
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# verilerin eğitim ve test olarak bölünmesi
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# çıktı değerlerinin kategorleştirilmesi
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# modelin oluşturulması
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(16, input_dim=20, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(4, activation="softmax"))
model.summary()


# modelin derlenmesi
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# modelin eğitilmesi
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150)

# eğitim ve doğrulama başarımlarının gösterilmesi
import matplotlib.pyplot as plt

plt.plot(model.history.history["accuracy"])
plt.plot(model.history.history["val_accuracy"])
plt.title("Model Başarımları")
plt.ylabel("Başarımlar")
plt.xlabel("Epok Sayısı")
plt.legend(["Eğitim", "Test"], loc="upper left")
plt.show()

plt.plot(model.history.history["loss"])
plt.plot(model.history.history["loss"])
plt.title("Model Kayıpları")
plt.ylabel("Kayıp")
plt.xlabel("Epok Sayısı")
plt.legend(["Eğitim", "Test"], loc="upper left")
plt.show()

# 1) Verilen kodu inceleyerek kendi modelinizi oluşturunuz (10p).
# 2) Verilen kodu inceleyerek çapraz doğrulama işlemi yapınız ve buna göre başarımı değerlendiriniz
# (30p).
# 3) Verisetinde (telefon_fiyat_değişimi) toplam 20 adet özellik bulunmaktadır. Bu verisetinden
# “blue”, “fc”, “int_memory”, “ram” ve “wifi” değerlerini çıkarıp, sınıflandırma işlemini tekrar
# yapınız (10p).
# 4) Diyabet verisetini kullanarak bir YSA modeli oluşturunuz. Bu YSA modeline, eğitim ve doğrulama
# işlemlerinin başarım ve kayıplarını belirleyiniz. Bu değerleri bir grafik üzerinde gösteriniz (20p).
# 5) Soru 4’te oluşturduğunuz modelin ROC eğrisini çiziniz (30p).