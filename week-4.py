# Örnek Uygulama (YSA ile cep telefonu fiyatlarının artışının belirlenmesi)

# Kütüphanelerin yüklenmesi
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Veri setinin yüklenmesi
veriler = pd.read_csv("datasets/mobile-price/train.csv")

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

# Verilen kodu inceleyerek kendi modelinizi oluşturunuz

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential() # modelin oluşturulması
model.add(Dense(16, input_dim=20, activation="relu"))
model.add(Dense(12, activation="relu")) # gizli katman
model.add(Dense(4, activation="softmax")) # çıktı katmanı
model.summary() # modelin özetini gösterir

# Verilen kodu inceleyerek çapraz doğrulama işlemi yapınız ve buna göre başarımı değerlendiriniz

from sklearn.model_selection import cross_val_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def build_classifier():
    model = Sequential() # modelin oluşturulması
    model.add(Dense(16, input_dim=20, activation="relu"))
    model.add(Dense(12, activation="relu")) # gizli katman
    model.add(Dense(4, activation="softmax")) # çıktı katmanı
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]) # modelin derlenmesi
    return model # modelin döndürülmesi

classifier = KerasClassifier(build_fn=build_classifier, epochs=150) # modelin oluşturulması
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10) # çapraz doğrulama işlemi
print("Average Accuracy: ", accuracies.mean()) # ortalama başarımların gösterilmesi
print("Standard Deviation: ", accuracies.std()) # standart sapmaların gösterilmesi

# Verisetinde (telefon_fiyat_değişimi) toplam 20 adet özellik bulunmaktadır. Bu verisetinden
# “blue”, “fc”, “int_memory”, “ram” ve “wifi” değerlerini çıkarıp, sınıflandırma işlemini tekrar
# yapınız

veriler = pd.read_csv("telefon_fiyat_değişimi.csv") # verilerin okunması
veriler = veriler.drop(["blue", "fc", "int_memory", "ram", "wifi"], axis=1) # verilerden özelliklerin çıkarılması
X = veriler.drop(["price_range"], axis=1) # bağımsız değişkenlerin oluşturulması
y = veriler["price_range"] # bağımlı değişkenin oluşturulması

# Diyabet verisetini kullanarak bir YSA modeli oluşturunuz. Bu YSA modeline, eğitim ve doğrulama
# işlemlerinin başarım ve kayıplarını belirleyiniz. Bu değerleri bir grafik üzerinde gösteriniz

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

veriler = pd.read_csv("datasets/diabetes.csv") # verilerin okunması
X = veriler.drop(["Outcome"], axis=1) # bağımsız değişkenlerin oluşturulması
y = veriler["Outcome"] # bağımlı değişkenin oluşturulması

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) # verilerin eğitim ve test olarak bölünmesi

sc = StandardScaler() # verilerin ölçeklendirilmesi
X_train = sc.fit_transform(X_train)  # eğitim verilerinin ölçeklendirilmesi
X_test = sc.transform(X_test) # verilerin ölçeklendirilmesi

model = Sequential() # modelin oluşturulması
model.add(Dense(12, input_dim=8, activation="relu")) # giriş katmanı
model.add(Dense(8, activation="relu")) # gizli katman
model.add(Dense(1, activation="sigmoid")) # çıktı katmanı
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]) # modelin derlenmesi
model.fit(X_train, y_train, epochs=150, batch_size=10, validation_data=(X_test, y_test)) # modelin eğitilmesi

plt.plot(model.history.history["accuracy"]) # eğitim başarımının gösterilmesi
plt.plot(model.history.history["val_accuracy"]) # doğrulama başarımının gösterilmesi
plt.title("Model Başarımları") # başlık
plt.ylabel("Başarımlar") # y ekseninin adı
plt.xlabel("Epok Sayısı") # x ekseninin adı
plt.legend(["Eğitim", "Test"], loc="upper left") # gösterilecek değerlerin belirlenmesi
plt.show() # grafik gösterimi

plt.plot(model.history.history["loss"]) # eğitim kaybının gösterilmesi
plt.plot(model.history.history["val_loss"]) # doğrulama kaybının gösterilmesi
plt.title("Model Kayıpları") # başlık
plt.ylabel("Kayıplar") # y ekseninin adı
plt.xlabel("Epok Sayısı") # x ekseninin adı
plt.legend(["Eğitim", "Test"], loc="upper left") # gösterilecek değerlerin belirlenmesi
plt.show() # grafik gösterimi

# Soru 4’te oluşturduğunuz modelin ROC eğrisini çiziniz

from sklearn.metrics import roc_curve
y_pred = model.predict(X_test) # modelin tahmin etmesi
fpr, tpr, thresholds = roc_curve(y_test, y_pred) # ROC eğrisinin oluşturulması
plt.plot(fpr, tpr) # ROC eğrisinin gösterilmesi
plt.plot([0, 1], [0, 1], linestyle="--") # eğriyi çizdirmek için
plt.xlabel("False Positive Rate") # x ekseninin adı
plt.ylabel("True Positive Rate") # y ekseninin adı
plt.title("ROC Curve") # başlık
plt.show() # grafik gösterimi


