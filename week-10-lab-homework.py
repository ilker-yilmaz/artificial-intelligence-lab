# SORU - 2
# Heart verisetine ait verileri kullanarak bir ESA mimarisi geliştiriniz.
# Geliştirmiş olduğunuz modelin doğrulama (accuracy), F1-skor ve kayıp değerlerini bulunuz.
# Bulduğunuz bu değerleri grafik üzerinde gösteriniz.
# Bunlara ek olarak tahmin işlemi gerçekleştiriniz

import pandas as pd
import numpy as np

# heart veriseti
heart = pd.read_csv("datasets/heart.csv")

nb_classes = 2
nb_features = 13
labels = heart["target"].values

# ESA mimarisi
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, Activation

model=Sequential()
model.add(Conv1D(512,1,input_shape=(nb_features,1)))
model.add(Activation("relu"))
model.add(MaxPooling1D(2))
model.add(Conv1D(256,1))
model.add(MaxPooling1D(2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(2048,activation="relu"))
model.add(Dense(1024,activation="relu"))
model.add(Dense(nb_classes,activation="softmax"))
model.summary()

#Ağın derlenmesi
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

# verilerin hazırlanması, özellik ve sınıf sayısının belirlenmesi
heart = heart.drop(['target'], axis=1)
nb_features = 13
nb_classes = 2

# eğitim verisindeki verilerin standartlaştırılması
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(heart.values)
heart = scaler.transform(heart.values)

# eğitim verisinin eğitim ve doğrulama için ayarlanması
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(heart, labels, test_size=0.2)

# etiketlerin kategorilerinin belirlenmesi
#from tensorflow.keras.utils import to_categorical
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid, num_classes=2)

# giriş verilerinin boyutlarının ayarlanması
X_train = X_train.reshape(X_train.shape[0], nb_features, 1)
X_valid = X_valid.reshape(X_valid.shape[0], nb_features, 1)

# eğitim işlemi
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_valid, y_valid))

# eğitim ve doğrulama verilerinin kayıp değerlerinin görselleştirilmesi
import matplotlib.pyplot as plt
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="valid loss")
plt.legend()
plt.show()

# eğitim ve doğrulama verilerinin doğrulama değerlerinin görselleştirilmesi
plt.plot(history.history["accuracy"], label="train accuracy")
plt.plot(history.history["val_accuracy"], label="valid accuracy")
plt.legend()
plt.show()

# modelin değerlendirilmesi
from sklearn.metrics import classification_report, confusion_matrix
y_pred = model.predict(X_valid)
y_pred = np.argmax(y_pred, axis=1)
y_valid = np.argmax(y_valid, axis=1)
print(classification_report(y_valid, y_pred))
print(confusion_matrix(y_valid, y_pred))

# tahmin işlemi
y_pred = model.predict(X_valid)
y_pred = np.argmax(y_pred, axis=1)
y_valid = np.argmax(y_valid, axis=1)

# tahmin edilen değerlerin görselleştirilmesi
plt.plot(y_pred, label="y_pred")
plt.plot(y_valid, label="y_valid")
plt.legend()
plt.show()
