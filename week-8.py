# yaprak sınıflandırması (1DESA)

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# test ve eğitim verilerinin okunması
train = pd.read_csv('datasets/leaf-classification/train.csv')
test = pd.read_csv('datasets/leaf-classification/test.csv')

# sınıfların belirlenmesi ve etiketlenmesi
label_encoder = LabelEncoder().fit(train.species)
labels = label_encoder.transform(train.species)
classes = list(label_encoder.classes_)

# verilerin hazırlanması, özellik ve sınıf sayısının belirlenmesi
train = train.drop(['id', 'species'], axis=1)
test = test.drop(['id'], axis=1)
nb_features = 192
nb_classes = len(classes)


# eğitim verisindeki verilerin standartlaştırılması
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(train.values)
train = scaler.transform(train.values)

# eğitim verisinin eğitim ve doğrulama için ayarlanması
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(train, labels, test_size=0.2)

# etiketlerin kategorilerinin belirlenmesi
#from tensorflow.keras.utils import to_categorical
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)

# giriş verilerinin boyutlarının ayarlanması
X_train = np.array(X_train).reshape(-1, nb_features, 1)
X_valid = np.array(X_valid).reshape(-1, nb_features, 1)

# 1DESA modelinin oluşturulması
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

#Modelin eğitilmesi
model.fit(X_train,y_train,epochs=15,validation_data=(X_valid,y_valid))


#Ortalama değerlerin gösterilmesi
print(("Ortalama eğitim kaybı:",np.mean(model.history.history["loss"])))
print(("Ortalama eğitim başarımı:",np.mean(model.history.history["accuracy"])))
print(("Ortalama doğrulama kaybı:",np.mean(model.history.history["val_loss"])))
print(("Ortalama doğrulama başarımı:",np.mean(model.history.history["val_accuracy"])))

#Değerlerin grafik üzerinde gösterilmesi
import matplotlib.pyplot as plt
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(15,15))
ax1.plot(model.history.history["loss"],color="g",label="Eğitimkaybı")
ax1.plot(model.history.history["val_loss"],color="y",label="Doğrulama kaybı")
ax1.set_xticks(np.arange(20,100,20))
ax2.plot(model.history.history["accuracy"],color="b",label="Eğitim başarımı")
ax2.plot(model.history.history["val_accuracy"],color="r",label="Doğrulama başarımı")
ax1.set_xticks(np.arange(20,100,20))
plt.legend()
plt.show()


# F1-skor, kesinlik (precision), duyarlılık (sensitivity) ve özgüllük (specificity) değerlerini bulunuz

#f1 skoru
from sklearn.metrics import f1_score
y_pred = model.predict(X_valid)
y_pred = np.argmax(y_pred, axis=1)
y_valid = np.argmax(y_valid, axis=1)
f1 = f1_score(y_valid, y_pred, average='macro')
print('F1 skoru: %f' % f1)

#kesinlik
from sklearn.metrics import precision_score
precision = precision_score(y_valid, y_pred, average='macro')
print('Kesinlik: %f' % precision)

#duyarlılık
from sklearn.metrics import recall_score
recall = recall_score(y_valid, y_pred, average='macro')
print('Duyarlılık: %f' % recall)

#özgüllük
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel()
specificity = tn / (tn+fp)
print('Özgüllük: %f' % specificity)





