import time
start = time.time()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## Sklearn kütüphanelerinin import edilmesi

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing

## Yapay Sinir Ağları için Keras kütüphanelerinin import edilmesi

from keras.models import Sequential
from keras.layers import merge
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers import Convolution2D, Convolution1D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

from google.colab import files
uploaded = files.upload()

## Train Veri Seti
parent_data = pd.read_csv('train.csv')
data = parent_data.copy()
data.pop('id')

## Test Veri Seti
test = pd.read_csv('test.csv')
testId = test.pop('id')

data.head()

## Sayısal hale dönüştürme işlemi yapıldı
species_label = data.pop('species')
species_label = LabelEncoder().fit(species_label).transform(species_label)
print(species_label.shape)

one_hot = to_categorical(species_label)
print(one_hot.shape)

preprocessed_train_data = preprocessing.MinMaxScaler().fit(data).transform(data)
preprocessed_train_data = StandardScaler().fit(data).transform(data)

print(preprocessed_train_data.shape)

## Eğitim setinden Test setine aynı dönüşümler yapıldı
test = preprocessing.MinMaxScaler().fit(test).transform(test)
test = StandardScaler().fit(test).transform(test)

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3,random_state=12345)
train_index, val_index = next(iter(sss.split(preprocessed_train_data, one_hot)))

x_train, x_val = preprocessed_train_data[train_index], preprocessed_train_data[val_index]
y_train, y_val = one_hot[train_index], one_hot[val_index]

print("x_train dim: ",x_train.shape)
print("x_val dim:   ",x_val.shape)

model = Sequential()

model.add(Dense(768,input_dim=192,  kernel_initializer='glorot_normal', activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(768, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(99, activation='softmax'))

model.summary()

## Adagrad, rmsprop, SGD, Adadelta, Adam, Adamax, Nadam

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics = ["accuracy"])

# Commented out IPython magic to ensure Python compatibility.
# %%time
# early_stopping = EarlyStopping(monitor='val_loss', patience=300)
#
# history = model.fit(x_train, y_train,batch_size=192,epochs=2500 ,verbose=1,
#                     validation_data=(x_val, y_val),callbacks=[early_stopping])

print('val_acc: ',max(history.history['val_accuracy']))
print('val_loss: ',min(history.history['val_loss']))
print('train_acc: ',max(history.history['accuracy']))
print('train_loss: ',min(history.history['loss']))

print()
print("train/val loss ratio: ", min(history.history['loss'])/min(history.history['val_loss']))

## Kayıp için geçmişi özetle
## Kaybı yineleme sayısı ile çizme
plt.semilogy(history.history['loss'])
plt.semilogy(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('model_loss.png')

## Hatayı yineleme sayısı ile çizme
## Her yinelemede hata sorunsuz şekilde çalışır
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('model_accuracy.png')

yPred = model.predict_on_batch(test)

## Converting the test predictions in a dataframe as depicted by sample submission
submission = pd.DataFrame(yPred,index=testId,columns=sort(parent_data.species.unique()))

submission.to_csv('leafClassificationSubmission.csv')

end = time.time()
print(round((end-start),2), "seconds")

submission.head()