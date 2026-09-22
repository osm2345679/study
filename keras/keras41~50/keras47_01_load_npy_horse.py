# 45-4 카피
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, GlobalAveragePooling2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import datetime
import time

#1. 데이터
np_path = './_data/npy_horse/'

x_train = np.load(np_path + 'keras46_01_x_train.npy')
y_train = np.load(np_path + 'keras46_01_y_train.npy')
x_test = np.load(np_path + 'keras46_01_x_test.npy')
y_test = np.load(np_path + 'keras46_01_y_test.npy')

# one-hot encoding
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1,1))
y_test = ohe.fit_transform(y_test.reshape(-1,1))

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(100,100,3)))  # (98,98,64)
model.add(Conv2D(64, (3,3), activation='relu'))  # (96,96,64)
model.add(Dropout(0.2))
model.add(MaxPooling2D())   # (48,48,64)
model.add(Conv2D(32, (3,3), activation='relu'))  # (46,46,32)
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3), activation='relu'))  # (44,44,32)
model.add(Dropout(0.2))
# model.add(Flatten())
model.add(GlobalAveragePooling2D()) # (1,1,32)
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='softmax'))

#3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True
)

path = './_save/keras47/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k47_1_", date, "-", filename])

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'min',
    verbose = 0,
    save_best_only = True,
    filepath = filepath
)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=3000,
    batch_size=32,
    verbose=1,
    validation_split=0.2,
    callbacks=[es, mcp]
)
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss[0])
print("acc : ", loss[1])

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test, y_pred)
print("acc : ", acc)

print("걸린 시간 : ", round(end_time-start_time, 2), "초")
# Results
# Epoch 108/3000
# 21/21 [==============================] - 1s 26ms/step - loss: 8.9148e-05 - acc: 1.0000 - val_loss: 0.0425 - val_acc: 0.9939
# 7/7 [==============================] - 0s 26ms/step - loss: 0.0118 - acc: 1.0000
# loss :  0.011795907281339169
# acc :  1.0
# 7/7 [==============================] - 0s 6ms/step
# acc :  1.0
# 걸린 시간 :  62.42 초