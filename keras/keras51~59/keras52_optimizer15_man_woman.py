# 44-3 카피
# 데이터셋 - https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, GlobalAveragePooling2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import datetime
import time

#1. 데이터
np_path = './_data/man_woman/'
x_train = np.load(np_path + 'keras46_03_x_train.npy')
x_test = np.load(np_path + 'keras46_03_x_test.npy')
y_train = np.load(np_path + 'keras46_03_y_train.npy')
y_test = np.load(np_path + 'keras46_03_y_test.npy')

print(x_train.shape, x_test.shape)  # (21733, 100, 100, 3) (5434, 100, 100, 3)
print(y_train.shape, y_test.shape)  # (21733,) (5434,)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(100,100,3)))  # (98,98,64)
model.add(Conv2D(64, (3,3), activation='relu'))  # (96,96,64)
model.add(Dropout(0.2))
model.add(MaxPooling2D())   # (48,48,64)
model.add(Conv2D(64, (3,3), activation='relu'))  # (46,46,32)
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), activation='relu'))  # (44,44,32)
model.add(Dropout(0.2))
# model.add(Flatten())
model.add(GlobalAveragePooling2D()) # (1,1,32)
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True
)

path = './_save/keras52/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k52_15_", date, "-", filename])

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'min',
    verbose = 0,
    save_best_only = True,
    filepath = filepath
)

# learning_rate = 0.01
# learning_rate = 0.001 # adam은 디폴트 0.001
learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])
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
acc = accuracy_score(y_test, np.round(y_pred))
print("acc : ", acc)

print("걸린 시간 : ", round(end_time-start_time, 2), "초")

# Results
# Epoch 93/3000
# 544/544 [==============================] - 16s 29ms/step - loss: 0.0408 - acc: 0.9851 - val_loss: 0.2661 - val_acc: 0.9282
# 170/170 [==============================] - 2s 9ms/step - loss: 0.1909 - acc: 0.9277
# loss :  0.19087645411491394
# acc :  0.927677571773529
# 170/170 [==============================] - 1s 7ms/step
# acc :  0.9276775855723224
# 걸린 시간 :  1722.89 초

# learning_rate = 0.01
# Epoch 59/3000
# 544/544 [==============================] - 17s 31ms/step - loss: 0.6471 - acc: 0.6512 - val_loss: 0.6503 - val_acc: 0.6455
# 170/170 [==============================] - 2s 9ms/step - loss: 0.6454 - acc: 0.6535
# loss :  0.6454188227653503
# acc :  0.6534780859947205
# 170/170 [==============================] - 1s 8ms/step
# acc :  0.6534781008465219
# 걸린 시간 :  962.4 초

# learning_rate = 0.0001
# Epoch 477/2000
# 11/11 [==============================] - 0s 4ms/step - loss: 0.0227 - acc: 0.9912 - val_loss: 0.0110 - val_acc: 1.0000
# 4/4 [==============================] - 0s 1ms/step - loss: 0.1794 - acc: 0.9737
# loss :  0.1794
# acc :  0.9737
# 4/4 [==============================] - 0s 0s/step
# 걸린 시간 :  32.36 초
# acc_score : 0.9736842105263158