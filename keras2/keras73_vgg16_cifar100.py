# 73-1 카피
# 실습
# 비교할 것
# - 이전 결과
# - 가중치 동결 결과
# - 가중치 동결 안했을 때 결과
# - 걸린 시간
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications import VGG16
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import datetime
import time

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

#################### 데이터 증폭 ####################
datagen = ImageDataGenerator(
    rescale=1./255,   # . 표시 : 형변환 표시 안해도 되는데 해주는 편이 형변환했음을 보기 쉬움
    
    #### 데이터 증폭
    horizontal_flip=True,   # 수평 뒤집기 (좌우반전)
    # vertical_flip=True, # 수직 뒤집기 (상하반전)
    width_shift_range=0.1, # 평형이동
    height_shift_range=0.1,
    rotation_range=15,    # 각도 조절(정해진 각도만큼 이미지 회전)
    zoom_range=0.1,
    shear_range=0.7, # 좌표 하나를 고정하고 다른 몇 개의 좌표를 이동. 찌부되는 거.
    fill_mode='nearest'
)

augment_size = 40000

# randidx = np.random.randint(x_train.shape[0], size=augment_size)   # 60000개 중 40000개 랜덤 뽑기
randidx = np.random.choice(x_train.shape[0], size=augment_size, replace=False)   # 60000개 중 40000개 랜덤 뽑기

x_augmented = x_train[randidx].copy()   # 메모리 공간 문제 방지를 위해 .copy() 사용함
y_augmented = y_train[randidx].copy()

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2], 3
)

x_augmented = datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False
).next()[0] # [0]으로 x만 

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

# scaling
x_train = x_train/255
x_test = x_test/255

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

# one-hot encoding
ohe = OneHotEncoder(sparse_output=False)

y_train = ohe.fit_transform(y_train.reshape(-1,1))
y_test = ohe.fit_transform(y_test.reshape(-1,1))

#2. 모델 구성
vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3)
)
vgg16.trainable = False # 동결
# vgg16.summary()

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

#3. 컴파일, 훈련
learning_rate = 0.001

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])

path = './_save/keras73/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k73_2_", date, "-", filename])

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    verbose=1,
    save_best_only=True,
    filepath = filepath
)
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=40,
    restore_best_weights=True
)
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=40,
    verbose=1,
    factor=0.5
)
start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=2000,
    batch_size=256,
    verbose=1,
    validation_split=0.2,
    callbacks=[es, mcp, rlr]
)
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss[0])
print("acc : ", loss[1])

y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_pred)
print("acc : ", acc)
print("걸린 시간 : ", round(end_time-start_time, 2), "초")

# Results
# 52-14
# learning_rate = 0.0001
# Epoch 920/2000
# 280/282 [============================>.] - ETA: 0s - loss: 1.4417 - acc: 0.5920
# Epoch 920: val_loss did not improve from 1.75492
# 282/282 [==============================] - 6s 21ms/step - loss: 1.4416 - acc: 0.5920 - val_loss: 1.7785 - val_acc: 0.5220
# 313/313 [==============================] - 1s 2ms/step - loss: 1.7922 - acc: 0.5242
# loss :  1.7921879291534424
# acc :  0.5242000222206116
# 313/313 [==============================] - 0s 1ms/step
# acc :  0.5242
# 걸린 시간 :  5890.0 초

# Freeze
# Epoch 62: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
# 282/282 [==============================] - 7s 26ms/step - loss: 0.9836 - acc: 0.7265 - val_loss: 2.9022 - val_acc: 0.3799 - lr: 0.0010
# 313/313 [==============================] - 2s 5ms/step - loss: 2.6744 - acc: 0.3655
# loss :  2.6744384765625
# acc :  0.36550000309944153
# 313/313 [==============================] - 1s 4ms/step
# acc :  0.3655
# 걸린 시간 :  455.65 초

# Unfreeze
# Epoch 52: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
# 282/282 [==============================] - 19s 67ms/step - loss: 0.2953 - acc: 0.9167 - val_loss: 3.7224 - val_acc: 0.4206 - lr: 0.0010
# 313/313 [==============================] - 2s 5ms/step - loss: 2.5207 - acc: 0.3866
# loss :  2.5207016468048096
# acc :  0.38659998774528503
# 313/313 [==============================] - 1s 4ms/step
# acc :  0.3866
# 걸린 시간 :  978.77 초