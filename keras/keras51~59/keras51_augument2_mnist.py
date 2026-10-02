# 39-1 카피
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import time


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

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
    x_augmented.shape[2], 1
)

x_augmented = datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False
).next()[0] # [0]으로 x만 

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

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
model = Sequential()
model.add(Conv2D(64, (5,5), input_shape=(28, 28, 1)))   # (24, 24, 64). input_shape에서 행 무시-열 우선. (60000,28,28,1) -> (28,28,1)
model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu')) # (20,20,64). 파라미터 이름도 적어봄
model.add(Dropout(0.2))
model.add(Conv2D(32, (5,5), activation='relu')) # (16,16,32)
model.add(Conv2D(32, (5,5), activation='relu')) # (12,12,32)
model.add(Dropout(0.2))
model.add(Conv2D(32, (5,5), activation='relu')) # (8,8,32)
model.add(Dropout(0.2))
model.add(Conv2D(32, (5,5), activation='relu')) # (4,4,32)
model.add(Dropout(0.2))
model.add(Flatten())    # (512,)
# 4차원 데이터(N,20,20,16)을 2차원 데이터(N,6400)로 변환
model.add(Dense(units=32, activation='relu'))   # units: Dense의 output 갯수
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(10, activation='softmax'))  # (10,)
#3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=40,
    restore_best_weights=True
)

path = './_save/keras51/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k51_2_", date, "-", filename])

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    verbose=1,
    save_best_only=True,
    filepath = filepath
)
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc'])
start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=2000,
    batch_size=128,
    verbose=1,
    validation_split=0.2,
    callbacks=[es, mcp]
)
end_time = time.time()

#4. 평가, 예측
print("=========model.evaluate==========")
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss[0])
print("acc : ", loss[1])

y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_pred)

print("accuracy_score : ", acc)
print("걸린 시간 : ", round(end_time-start_time, 2), "초")

# Results   
# Epoch 75: val_loss did not improve from 0.13653
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 48s 77ms/step - acc: 0.9943 - loss: 0.0244 - val_acc: 0.9683 - val_loss: 0.1929
# =========model.evaluate==========
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - acc: 0.9925 - loss: 0.0289 
# loss :  0.028916293755173683
# acc :  0.9925000071525574
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step  
# accuracy_score :  0.9925
# 걸린 시간 :  4362.1 초