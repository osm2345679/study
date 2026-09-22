# 47-3 카피
# 데이터셋 - https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, GlobalAveragePooling2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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

x_train_woman = x_train[np.where(y_train==1)]
print(x_train_woman.shape)  # (7606, 100, 100, 3)

y_train_woman = y_train[np.where(y_train==1)]
print(y_train_woman.shape)  # (7606,)

#################### 데이터 증폭 ####################
datagen = ImageDataGenerator(
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

augment_size = 7000

# randidx = np.random.randint(x_train.shape[0], size=augment_size)   # 60000개 중 40000개 랜덤 뽑기
randidx = np.random.choice(x_train_woman.shape[0], size=augment_size, replace=False)   # 60000개 중 40000개 랜덤 뽑기

x_augmented = x_train_woman[randidx].copy()   # 메모리 공간 문제 방지를 위해 .copy() 사용함
print(x_augmented.shape)
y_augmented = y_train_woman[randidx].copy()

x_augmented = datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False
).next()[0] # [0]으로 x만 

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

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

path = './_save/keras51/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k51_5_", date, "-", filename])

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'min',
    verbose = 0,
    save_best_only = True,
    filepath = filepath
)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
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
# Epoch 163/3000
# 669/669 [==============================] - 20s 30ms/step - loss: 0.0234 - acc: 0.9914 - val_loss: 0.3295 - val_acc: 0.8999
# 170/170 [==============================] - 2s 9ms/step - loss: 0.2508 - acc: 0.9223
# loss :  0.2508142292499542
# acc :  0.9223408102989197
# 170/170 [==============================] - 1s 7ms/step
# acc :  0.9223408170776591
# 걸린 시간 :  3354.41 초