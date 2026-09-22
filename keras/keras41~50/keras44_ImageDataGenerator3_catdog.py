# 44-2 카피
# 데이터셋 - https://www.kaggle.com/datasets/tongpython/cat-and-dog/data
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
train_datagen = ImageDataGenerator(
    rescale=1./255,   # . 표시 : 형변환 표시 안해도 되는데 해주는 편이 형변환했음을 보기 쉬움

    #### 데이터 증폭
    # horizontal_flip=True,   # 수평 뒤집기
    # vertical_flip=True, # 수직 뒤집기
    # width_shift_range=0.1, # 평형이동
    # height_shift_range=0.1,
    # rotation_range=5,    # 각도 조절(정해진 각도만큼 이미지 회전)
    # zoom_range=1.2,
    # shear_range=0.7, # 좌표 하나를 고정하고 다른 몇 개의 좌표를 이동. 찌부되는 거.
    # fill_mode='nearest'
)
test_datagen = ImageDataGenerator(
    rescale=1./255
)

path_train = './_data/image/cat_dog/training_set/'
path_test = './_data/image/cat_dog/test_set/'

xy_train = train_datagen.flow_from_directory(
    path_train, # 경로
    target_size=(100,100),
    batch_size=10000,
    class_mode='binary', # 이진분류
    color_mode='rgb',  # 흑백
    shuffle=True
)
# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(100,100),
    batch_size=10000,
    class_mode='binary', # 이진분류
    color_mode='rgb',  # 흑백
    shuffle=False
    # shuffle=True  # test에서는 불필요. 안 건드는 게 맞음.
)
# Found 120 images belonging to 2 classes.

# print(xy_train) # <keras.preprocessing.image.DirectoryIterator object at 0x000001BEFB287FA0>
# print(xy_train.next())  # 첫번째 보여짐
# print(xy_train.next())  # 그 다음(두번째) 보여짐

# print(xy_train[0][0])   # 첫번째 배치의 x 데이터
# print(xy_train[0][1])   # 첫번째 배치의 y 데이터

print(xy_train[0][0].shape) # ((8005, 100, 100, 3))
print(xy_train[0][1].shape) # (8005,)

# print(xy_train[16][0])  # 에러남. 총 160장, 배치 16개라서

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

# plt.imshow(x_train[0])
# plt.show()

print(x_train.shape, y_train.shape) # (8005, 100, 100, 3) (8005,)
print(x_test.shape, y_test.shape)   # (2023, 100, 100, 3) (2023,)

#2. 모델 구성
model = Sequential()
# model.add(Conv2D(64, (3,3), input_shape=(100,100,3)))  # (98,98,64)
# model.add(Conv2D(64, (3,3), activation='relu'))  # (96,96,64)
# model.add(Dropout(0.2))
# model.add(MaxPooling2D())   # (48,48,64)
# model.add(Conv2D(64, (3,3), activation='relu'))  # (46,46,64)
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (3,3), activation='relu'))  # (44,44,32)
# model.add(Dropout(0.2))
# # model.add(Flatten())
# model.add(GlobalAveragePooling2D()) # (1,1,32)
# model.add(Dense(16, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

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
model.add(Dense(1, activation='sigmoid'))

# model.summary()
# config = model.get_config()
# print(config)

# print(model.to_json())

# print(model.layers[0].get_config())

#3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True
)

path = './_save/keras44/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, "k44_3_", date, "-", filename])

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
# Epoch 117/3000
# 201/201 [==============================] - 5s 26ms/step - loss: 0.0942 - acc: 0.9625 - val_loss: 0.4974 - val_acc: 0.8413
# 64/64 [==============================] - 1s 9ms/step - loss: 0.3927 - acc: 0.8497
# loss :  0.39274874329566956
# acc :  0.8497281074523926
# 64/64 [==============================] - 0s 6ms/step
# acc :  0.8497281265447355
# 걸린 시간 :  617.26 초