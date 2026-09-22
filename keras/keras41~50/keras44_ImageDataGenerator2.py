# 44-1 카피
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, GlobalAveragePooling2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import numpy as np
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

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(
    path_train, # 경로
    target_size=(150,150),
    batch_size=800,
    class_mode='binary', # 이진분류
    color_mode='grayscale',  # 흑백
    shuffle=True
)
# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(150,150),
    batch_size=200,
    class_mode='binary', # 이진분류
    color_mode='grayscale',  # 흑백
    shuffle=False
    # shuffle=True  # test에서는 불필요. 안 건드는 게 맞음.
)
# Found 120 images belonging to 2 classes.

# print(xy_train) # <keras.preprocessing.image.DirectoryIterator object at 0x000001BEFB287FA0>
# print(xy_train.next())  # 첫번째 보여짐
# print(xy_train.next())  # 그 다음(두번째) 보여짐

# print(xy_train[0][0])   # 첫번째 배치의 x 데이터
# print(xy_train[0][1])   # 첫번째 배치의 y 데이터

# print(xy_train[0][0].shape) # (160, 100, 100, 1)
# print(xy_train[0][1].shape) # (160,)

# print(xy_train[16][0])  # 에러남. 총 160장, 배치 16개라서

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

# print(x_train.shape, y_train.shape) # (160, 150, 150, 1) (160,)
# print(x_test.shape, y_test.shape)   # (120, 150, 150, 1) (120,)

#2. 모델 구성
model = Sequential()
# model.add(Conv2D(64, (11,11), strides=4, input_shape=(150,150,1)))
# model.add(Conv2D(64, (5,5), strides=1, activation='relu'))
# model.add(Dropout(0.2))
# model.add(MaxPooling2D())
# model.add(Conv2D(16, (5,5), strides=1, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Flatten())


model.add(Conv2D(64, (11,11), strides=4, input_shape=(150,150,1)))
model.add(Conv2D(64, (5,5), strides=1, activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Conv2D(16, (5,5), strides=1, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())


model.add(Dense(1, activation='sigmoid'))

model.summary()
config = model.get_config()
print(config)

print(model.to_json())

print(model.layers[0].get_config())

exit()

#3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True
)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=3000,
    batch_size=32,
    verbose=1,
    validation_split=0.2,
    callbacks=[es]
)
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss[0])
print("acc : ", loss[1])

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, np.round(y_pred))
print("acc : ", acc)

if acc == 1.0 :
    path = './_save/keras44/'
    model.save(path + 'keras44_ImageDataGenerator1.keras')

print("걸린 시간 : ", round(end_time-start_time, 2), "초")

# Results
# Epoch 724/3000
# 4/4 [==============================] - 0s 9ms/step - loss: 6.5032e-06 - acc: 1.0000 - val_loss: 2.4394e-04 - val_acc: 1.0000
# 4/4 [==============================] - 0s 17ms/step - loss: 0.0028 - acc: 1.0000
# loss :  0.002847416093572974
# acc :  1.0
# 4/4 [==============================] - 0s 3ms/step
# acc :  1.0
# 걸린 시간 :  26.53 초