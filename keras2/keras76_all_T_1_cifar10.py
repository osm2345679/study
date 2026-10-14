# 75 카피
import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import DenseNet201, DenseNet169, DenseNet121
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.applications import NASNetMobile, NASNetLarge
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
from tensorflow.keras.applications import Xception

model_list = [VGG19(include_top=False, input_shape=(32, 32, 3)),
              ResNet50(include_top=False, input_shape=(32, 32, 3)),
              ResNet101(include_top=False, input_shape=(32, 32, 3)),
              DenseNet201(include_top=False, input_shape=(32, 32, 3)),
            #   InceptionV3(), InceptionResNetV2(), # 17, 18에는 적용
              MobileNetV2(include_top=False, input_shape=(32, 32, 3)),
              EfficientNetB0(include_top=False, input_shape=(32, 32, 3))
            ]

from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
import datetime
import random
import time

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

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

#2. 모델 구성
for model_i in model_list:
    model_i.trainable = False

    model = Sequential()
    model.add(model_i)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    #3. 컴파일, 훈련
    learning_rate = 0.001

    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])

    path = './_save/keras76/'
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")
    filename = '{epoch:04d}-{val_loss:.4f}.keras'
    filepath = "".join([path, "k76_1_", date, "-", filename])

    mcp = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        verbose=0,
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
        patience=20,
        verbose=0,
        factor=0.5
    )
    start_time = time.time()
    model.fit(
        x_train, y_train,
        epochs=2000,
        batch_size=128,
        verbose=0,
        validation_split=0.2,
        callbacks=[es, mcp, rlr]
    )
    end_time = time.time()

    #4. 평가, 예측
    loss = model.evaluate(x_test, y_test, verbose=0)
    print("=======================")
    print("모델명 :", model_i.name, "loss :", loss[0], "acc :", loss[1])

    y_pred = model.predict(x_test)

    y_pred = np.argmax(y_pred, axis=1)

    acc = accuracy_score(y_test, y_pred)
    print("acc : ", acc)
    print("걸린 시간 : ", round(end_time-start_time, 2), "초")

# Results
# =======================
# 모델명 : vgg19 loss : 1.1752737760543823 acc : 0.6069999933242798
# 313/313 [==============================] - 2s 5ms/step
# acc :  0.607
# 걸린 시간 :  493.53 초
# =======================
# 모델명 : resnet50 loss : 1.4141433238983154 acc : 0.5073000192642212
# 313/313 [==============================] - 3s 7ms/step
# acc :  0.5073
# 걸린 시간 :  1996.57 초
# =======================
# 모델명 : resnet101 loss : 1.6211379766464233 acc : 0.43320000171661377
# 313/313 [==============================] - 5s 12ms/step
# acc :  0.4332
# 걸린 시간 :  3389.87 초
# =======================
# 모델명 : densenet201 loss : 1.0588675737380981 acc : 0.6326000094413757
# 313/313 [==============================] - 8s 19ms/step
# acc :  0.6326
# 걸린 시간 :  685.63 초
# =======================
# 모델명 : mobilenetv2_1.00_224 loss : 1.7759287357330322 acc : 0.36070001125335693
# 313/313 [==============================] - 2s 6ms/step
# acc :  0.3607
# 걸린 시간 :  239.06 초
# =======================
# 모델명 : efficientnetb0 loss : 2.302600860595703 acc : 0.10000000149011612
# 313/313 [==============================] - 3s 7ms/step
# acc :  0.1
# 걸린 시간 :  466.38 초