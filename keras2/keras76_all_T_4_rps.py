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

model_list = [VGG19(include_top=False, input_shape=(100, 100, 3)),
              ResNet50(include_top=False, input_shape=(100, 100, 3)),
              ResNet101(include_top=False, input_shape=(100, 100, 3)),
              DenseNet201(include_top=False, input_shape=(100, 100, 3)),
            #   InceptionV3(), InceptionResNetV2(), # 17, 18에는 적용
              MobileNetV2(include_top=False, input_shape=(100, 100, 3)),
              EfficientNetB0(include_top=False, input_shape=(100, 100, 3))
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
np_path = './_data/npy_rps/'

x_train = np.load(np_path + 'keras46_02_x_train.npy')
y_train = np.load(np_path + 'keras46_02_y_train.npy')
x_test = np.load(np_path + 'keras46_02_x_test.npy')
y_test = np.load(np_path + 'keras46_02_y_test.npy')

#2. 모델 구성
for model_i in model_list:
    model_i.trainable = False

    model = Sequential()
    model.add(model_i)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    #3. 컴파일, 훈련
    learning_rate = 0.001

    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])

    path = './_save/keras76/'
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")
    filename = '{epoch:04d}-{val_loss:.4f}.keras'
    filepath = "".join([path, "k76_4_", date, "-", filename])

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
        patience=50,
        restore_best_weights=True
    )
    rlr = ReduceLROnPlateau(
        monitor='val_loss',
        mode='auto',
        patience=40,
        verbose=0,
        factor=0.5
    )
    start_time = time.time()
    model.fit(
        x_train, y_train,
        epochs=3000,
        batch_size=32,
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