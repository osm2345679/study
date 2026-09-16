from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import pandas as pd
import numpy as np

#2. 모델 구성
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(10,10,1),  # (10,10,10)
                 strides=1,
                 padding='same'
))
model.add(MaxPooling2D())
model.add(Conv2D(filters=9, kernel_size=(3,3),  # (8,8,9)
                 strides=1, # defalut는 1.
                 padding='valid', # defalut는 valid
))
model.summary()