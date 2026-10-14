from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D
from tensorflow.keras.applications import VGG16
import numpy as np

#2. 모델 구성
vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3)
)
vgg16.trainable = False

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

print(len(model.weights))   # 30. Conv layer 13개 있고(MaxPooling은 연산 안함) 각각 weight와 bias를 가지고 있으므로 13*2 = 26 이후 아래 dense 포함해서 26+4 = 30
# 30 = 13(vgg16) * 2(weight, bias) + 2(dense) * 2(weight, bias)
print(len(model.trainable_weights)) # 4. Dense layer 2개 있고 각각 weight와 bias를 가지고 있으므로 2*2 = 4