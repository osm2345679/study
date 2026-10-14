# 74-3 카피
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D
from tensorflow.keras.applications import VGG16
import pandas as pd
import numpy as np

#2. 모델 구성
model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3)
)
# model.trainable = False

model.layers[11].trainable = False
model.summary()

pd.set_option('max_colwidth', None)
layers = [
    (type(layer).__name__, layer.name, layer.trainable)
    for layer in model.layers
]
results = pd.DataFrame(
    layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable']
)
# pandas 기초
# pandas에 있는 객체는 series, dataframe뿐. 벡터가 series, 행렬이 dataframe

print(results)
#    Layer Type Layer Name  Layer Trainable
# 0  Functional      vgg16            False
# 1     Flatten    flatten             True
# 2       Dense      dense             True
# 3       Dense    dense_1             True
# 4       Dense    dense_2             True
# 5       Dense    dense_3             True
# 6       Dense    dense_4             True