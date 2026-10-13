from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np
import random

print(tf.__version__)   # 2.9.3

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(10))
model.add(Dense(1))

model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  dense (Dense)               (None, 3)                 6         
                                                                 
#  dense_1 (Dense)             (None, 10)                40        
                                                                 
#  dense_2 (Dense)             (None, 1)                 11        
                                                                 
# =================================================================
# Total params: 57
# Trainable params: 57
# Non-trainable params: 0
# _________________________________________________________________

print("================model.weights================")
print(model.weights)
print("================model.trainable weights================")
print(model.trainable_weights)  # print(model.weights)와 동일한 결과 출력
print("===============================")
print(len(model.weights))   # 6
print(len(model.trainable_weights)) # 6

################### 동결 ###################
model.trainable = False # 동결
################### 동결 ###################
print("===============================")
print(len(model.weights))   # 6
print(len(model.trainable_weights)) # 0
print("================model.trainable weights================")
print(model.trainable_weights)  # []
print("================model.weights================")
print(model.weights)

model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  dense (Dense)               (None, 3)                 6         
                                                                 
#  dense_1 (Dense)             (None, 10)                40        
                                                                 
#  dense_2 (Dense)             (None, 1)                 11        
                                                                 
# =================================================================
# Total params: 57
# Trainable params: 0
# Non-trainable params: 57
# _________________________________________________________________