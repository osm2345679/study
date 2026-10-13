# 71-1 카피
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
model.add(Dense(50, input_dim=1))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

################### 동결 ###################
model.trainable = False # 동결
# model.trainable = True
################### 동결 ###################

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(
    x, y,
    epochs=100,
    batch_size=1,
    verbose=1,
    callbacks=[]
)

#4. 평가, 예측
y_pred = model.predict(x)
print(y_pred)

# Results
# Epoch 97/100
# 5/5 [==============================] - 0s 877us/step - loss: 12.9133
# Epoch 98/100
# 5/5 [==============================] - 0s 801us/step - loss: 12.9133
# Epoch 99/100
# 5/5 [==============================] - 0s 829us/step - loss: 12.9133
# Epoch 100/100
# 5/5 [==============================] - 0s 759us/step - loss: 12.9133
# 1/1 [==============================] - 0s 95ms/step
# [[-0.08347169]
#  [-0.16694337]
#  [-0.25044987]
#  [-0.33388674]
#  [-0.417488  ]]