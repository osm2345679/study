# 71-2 카피
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np
import random

# print(tf.__version__)   # 2.9.3

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

#1. 데이터
x = np.array([1])
y = np.array([1])

#2. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

################### 동결 ###################
# model.trainable = False # 동결
model.trainable = True
################### 동결 ###################

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(
    x, y,
    epochs=1000,
    batch_size=1,
    verbose=1,
    callbacks=[]
)

#4. 평가, 예측
y_pred = model.predict(x)
print(y_pred)

print("=======================")
print(model.weights)
print("=======================")

# Results
# Epoch 999/1000
# 1/1 [==============================] - 0s 5ms/step - loss: 2.4770e-09
# Epoch 1000/1000
# 1/1 [==============================] - 0s 2ms/step - loss: 2.3888e-09
# 1/1 [==============================] - 0s 42ms/step
# [[1.0003544]]
# =======================
# [<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.08450541, -0.08842915,  0.8283203 ]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([-0.05152767, -0.05362988,  0.05395508], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
# array([[-0.9780324 ,  0.7826826 ],
#        [ 0.6976057 ,  1.0858648 ],
#        [-0.5848015 , -0.68448645]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([-0.06430405, -0.05273598], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
# array([[-0.08098985],
#        [-1.1340631 ]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.05222915], dtype=float32)>]
# =======================