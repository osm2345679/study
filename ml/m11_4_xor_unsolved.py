# 11-1 카피

from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np
tf.random.set_seed(42)
np.random.seed(42)

#1. 데이터
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([0, 1, 1, 0])

#2. 모델
# model = Perceptron()
# model = LinearSVC()
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, epochs=1000)

#4. 평가, 예측
y_predict = model.predict(x_data)

results = model.evaluate(x_data, y_data)
print("loss, acc :", results)

acc = accuracy_score(y_data, np.round(y_predict))
print("accuracy_socre : ", acc)

# Results
# Epoch 1000/1000
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 35ms/step - acc: 0.2500 - loss: 0.6955
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 47ms/step
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 105ms/step - acc: 0.2500 - loss: 0.6955
# loss, acc : [0.6954502463340759, 0.25]
# accuracy_socre :  0.25