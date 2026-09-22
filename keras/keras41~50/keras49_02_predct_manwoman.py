# 49-1 카피
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

#1. 데이터
np_path = './_data/man_woman/'
x_train = np.load(np_path + 'keras46_03_x_train.npy')
x_test = np.load(np_path + 'keras46_03_x_test.npy')
y_train = np.load(np_path + 'keras46_03_y_train.npy')
y_test = np.load(np_path + 'keras46_03_y_test.npy')

#2. 모델 구성
path = './_save/keras47/'
model = load_model(path + 'k47_3_0921_1519-0043-0.1910.keras')

#3. 컴파일, 훈련

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss[0])
print("acc : ", loss[1])

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, np.round(y_pred))
print("acc : ", acc)

# 사진 가지고 예측
np_path = './_data/kaggle_cat_dog_npy/'
selfy = np.load(np_path + 'keras48_me_.npy')

selfy_pred = model.predict(selfy)
print(selfy_pred[0])

# Results
# 170/170 [==============================] - 23s 56ms/step - loss: 0.1909 - acc: 0.9277
# loss :  0.19087645411491394
# acc :  0.927677571773529
# 2026-09-21 16:04:17.310118: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 652080000 exceeds 10% of free system memory.
# 170/170 [==============================] - 10s 57ms/step
# acc :  0.9276775855723224
# 1/1 [==============================] - 0s 94ms/step
# [0.04790402]