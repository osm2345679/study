from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import numpy as np

#1. 데이터
np_path = './_data/kaggle_cat_dog_npy/'
x_train = np.load(np_path + 'keras45_03_x_train.npy')
x_test = np.load(np_path + 'keras45_03_x_test.npy')
y_train = np.load(np_path + 'keras45_03_y_train.npy')
y_test = np.load(np_path + 'keras45_03_y_test.npy')



#2. 모델 구성
path = './_save/keras44/'
model = load_model(path + 'k44_3_0918_1723-0067-0.3732.keras')

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
# 0.28 정도