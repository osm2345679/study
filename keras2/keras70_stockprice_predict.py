from tensorflow.keras.models import load_model
import numpy as np

np_path = './_data/ensemble/'
x1_train = np.load(np_path +  'keras70_x1_train.npy')
x1_test = np.load(np_path +  'keras70_x1_test.npy')
x2_train = np.load(np_path +  'keras70_x2_train.npy')
x2_test = np.load(np_path +  'keras70_x2_test.npy')
y_train = np.load(np_path + 'keras70_y_train.npy')
y_test = np.load(np_path + 'keras70_y_test.npy')
x1_pred = np.load(np_path + 'keras70_x1_pred.npy')
x2_pred = np.load(np_path + 'keras70_x2_pred.npy')

path = './_save/keras70/'
model = load_model(path + 'keras70_1012_1505-0108-37135724.0000.keras')

results = model.evaluate([x1_test, x2_test], y_test)
print("loss : ", results)

y_pred = model.predict([x1_pred, x2_pred])
print("2026-10-13의 삼성전자 예측 시가 : ", y_pred, "원")
