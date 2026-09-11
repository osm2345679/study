from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

x_train = x[0:8]
x_val = x[8:12]
x_test = x[12:17]
y_train = x[0:8]
y_val = x[8:12]
y_test = x[12:17]

print(x_train, x_val, x_test, y_train, y_val, y_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=4,verbose=1,validation_data=(x_val, y_val))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

result = model.predict(x_test)
print("result : ", result)

# Results
# 
# Epoch 300/300
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step - loss: 0.0258 - val_loss: 0.1090
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 68ms/step - loss: 0.3408
# loss :  0.3408341407775879
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step
# result :  [[12.517046]
#  [13.452769]
#  [14.388492]
#  [15.324213]]