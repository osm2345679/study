import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.3,
    random_state=42
)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=5)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

results = model.predict(x)
print("x의 예측값 : ", results)

plt.scatter(x, y)
plt.plot(results, color='red')
plt.show()

# Results
# 
# Epoch 300/300
# 3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 6.8365 
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 61ms/step - loss: 20.5841
# loss :  20.58412742614746
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step
# x의 예측값 :  [[ 1.0613787]
#  [ 2.027241 ]
#  [ 2.993103 ]
#  [ 3.9589655]
#  [ 4.9248276]
#  [ 5.8906894]
#  [ 6.8565516]
#  [ 7.822414 ]
#  [ 8.788276 ]
#  [ 9.754137 ]
#  [10.719999 ]
#  [11.685863 ]
#  [12.651722 ]
#  [13.617586 ]
#  [14.583447 ]
#  [15.54931  ]
#  [16.515171 ]
#  [17.481033 ]
#  [18.446894 ]
#  [19.412758 ]]