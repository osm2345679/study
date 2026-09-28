# 54-1 카피

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
import datetime


#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10]) # (10,)

x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9]]) # (7,3)

y = np.array([4,5,6,7,8,9,10])  # (7,)

print(x.shape, y.shape) # (7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1)    # (7,3,1)
print(x.shape)  # (7,3,1)

#2. 모델 구성
model = Sequential()
model.add(SimpleRNN(units=5, input_shape=(3,1)))   # 행 무시 열 우선 : (7,3,1) -> (3,1)
# model.add(SimpleRNN(10, input_shape=(3,1)))   # 위와 같음 
# 3차원으로 들어가서 2(1)차원으로 나옴 -> 바로 Dense와 연결 가능
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련

#4. 평가, 예측

# Results
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  simple_rnn (SimpleRNN)      (None, 3)                 15        
                                                                 
#  dense (Dense)               (None, 7)                 28        
                                                                 
#  dense_1 (Dense)             (None, 1)                 8         
                                                                 
# =================================================================
# Total params: 51
# Trainable params: 51
# Non-trainable params: 0
# _________________________________________________________________

# Non-trainable params는 전이학습과 같이 가중치 가져오고 freeze할 때 중요

###### 파라미터 계산 ######
# 파라미터의 개수 = units*features + unites*bias + units*units
# 