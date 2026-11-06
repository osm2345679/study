# 53-2 카피

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
# model = Sequential()
# model.add(Dense(32, input_dim=10))
# model.add(Dropout(0.2))
# model.add(Dense(16))
# model.add(Dropout(0.2))
# model.add(Dense(4))
# model.add(Dropout(0.2))
# model.add(Dense(2))
# model.add(Dropout(0.2))
# model.add(Dense(1))

def build_model(drop=0.5, optimizer='Adam', activation='relu', node1=128, node2=64, node3=32, node4=16, node5=8, lr=0.001):
    inputs = Input(shape=(10,), name='inputs')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(node4, activation=activation, name='hidden4')(x)
    x = Dense(node5, activation=activation, name='hidden5')(x)
    outputs = Dense(1, activation='linear', name='outputs')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    
    return model

def create_hyperparameter():
    batchs = [32, 16, 8, 1, 64]
    optimizers = ['Adam', 'RMSProp', 'Adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    node1 = [128, 64, 32, 16]
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16]
    node4 = [128, 64, 32, 16]
    node5 = [128, 64, 32, 16]
    return {
        'batch_size' : batchs,
        'optimizer' : optimizers,
        'drop' : dropouts,
        'activation' : activations,
        'node1' : node1,
        'node2' : node2,
        'node3' : node3,
        'node4' : node4,
        'node5' : node5,
    }

hyperparameters = create_hyperparameter()   # dictionary 타입
# print(hyperparameters)
# {'batch_size': [32, 16, 8, 1, 64],
#  'drop': [0.2, 0.3, 0.4, 0.5],
#  'activation': ['relu', 'elu', 'selu', 'linear                                                              '],
#  'node1': [128, 64, 32, 16],
#  'node2': [128, 64, 32, 16],
#  'node3': [128, 64, 32, 16],
#  'node4': [128, 64, 32, 16],
#  'node5': [128, 64, 32, 16]}

from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 10,
    restore_best_weights = True,
    verbose = 1
)

rlr = ReduceLROnPlateau(
    monitor = 'val_loss',
    mode = 'auto',
    factor = 0.8, 
    patience = 8,
    verbose = 1
)

keras_model = KerasRegressor(build_fn=build_model, verbose=1)

model = RandomizedSearchCV(
    keras_model,
    hyperparameters,
    cv=3,
    random_state=42,
    # n_jobs=-1,    # 메모리 부족 오류나서 주석처리함.
    refit=True  # defalut
)

#3. 컴파일, 훈련
start = time.perf_counter()
model.fit(x_train, y_train, epochs=1000, callbacks=[es, rlr], validation_split=0.1)
end = time.perf_counter()

print("최적의 매개변수 :", model.best_estimator_)
print("최적의 파라미터 :", model.best_params_)

#4. 평가, 예측
print("model.score :", model.score(x_test, y_test))
print("best_score :", model.best_score_)    # best_score_: x_train 내부에서 수행한 3겹 교차검증의 검증 점수 평균

print("걸린 시간 :", round(end-start, 2), "초")

# Results
# Epoch 22: early stopping
# 최적의 매개변수 : <keras.wrappers.scikit_learn.KerasRegressor object at 0x000002AD316BE5F0>
# 최적의 파라미터 : {'optimizer': 'Adam', 'node5': 16, 'node4': 128, 'node3': 64, 'node2': 16, 'node1': 32, 'drop': 0.2, 'batch_size': 16, 'activation': 'selu'}
# 1/6 [====>.........................] - ETA: 0s - loss:4/6 [===================>..........] - ETA: 0s - loss:6/6 [==============================] - 0s 15ms/step - loss: 2909.9060 - mae: 43.3452
# model.score : -2909.906005859375
# best_score : -3308.833984375
# 걸린 시간 : 3265.96 초