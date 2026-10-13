# 61-3 카피
# ohe 적용해보기
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import datetime
import time

#1. 데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다', '한 번 더 보고 싶어요', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다', '참 재밌네요',
    '개똥이 바보', '말똥이 잘생겼다', '길동이 또 구라친다'
]
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])  # (15,)


token = Tokenizer()
token.fit_on_texts(docs)

x = token.texts_to_sequences(docs)

########### 패딩 ###########
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_x = pad_sequences(x,
                         padding='pre',  # pre : 앞을 0으로 채우기, post : 뒤를 0으로 채우기
                         maxlen=5,   # 요소의 길이.
                         truncating='post'  # defalut는 pre
)   
# print(padded_x.shape)   # (15, 5)

x_train, x_test, y_train, y_test = train_test_split(
    padded_x,labels,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# 추가로 예측하기
# x_predict = ["개똥이 잘생겼다"]
# x_predict = token.texts_to_sequences(x_predict)
# # print(x_predict)  # [[24, 27]]

# padded_x_predict = pad_sequences(x_predict,
#                          padding='pre',  # pre : 앞을 0으로 채우기, post : 뒤를 0으로 채우기
#                          maxlen=5,   # 요소의 길이.
#                          truncating='post'  # defalut는 pre
# )   
# # print(padded_x_predict) # [[ 0  0  0 24 27]]
# # print(padded_x_predict.shape)   # (1, 5)

# padded_x_predict = to_categorical(padded_x_predict)
# print(padded_x_predict)
# print(padded_x_predict.shape)

#2. 모델 구성
from tensorflow.keras.layers import Embedding

model = Sequential()
################ 임베딩 1 ################
model.add(Embedding(input_dim=30, output_dim=10, input_length=5))
#                   단어사전의 개수(vocabulary size), 출력 차원(Embedding Dimension)
# input_dim은 입력 데이터에 비해 적으면 알아서 자르고 많으면 그냥 많은 거라 상관없음.
# output_dim은 입력 데이터와 관계 없음
# input_length은 입력 데이터와 shape가 맞아야 함.
model.add(SimpleRNN(10))
model.add(Dense(1, activation='sigmoid'))
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  embedding (Embedding)       (None, 5, 10)             300       
                                                                 
#  simple_rnn (SimpleRNN)      (None, 10)                210       
                                                                 
#  dense (Dense)               (None, 1)                 11        
                                                                 
# =================================================================
# Total params: 521
# Trainable params: 521
# Non-trainable params: 0
# _________________________________________________________________

# embedding layer의 파라미터 수 계산
# (입력 차원 수) x (출력 차원 수)
# V(vocabulary size) x E(embeding dimension)

# 30 x 10 = 300

################ 임베딩 2 ################
model.add(Embedding(input_dim=30, output_dim=10))  # input_length 명시 안하면 알아서 맞춤 
#                   단어사전의 개수(vocabulary size), 출력 차원(Embedding Dimension)
model.add(SimpleRNN(10))
model.add(Dense(1, activation='sigmoid'))

################ 임베딩 3 ################
model.add(Embedding(30, 10))  # input_dim, output_dim 순. class Embedding(input_dim: int, output_dim: int, ...)
# model.add(Embedding(30, 10, 5))   # 얘는 에러남.
# model.add(Embedding(30, 10, input_length=5))    # input_length는 앞 인자들 다 적는 거 아니면 명시해줘야 함.
#                   단어사전의 개수(vocabulary size), 출력 차원(Embedding Dimension)
model.add(SimpleRNN(10))
model.add(Dense(1, activation='sigmoid'))

model.summary()

#3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True
)
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=100,
    verbose=1
)

filepath = './_save/keras61/'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([filepath, "k61_4_", date, "-", filename])

mcp = ModelCheckpoint(
    filepath=filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)

learning_rate = 0.01
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])
start_time = time.time()
model.fit(
    x_train, y_train,
    epochs=3,
    batch_size=1,
    verbose=1,
    validation_split=0.1,
    callbacks=[es, mcp, rlr]
)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_train, y_train)
print("loss : ", results[0])
print("acc : ", results[1])

y_pred = model.predict(x_test)
print(y_pred)
# [[2.4872404e-04]
#  [7.7388373e-05]
#  [1.4032151e-01]]

acc = accuracy_score(y_test, np.round(y_pred))
print("accuracy : ", acc)

print("걸린 시간 : ", round(end_time-start_time, 2))