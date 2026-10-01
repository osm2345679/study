# 61-2 카피
# ohe 적용해보기
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
# print(token.word_counts)
# OrderedDict([('너무', 2), ('재미있다', 1), ('참', 3), ('최고에요', 1), ('잘만든', 1), ('영화에요', 1),
#  ('추천하고', 1), ('싶은', 1), ('영화입니다', 1), ('한', 1), ('번', 1), ('더', 1), ('보고', 1), ('싶어요', 1),
#  ('글쎄', 1), ('별로에요', 1), ('생각보다', 1), ('지루해요', 1), ('연기가', 1), ('어색해요', 1), ('재미없어요', 1),
#  ('재미없다', 1), ('재밌네요', 1), ('개똥이', 1), ('바보', 1), ('말똥이', 1), ('잘생겼다', 1), ('길동이', 1), ('또', 1), ('구라친다', 1)])

# print(token.word_index)
# {'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화에요': 6, '추천하고': 7, '싶은': 8, '영화입니다': 9, '한': 10, '번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15, '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 21, '재미없다': 22, '재밌네요': 23, '개똥이': 24, '바보': 25, '말똥이': 26, '잘생겼다': 27, '길동이': 28, '또': 29, '구라친다': 30}

x = token.texts_to_sequences(docs)
# print(x)
# [[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14], [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23], [24, 25], [26, 27], [28, 29, 30]]


# 데이터 간 길이(shape)가 일치하지 않으므로 shape를 맞춰야 함
# 짧은 건 padding으로 늘리고 긴 건 자름.

########### 패딩 ###########
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_x = pad_sequences(x,
                         padding='pre',  # pre : 앞을 0으로 채우기, post : 뒤를 0으로 채우기
                         maxlen=5,   # 요소의 길이.
                         truncating='post'  # defalut는 pre
)   
# print(padded_x)
# [[ 0  0  0  2  3]
#  [ 0  0  0  1  4]
#  [ 0  0  1  5  6]
#  [ 0  0  7  8  9]
#  [10 11 12 13 14]
#  [ 0  0  0  0 15]
#  [ 0  0  0  0 16]
#  [ 0  0  0 17 18]
#  [ 0  0  0 19 20]
#  [ 0  0  0  0 21]
#  [ 0  0  0  2 22]
#  [ 0  0  0  1 23]
#  [ 0  0  0 24 25]
#  [ 0  0  0 26 27]
#  [ 0  0 28 29 30]]
# print(padded_x.shape)   # (15, 5)

padded_x = to_categorical(padded_x)
# print(padded_x)
# print(padded_x.shape)   # (15, 5, 31)

x_train, x_test, y_train, y_test = train_test_split(
    padded_x,labels,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# 추가로 예측하기
x_predict = ["개똥이 잘생겼다"]
x_predict = token.texts_to_sequences(x_predict)
# print(x_predict)  # [[24, 27]]

padded_x_predict = pad_sequences(x_predict,
                         padding='pre',  # pre : 앞을 0으로 채우기, post : 뒤를 0으로 채우기
                         maxlen=5,   # 요소의 길이.
                         truncating='post'  # defalut는 pre
)   
# print(padded_x_predict) # [[ 0  0  0 24 27]]
# print(padded_x_predict.shape)   # (1, 5)

padded_x_predict = to_categorical(padded_x_predict)

#2. 모델 구성
# model = Sequential()
# model.add(LSTM(10, input_shape=(5,1)))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

model = Sequential()
model.add(LSTM(10, input_shape=(5,31), dropout=0.2))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))

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
filepath = "".join([filepath, "k61_1_", date, "-", filename])

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
    epochs=3000,
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

y_pred = model.predict(padded_x_predict)
# print(y_pred)   # [[0.30582455]]
# print(y_pred.shape) # (1, 1)

print("\"개똥이 잘생겼다\"의 예측값 : ", y_pred)

print("걸린 시간 : ", round(end_time-start_time, 2))

# Results
# Epoch 239: ReduceLROnPlateau reducing learning rate to 0.0019999999552965165.
# 1/1 [==============================] - 0s 21ms/step - loss: 0.0683 - acc: 1.0000 - val_loss: 0.0131 - val_acc: 1.0000 - lr: 0.0200
# 1/1 [==============================] - 0s 11ms/step - loss: 0.1507 - acc: 0.9167
# loss :  0.15074686706066132
# acc :  0.9166666865348816
# 1/1 [==============================] - 0s 48ms/step
# accuracy :  0.6666666666666666
# 1/1 [==============================] - 0s 10ms/step
# "개똥이 잘생겼다"의 예측값 :  [[0.20792452]]
# 걸린 시간 :  6.18

# LSTM
# Epoch 640: val_loss did not improve from 0.00003
# 1/1 [==============================] - 0s 28ms/step - loss: 1.4412e-04 - acc: 1.0000 - val_loss: 3.1489e-05 - val_acc: 1.0000 - lr: 2.0000e-06
# 1/1 [==============================] - 0s 4ms/step - loss: 1.2510e-04 - acc: 1.0000
# loss :  0.00012509508815128356
# acc :  1.0
# 1/1 [==============================] - 0s 279ms/step
# accuracy :  0.6666666666666666
# 1/1 [==============================] - 0s 14ms/step
# "개똥이 잘생겼다"의 예측값 :  [[0.31448272]]
# 걸린 시간 :  21.53

# LSTM + ohe
# Epoch 160: ReduceLROnPlateau reducing learning rate to 0.0009999999776482583.
# 10/10 [==============================] - 0s 4ms/step - loss: 0.0017 - acc: 1.0000 - val_loss: 0.5017 - val_acc: 0.5000 - lr: 0.0100
# 1/1 [==============================] - 0s 223ms/step - loss: 0.0016 - acc: 1.0000
# loss :  0.00155320530757308
# acc :  1.0
# 1/1 [==============================] - 0s 232ms/step
# [[0.01067184]
#  [0.9995877 ]
#  [0.00168191]]
# accuracy :  0.6666666666666666
# 1/1 [==============================] - 0s 157ms/step
# "개똥이 잘생겼다"의 예측값 :  [[0.87289596]]
# 걸린 시간 :  8.82