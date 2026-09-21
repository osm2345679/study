import numpy as np
from tensorflow.keras.preprocessing import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, GlobalAveragePooling2D, Flatten
from tensorflow.keras.callbakcs import EarlyStopping, ModelCheckPoint
import time

#1. 데이터
datagen = ImageDataGenerator(
    rescale=1./255
)
path = './_data/image/man_woman'
xy = datagen.flow_from_directory(
    path
)
x = xy[0][0]
y = xy[0][1]

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (5,5), input_size=(100, 100, 3)))
model.add(Conv2D(64, (5,5), activation='relu')),
model.add(Dropout(0.2)),
model.add(Conv2D(64, (5,5), activation='relu')),
model.add(Dropout(0.2)),
model.add(GlobalAveragePooling2D())
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True
)
mcp = ModelCheckPoint(
    
    mode='min'
)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
start_time = time.time()
model.fit(
    x_test, y_test,
    epochs=3,
    batch_size=128,
    verbose=1,
    validation_split=0.25,
    callbacks=[es, mcp]
)
end_time = time.time()
#4. 평가, 예측