from tensorflow.keras.preprocessing.image import ImageDataGenerator

#1. 데이터

# 데이터 가져온 후

datagen = ImageDataGenerator(rescale=1./255)

augumented_size = 100

datagen.flow()

#2. 모델 구성

#3. 컴파일, 훈련

#4. 평가, 예측