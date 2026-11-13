# 2017년 구글에서 발표한 swish
# 정확히는 SiLu라고 발표한 연구팀이 있는데 구글에서도 비슷한 시기에 swish라는 이름으로 발표함
# 더 정확히는 SiLu는 x * sigmoid() 인데 swish는 x * z * sigmoid()로 파라미터가 존재함
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

def SiLu(x):
    return x * (1 / (1 + np.exp(-x)))

# SiLu = lambda x: x * (1 / (1 + np.exp(-x)))  # lambda 매개변수 : 표현식

y = SiLu(x)
plt.plot(x, y)
plt.grid()
plt.show()