import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

alpha = 0.01

# def leaky_relu(x, alpha): # alpha 인자로 넣어놨지만 보통은 alpha값 0.01로 거의 고정
#     # return np.maximum(alpha * x, x)
#     return np.where(x > 0, x , alpha * x) # 위와 같음

leaky_relu = lambda x, alpha: np.where(x > 0, x, alpha * x)   # lambda 매개변수 : 표현식

y = leaky_relu(x, alpha)
plt.plot(x, y)
plt.grid()
plt.show()