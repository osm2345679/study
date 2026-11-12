import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

# SELU 고정 상수 (논문 제안값)
alpha = 0.01

# def leaky_relu(x, alpha):
#     # return np.maximum(alpha * x, x)
#     return np.where(x > 0, x , alpha*x) # 위와 같음

leaky_relu = lambda x, alpha : np.where(x > 0, x, alpha * x)   # lambda 매개변수 : 표현식

y = leaky_relu(x, alpha)
plt.plot(x, y)
plt.grid()
plt.show()