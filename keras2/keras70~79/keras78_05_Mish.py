import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

# def Mish(x):
#     return x * np.tanh(np.log(1 + np.exp(x)))

Mish = lambda x: x * np.tanh(np.log(1 + np.exp(x)))  # lambda 매개변수 : 표현식

y = Mish(x)
plt.plot(x, y)
plt.grid()
plt.show()