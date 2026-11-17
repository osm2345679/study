import numpy as np
aaa = np.array([[-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50],
               [100, 200, -30, 400, 500, 600, -70000, 800, 900, 1000, 210, 420, 350]]).T

def outlier(data):
    quartile_1, q2, quartile_3 = np.percentile(data, [25, 50, 75])
    iqr = quartile_3 - quartile_1
    upper_bound = quartile_3 + iqr * 1.5
    lower_bound = quartile_1 - iqr * 1.5
    return np.where((data < lower_bound) | (data > upper_bound)), \
    iqr, upper_bound, lower_bound

outlier_loc, iqr, up, low = outlier(aaa[:, 0])
print("이상치의 위치 :", outlier_loc)

# import matplotlib.pyplot as plt

# plt.boxplot(aaa[:, 0])
# plt.axhline(up, color='pink', label='upper bound')
# plt.axhline(low, color='pink', label='lower bound')
# plt.legend()
# plt.show()

outlier_loc2, iqr2, up2, low2 = outlier(aaa[:, 1])
print("이상치의 위치 :", outlier_loc2)

# plt.boxplot(aaa[:, 1])
# plt.axhline(up2, color='pink', label='upper bound')
# plt.axhline(low2, color='pink', label='lower bound')
# plt.legend()
# plt.show()

# print(outlier_loc)
# print(type(outlier_loc))

aaa_clipped = np.clip(aaa, [low, low2], [up, up2])
print(aaa_clipped)

aaa = np.where(aaa < low, np.nan, aaa)
print(aaa)