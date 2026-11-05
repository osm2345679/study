from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
import numpy as np

#1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y
)

learning_rate = [0.1, 0.05, 0.01, 0.005, 0.001]
max_depth = [3, 4, 5, 6, 7]
best_score = 0
best_prameters = ''

for i, lr in enumerate(learning_rate):
    for j, md in enumerate(max_depth):
        model = HistGradientBoostingClassifier(
            learning_rate=lr,
            max_depth=md
        )
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)

        if score > best_score:
            best_score = score
            best_prameters = ("learing rate :", lr, "max_depth :", md)
        print(i, ",", j, "번째 score :", round(score, 4), "도는 중...")

print("최고 점수 : {:.2f}".format(best_score))
print("최적 파라미터 :", best_prameters)

# Results
# 최고 점수 : 0.97
# 최적 파라미터 : ('learing rate :', 0.1, 'max_depth :', 5)