from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Зареждане на данни
iris = load_iris()

X = iris.data
y = iris.target

# Обучение
model = RandomForestClassifier()
model.fit(X, y)

# Запазване
joblib.dump(model, "model.pkl")

print("Model saved!")