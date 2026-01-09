import pandas as pd
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("dataset/winequality.csv")
X = df.drop(columns=["quality"])
y = df["quality"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = ElasticNet(alpha=0.5, l1_ratio=0.3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R²:", r2)

joblib.dump(model, "outputs/model.pkl")

results = {"mse": mse, "r2": r2}
with open("outputs/results.json", "w") as f:
    json.dump(results, f, indent=4)
