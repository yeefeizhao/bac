from sklearn.linear_model import LinearRegression
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv("application_data.csv")
#print(df)

X = df[["x0", "x1", "x2", "x3", "x4"]]
y = df["y"]

model = LinearRegression()
model.fit(X, y)

coefficients = model.coef_
intercept = model.intercept_

print("Linear Regression Formula:")
formula = "y = "
for i, coef in enumerate(coefficients):
    formula += f"{coef:.2f} * x{i} + "
formula += f"{intercept:.2f}"
print(formula)