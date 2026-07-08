import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl

data = pd.read_csv("cost_revenue_clean.csv")
x = pd.DataFrame(data, columns=['production_budget_usd'])
y = pd.DataFrame(data, columns=['worldwide_gross_usd'])
regression = skl.linear_model.LinearRegression()
regression.fit(x, y)
coefficient = regression.coef_
intercept = regression.intercept_
print(type(coefficient), type(intercept))
budget = int(input("Enter your Film Budget: "))
prediction = intercept + (coefficient * budget)
print("\nPrediction : ", int(prediction.item()))
score = regression.score(x, y)
print("Score : ", score)



