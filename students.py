
import pandas as pd 
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , r2_score
df = pd.read_csv("student_performance_dataset.csv")
x = df[[
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Score",
    "Practice_Tests"
]]
y = df["Final_Score"]
x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)
model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
plt.scatter(y_test,y_pred)
plt.xlabel("Actual Scores")
plt.ylabel("Expected Scores")
plt.show()
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print(f"MSE: {mse} \n R2 Score : {r2} \n{model.coef_} \n{model.intercept_}")
new_stud = pd.DataFrame({
    "Hours_Studied": [7],
    "Attendance": [85],
    "Sleep_Hours": [8],
    "Previous_Score": [75],
    "Practice_Tests": [4]
})
prediction = model.predict(new_stud)
print(prediction)
