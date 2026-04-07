import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd


# load dataset

data = pd.read_csv("SYKIT_LEArn\student_dataset_50_rows.csv")


# Input & Output
x = data[['study_hours']]
y = data['final_score']

# train model

model = LinearRegression()
model.fit(x, y)
predicted_scores = model.predict(x)

#valid regression metrices

mae = mean_absolute_error(y, predicted_scores)
mse = mean_squared_error(y, predicted_scores)
rmse = np.sqrt(mse)
r2 = r2_score(y, predicted_scores)
# show result

print("Mean Absolute Error (MAE): ", round(mae, 2))
print("Mean Squared Error (MSE): ", round(mse, 2))
print("Root Mean Squared Error(Rmse): ", round(rmse, 2))
print("R^2 Score (Model Accuracy): ", round(r2, 4))  # closer to 1 = better


# hostogram
plt.figure(figsize=(10, 6))
plt.hist(data["final_score"], bins=30, color ='skyblue', edgecolor='black')
plt.title("Distribution of Final Exam Scores")
plt.xlabel("Final Exam Score")
plt.ylabel("Number of Students")
plt.grid(True)
plt.show()