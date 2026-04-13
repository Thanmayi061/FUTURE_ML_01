import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load dataset
df = pd.read_csv("sales.csv")

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Create time-based features
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year

# Features & target
X = df[["day", "month", "year"]]
y = df["sales"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict future (next 5 days)
future_days = pd.date_range(start=df["date"].max(), periods=6)[1:]

future_df = pd.DataFrame({
    "date": future_days
})

future_df["day"] = future_df["date"].dt.day
future_df["month"] = future_df["date"].dt.month
future_df["year"] = future_df["date"].dt.year

predictions = model.predict(future_df[["day", "month", "year"]])

# Print predictions
print("Future Sales Prediction:")
for date, pred in zip(future_df["date"], predictions):
    print(date.date(), "→", int(pred))

# Plot graph
plt.figure()
plt.plot(df["date"], df["sales"], label="Actual Sales")
plt.plot(future_df["date"], predictions, linestyle="--", label="Forecast")

plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Forecasting")
plt.legend()

plt.show()
