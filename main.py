import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -------------------------------
# LOAD DATASET
# -------------------------------
df = pd.read_csv("sales.csv")

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year

# Convert categorical features
df["holiday"] = df["holiday"].map({"Yes": 1, "No": 0})
df["promotion"] = df["promotion"].map({"Yes": 1, "No": 0})

# -------------------------------
# FEATURES & TARGET
# -------------------------------
X = df[["day", "month", "year", "holiday", "promotion", "temperature", "fuel_price"]]
y = df["sales"]

# -------------------------------
# MODEL TRAINING
# -------------------------------
model = LinearRegression()
model.fit(X, y)

# -------------------------------
# FUTURE PREDICTION
# -------------------------------
future_dates = pd.date_range(start=df["date"].max(), periods=6)[1:]

future_df = pd.DataFrame({"date": future_dates})

future_df["day"] = future_df["date"].dt.day
future_df["month"] = future_df["date"].dt.month
future_df["year"] = future_df["date"].dt.year

# Assume future conditions
future_df["holiday"] = 0
future_df["promotion"] = 1
future_df["temperature"] = 30
future_df["fuel_price"] = 85

# Predict
predictions = model.predict(
    future_df[["day", "month", "year", "holiday", "promotion", "temperature", "fuel_price"]]
)

# -------------------------------
# PRINT OUTPUT
# -------------------------------
print("\n📊 Future Sales Prediction:")
for date, pred in zip(future_df["date"], predictions):
    print(f"{date.date()} → {int(pred)}")

# -------------------------------
# VISUALIZATION
# -------------------------------
plt.figure()

# Actual sales
plt.plot(df["date"], df["sales"], label="Actual Sales")

# Forecasted sales
plt.plot(future_df["date"], predictions, linestyle="--", label="Forecast")

plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Forecasting")
plt.legend()

plt.show()
