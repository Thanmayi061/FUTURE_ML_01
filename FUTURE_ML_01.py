# ================================
# 1️⃣ Import Libraries
# ================================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ================================
# 2️⃣ Load Dataset
# ================================
df = pd.read_csv("data.csv", encoding="latin1")

# ================================
# 3️⃣ Data Cleaning
# ================================
df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])

df = df.dropna()  # Properly drop missing values
df = df.sort_values(by='ORDERDATE')  # Properly sort

# ================================
# 4️⃣ Monthly Aggregation
# ================================
df['Month'] = df['ORDERDATE'].dt.to_period('M')

monthly_sales = df.groupby('Month')['SALES'].sum().reset_index()

monthly_sales['Month'] = monthly_sales['Month'].dt.to_timestamp()

# ================================
# 5️⃣ Visualize Monthly Trend
# ================================
plt.figure(figsize=(10,5))
plt.plot(monthly_sales['Month'], monthly_sales['SALES'])
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.title("Monthly Sales Trend")
plt.show()

# ================================
# 6️⃣ Feature Engineering
# ================================
monthly_sales['year'] = monthly_sales['Month'].dt.year
monthly_sales['month'] = monthly_sales['Month'].dt.month
monthly_sales['time_index'] = range(len(monthly_sales))  # Trend feature

X = monthly_sales[['year','month','time_index']]
y = monthly_sales['SALES']

# ================================
# 7️⃣ Train-Test Split (Time-Based)
# ================================
split = int(len(monthly_sales)*0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# ================================
# 8️⃣ Train Model
# ================================
model = LinearRegression()
model.fit(X_train, y_train)

# ================================
# 9️⃣ Predictions
# ================================
y_pred = model.predict(X_test)

# ================================
# 🔟 Model Evaluation
# ================================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Model Performance:")
print("MAE:", round(mae,2))
print("RMSE:", round(rmse,2))

# ================================
# 1️⃣1️⃣ Actual vs Predicted Plot
# ================================
plt.figure(figsize=(10,5))
plt.plot(monthly_sales['Month'][:split], y_train, label="Train Data")
plt.plot(monthly_sales['Month'][split:], y_test, label="Actual")
plt.plot(monthly_sales['Month'][split:], y_pred, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted Sales")
plt.show()

# ================================
# 1️⃣2️⃣ Future Forecast (Next 6 Months)
# ================================

future_months = pd.date_range(
    start=monthly_sales['Month'].max() + pd.offsets.MonthEnd(1),
    periods=6,
    freq='M'
)

future_df = pd.DataFrame({
    'year': future_months.year,
    'month': future_months.month,
    'time_index': range(len(monthly_sales), len(monthly_sales)+6)
})

future_sales = model.predict(future_df)

# ================================
# 1️⃣3️⃣ Future Forecast Visualization
# ================================
plt.figure(figsize=(10,5))
plt.plot(monthly_sales['Month'], monthly_sales['SALES'], label='Historical')
plt.plot(future_months, future_sales, label='Future Forecast', linestyle='dashed')
plt.legend()
plt.title("Future Sales Forecast (Next 6 Months)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()

# ================================
# 1️⃣4️⃣ Print Future Predictions
# ================================
future_results = pd.DataFrame({
    'Month': future_months,
    'Predicted Sales': future_sales
})

print("\nFuture Sales Forecast:")
print(future_results)
