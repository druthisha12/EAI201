# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Step 1: Create the dataset
data = pd.DataFrame({
    'Area':     [1200, 1400, 1600, 1700, 1850],
    'Rooms':    [3, 4, 3, 5, 4],
    'Distance': [5, 3, 8, 2, 4],
    'Age':      [10, 3, 20, 15, 7],
    'Price':    [120, 150, 130, 180, 170]
})

# Step 2: Split into features (X) and target (y)
X = data[['Area', 'Rooms', 'Distance', 'Age']]
y = data['Price']

# Step 3: Train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Step 4: Display coefficients and intercept
print("Coefficients (Area, Rooms, Distance, Age):", model.coef_)
print("Intercept:", model.intercept_)

# Step 5: Predict price for a new house
# Example: 1600 sqft, 4 rooms, 5 km distance, 10 years old
new_house = [[1600, 4, 5, 10]]
predicted_price = model.predict(new_house)
print("\nPredicted Price (₹ Lacs):", round(predicted_price[0], 2))

# Step 6: Evaluate model using R² score
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print("\nModel R² Score:", round(r2, 3))

# Step 7 (Optional): Plot actual vs predicted prices
plt.scatter(y, y_pred, color='blue')
plt.xlabel("Actual Price (₹ Lacs)")
plt.ylabel("Predicted Price (₹ Lacs)")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()
