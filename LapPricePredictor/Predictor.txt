import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load the dataset
url = "https://github.com/campusx-official/laptop-price-predictor-regression-project/raw/main/laptop_data.csv"
Laptop = pd.read_csv(url)

# Define target variable and drop specified columns to get features
y = Laptop['Price']
x = Laptop.drop(['Unnamed: 0', 'Company', 'TypeName', 'ScreenResolution', 'Cpu', 'Ram', 'Memory', 'Gpu', 'OpSys', 'Weight', 'Price'], axis=1)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=2529)

# Create and train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict function for new input
def predict_laptop_price(new_input):
    # Convert the new input into a DataFrame
    new_input_df = pd.DataFrame([new_input], columns=x.columns)
    # Use the trained model to predict the price
    predicted_price = model.predict(new_input_df)
    return predicted_price[0]

# Example new input based on remaining features
new_input = {
    'Inches': 13.3,
    'Weight': 1.37,
    'Cpu': 'Intel Core i5 2.3GHz',
    'Ram': 8,
    'Memory': '128GB SSD',
    'Gpu': 'Intel Iris Plus Graphics 640'
}

# Predict the price for the new input
predicted_price = predict_laptop_price(new_input)
print("Predicted Laptop Price:", predicted_price)

# Make predictions on the test set
y_pred = model.predict(x_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)  # Calculate RMSE

# Print evaluation metrics
print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)

# Plotting predicted vs actual prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Actual vs Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3, color='red', label='Ideal Prediction')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Laptop Prices')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()