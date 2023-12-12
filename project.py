import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

#Flight Analysis Group Project

# Function to convert travel time to minutes
def convert_time_to_minutes(time_str):
    if isinstance(time_str, str):
        try:
            time_str = time_str.replace('h', ' ').replace('m', ' ').strip()
            hours, minutes = map(int, time_str.split())
            return hours * 60 + minutes
        except ValueError:
            return None
    return None

# Function to convert departure time to minutes since midnight
def convert_departure_time(time_str):
    if isinstance(time_str, str):
        time_str = time_str.replace("a", "AM").replace("p", "PM")
        time_24hr = datetime.strptime(time_str, '%I:%M%p').time()
        return time_24hr.hour * 60 + time_24hr.minute
    return None


def validate_time_format(time_str):
    if pd.isna(time_str):
        return False
    if len(time_str) != 6:
        return False
    hour, minute, meridian = time_str[:2], time_str[3:5], time_str[5]
    if not (hour.isdigit() and minute.isdigit() and meridian in ['a', 'p']):
        return False
    if int(hour) > 23 or int(minute) > 59:
        return False
    return True





# Load the dataset
filepath = '/Users/pragunkalra/Desktop/475project/flight.csv'
flight_data = pd.read_csv(filepath, low_memory=False)

# Renaming the 'Airline name' column to remove extra space, if needed
flight_data.rename(columns={'Airline name ': 'Airline_name'}, inplace=True)

# Selecting relevant columns
relevant_columns = ['Depreture Time', 'Travel  Time', 'Ticket prize(Doller)', 'Airline_name']
trimmed_data = flight_data[relevant_columns]

# Removing rows with missing values
cleaned_data = trimmed_data.dropna()

# Removing rows with non-standard time formats in the "Depreture Time" column

cleaned_data = cleaned_data[cleaned_data['Depreture Time'].apply(validate_time_format)]

# Converting departure time and travel time to minutes
cleaned_data['Departure Time (minutes)'] = cleaned_data['Depreture Time'].apply(convert_departure_time)
cleaned_data['Travel Time (minutes)'] = cleaned_data['Travel  Time'].apply(convert_time_to_minutes)

# Converting the ticket price to numeric
cleaned_data['Ticket Price (USD)'] = pd.to_numeric(cleaned_data['Ticket prize(Doller)'], errors='coerce')

# Dropping the original columns
cleaned_data = cleaned_data.drop(columns=['Depreture Time', 'Travel  Time', 'Ticket prize(Doller)'])

# Removing any rows with null values after conversion
cleaned_data_cleaned = cleaned_data.dropna()

# Display the cleaned data
print(cleaned_data_cleaned.head())


X = cleaned_data_cleaned[['Departure Time (minutes)', 'Travel Time (minutes)']]
y = cleaned_data_cleaned['Ticket Price (USD)']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R^2 Score:", r2)

# Calculating residuals
residuals = y_test - y_pred

# Preparing data for feature importance plot
feature_importances = rf_model.feature_importances_
features = X_train.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
print(feature_importance_df)

# Visualization 1: Feature Importance Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.sort_values(by='Importance', ascending=False))
plt.title('Feature Importance in Random Forest Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Visualization 2: Actual vs Predicted Prices Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=3)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.show()

# Visualization 3: Residuals Plot
plt.figure(figsize=(10, 6))
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 2})
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residuals of Predictions')
plt.show()


print(cleaned_data_cleaned.columns)

# Visualization 4: Price Comparison by Airline
plt.figure(figsize=(12, 6))
sns.boxplot(x='Airline_name', y='Ticket Price (USD)', data=cleaned_data_cleaned)
plt.xticks(rotation=45)
plt.title('Price Comparison by Airline')
plt.ylabel('Ticket Price (USD)')
plt.xlabel('Airline Name')
plt.tight_layout()
plt.show()

# Visualization 5: Ticket Price vs. Travel Time
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Travel Time (minutes)', y='Ticket Price (USD)', data=cleaned_data_cleaned)
plt.title('Ticket Price vs. Travel Time')
plt.xlabel('Travel Time (minutes)')
plt.ylabel('Ticket Price (USD)')
plt.tight_layout()
plt.show()
