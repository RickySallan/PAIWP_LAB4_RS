import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load your dataset
file_path1 = '/workspaces/PAIWP_LAB4_RS/DailyDelhiClimateTest.csv'
file_path2 = '/workspaces/PAIWP_LAB4_RS/DailyDelhiClimateTrain.csv'
df = pd.read_csv(file_path1, parse_dates=['date'])

# Assuming the dataset has columns: 'date', 'meantemp', 'humidity', 'wind_speed', 'meanpressure'
# Prepare the dataset for time series analysis
df.set_index('date', inplace=True)

# Fill missing values, if any
df.fillna(method='ffill', inplace=True)

# Feature Engineering - Create more features, like day, month, year
df['day'] = df.index.day
df['month'] = df.index.month
df['year'] = df.index.year

# Target variable
y = df['meantemp']

# Feature set
X = df.drop('meantemp', axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize ensemble models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Train the models
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# Make predictions
rf_predictions = rf_model.predict(X_test)
gb_predictions = gb_model.predict(X_test)

# Evaluate the models
rf_mse = mean_squared_error(y_test, rf_predictions)
gb_mse = mean_squared_error(y_test, gb_predictions)

print(f'Random Forest MSE: {rf_mse}')
print(f'Gradient Boosting MSE: {gb_mse}')


##############################################

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split

# Load your dataset
file_path2 = '/workspaces/PAIWP_LAB4_RS/DailyDelhiClimateTrain.csv'
df = pd.read_csv(file_path2, parse_dates=['date'])
df.set_index('date', inplace=True)

# Fill missing values
df.fillna(method='ffill', inplace=True)

# Decompose the series (optional, for EDA)
decomposition = seasonal_decompose(df['meantemp'], model='additive', period=365)
decomposition.plot()

# Feature Engineering - Create lag features, if necessary
for lag in range(1, 4):  # you can adjust the number of lags
    df[f'meantemp_lag{lag}'] = df['meantemp'].shift(lag)

# Drop any NaN values created by lagging
df.dropna(inplace=True)

# Define your target and features
y = df['meantemp']
X = df.drop('meantemp', axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a SARIMA model
# Note: The order (p, d, q) and seasonal_order (P, D, Q, s) need to be defined. These are typically found through grid search.
model = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima = model.fit()

# Make predictions
predictions = sarima.predict(start=X_test.index[0], end=X_test.index[-1])

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'SARIMA Model MSE: {mse}')

