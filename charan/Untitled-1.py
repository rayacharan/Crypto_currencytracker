import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense

# Load the cryptocurrency price data
data = pd.read_csv('cryptocurrency_data.csv')

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Price'].values.reshape(-1, 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_data[:-1], scaled_data[1:], test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential()
model.add(Dense(32, input_dim=1, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# Evaluate the model
train_score = model.evaluate(X_train, y_train, verbose=0)
print('Train loss:', train_score)
test_score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', test_score)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse scale the predictions
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)

# Print the predictions
print('Train Predictions:', train_predictions)
print('Test Predictions:', test_predictions)
