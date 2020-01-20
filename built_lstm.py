# Building an LSTM

# Part 1: Data Preprocessing 

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Training Set (as a dataframe)
dataset_train = pd.read_csv('Full_Google_Training_Data.csv')

# Creating numpy array (which is how we have to deal with our data) of daily opening values 
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling 
from sklearn.preprocessing import MinMaxScaler

# Creating an object of the class MinMaxScaler to normalize data 
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output. This means at any given moment the RNN
# is looking at the trend of the past 60 financial days (one item of numpy array X_train) to 
# predict the next day (1 item of numpy array y_train).
X_train = []
y_train = []

for i in range (60, 3828):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping (can add extra dimensions of indicators to make a more robust financial prediction)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2: Building the RNN

# Importing Keras Libraries and Packages
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising RNN (called regressor because it is doing regression, predicting a continuous value)
regressor = Sequential() 

# Layer 1 of LSTM  
# Adding first LSTM layer to RNN with 50 neurons that will return sequences to the next layer in the input 
# shape of our input data in numpy array X_train
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
# Adding dropout regularisation to first layer (to reduce overfitting)
regressor.add(Dropout(0.2))

# Layer 2
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Layer 3
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Layer 4 (does not return sequences because it is the last layer of the LSTM)
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training Set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


# Part 3: Making Predictions and Visualizing Results 

# Getting the actual stock prices of November 2019
dataset_test = pd.read_csv('Full_Google_Test_Data.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock prices of November 2019

# Creating a combined dataframe of our test and training data (because some of our test data will require
# earlier test data days to make a stock price prediction). Axis = 0 means the data is combined vertically
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

# Numpy array containing all days we need for our prediction, starting 60 financial days before the first 
# financial day of November and going until the second last stock price in the dataset 
# (in the proper shape).
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1) 
inputs = sc.transform(inputs)

# Creating the proper data structure we will need to input into our regressor to make predictions
X_test = []
for i in range (60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Making the stock price prediction and converting it back into the correct scale of its actual values
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# Visualizing the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

# Calculating Error
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

relative_error = rmse/1302.348492

relative_error
