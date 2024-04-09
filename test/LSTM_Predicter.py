# prompt: 利用amzn_base里面的Close利用LSTM方法预测未来30天的价格
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

predict_days = 360
look_back = 60

# Read the first excel file
price_base = pd.read_xlsx("amzn_base.csv")

# Convert the second column to datetime format
price_base['Date'] = pd.to_datetime(price_base['Date'], format="%d/%m/%Y")
#price_base = price_base[:-predict_days]
#price_gt = price_base[-predict_days:]
#print(price_base['Date'])

# Assuming 'amzn_base' is already loaded and contains the 'Date' and 'Close' columns

# Select the 'Close' column
close_prices = price_base['Close'].values.reshape(-1, 1)
training_set = close_prices[0:-predict_days]
testing_set = close_prices[-predict_days:]

# Scale the 'Close' prices
scaler = MinMaxScaler(feature_range=(0, 1))
training_set= scaler.fit_transform(training_set)
testing_set = scaler.transform(testing_set)
# Function to create dataset for LSTM
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(look_back,len(dataset)):
        a = dataset[i-look_back:i, 0]
        X.append(a)
        Y.append(dataset[i, 0])
    return np.array(X), np.array(Y)

# Prepare the data with a look_back period

trainX, trainY = create_dataset(training_set, look_back)
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

testX, testY = create_dataset(testing_set, look_back)
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# Create and fit the LSTM network
# model = Sequential()
# model.add(LSTM(50, input_shape=(look_back, 1), return_sequences=True))
# model.add(LSTM(50))
# model.add(Dense(1))
#model.compile(optimizer='adam', loss='mean_squared_error')
from keras.layers import Dropout, Dense, LSTM
import tensorflow as tf
model = tf.keras.Sequential([
    LSTM(80, return_sequences=True),   # 定义了一个具有80个记忆体的LSTM层，这一层会在每个时间步输出ht
    Dropout(0.2),
    LSTM(100),   # 定义了一个具有100个记忆体的LSTM层，这一层仅在最后一个时间步输出ht
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error')


# Fit model with early stopping
#early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
history = model.fit(trainX, trainY, epochs=50, batch_size=64,validation_data=(testX, testY))
loss = history.history['loss']
val_loss = history.history['val_loss']

# loss可视化
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training  Loss')
plt.legend()
plt.show()

# # Making predictions
# def predict_next_days(model, last_days_scaled):
#     prediction_list = last_days_scaled
#     for _ in range(predict_days):
#         x = prediction_list[-look_back:]
#         x = x.reshape((1, look_back, 1))
#         out = model.predict(x)[0][0]
#         prediction_list = np.append(prediction_list, out)
#     prediction_list = prediction_list[look_back:]
#     return prediction_list

# # Predict the next 30 days
# last_days_scaled = scaled_prices[-look_back:]
# predicted_next = predict_next_days(model, last_days_scaled)
# predicted_next = scaler.inverse_transform(predicted_next.reshape(-1, 1))

predicted_stock_price = model.predict(testX)
# 对预测数据还原---从（0，1）反归一化到原始范围
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
# 对真实数据还原---从（0，1）反归一化到原始范围
real_stock_price = scaler.inverse_transform(testing_set[look_back:])
# print('real_stock_price',real_stock_price.shape)
# print('predicted_stock_price',predicted_stock_price.shape)

plt.plot(real_stock_price, color='red', label='Target Stock Price')   # 真实值曲线
plt.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')  # 预测值曲线
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
mse = mean_squared_error(predicted_stock_price, real_stock_price)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse = math.sqrt(mean_squared_error(predicted_stock_price, real_stock_price))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae = mean_absolute_error(predicted_stock_price, real_stock_price)

print('均方误差: %.6f' % mse)
print('均方根误差: %.6f' % rmse)
print('平均绝对误差: %.6f' % mae)

