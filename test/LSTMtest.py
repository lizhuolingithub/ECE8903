import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
import tensorflow as tf





"""
Forecast future stock prices using LSTM model.
# LSTM预测函数，分别预测未来5 20 60天的数据 也就是周 月 季度的股价数据

参数:
stock_data (DataFrame): 包含历史股价的DataFrame
forecast_days (int): 预测未来的天数
look_back (int): 要考虑进行预测的过去天数（默认值：60）

返回:
forecast_price (n-维数组): Forecasted stock prices for the specified number of days
"""
def LSTM_forecast(price_base, predict_days, look_back=60):

    # 专为LSTM对预测数组的长度等于predict_days减去look_back，所以在这块要加一下，符合调用的逻辑
    predict_days = predict_days + look_back

    print(price_base)
    print("\npredict_days",predict_days)

    # 将日期字符串转换为 datetime 对象
    price_base['Date'] = pd.to_datetime(price_base['Date'], format="%Y-%m-%d")

    # 选择 'Close' 列并将其转换为适合模型的形式
    close_prices = price_base['Close'].values.reshape(-1, 1)
    training_set = close_prices[0:-predict_days]
    testing_set = close_prices[-predict_days:]

    # 对 'Close' 价格进行归一化处理
    scaler = MinMaxScaler(feature_range=(0, 1))
    training_set = scaler.fit_transform(training_set)
    testing_set = scaler.transform(testing_set)

    # 函数：根据 look_back 参数创建数据集
    def create_dataset(dataset, look_back):
        X, Y = [], []
        for i in range(look_back, len(dataset)):
            a = dataset[i - look_back:i, 0]
            X.append(a)
            Y.append(dataset[i, 0])
        return np.array(X), np.array(Y)

    # 准备带有 look_back 期的数据
    trainX, trainY = create_dataset(training_set, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX, testY = create_dataset(testing_set, look_back)
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    # 构建 LSTM 模型
    model = Sequential([
        LSTM(80, return_sequences=True, input_shape=(trainX.shape[1], 1)),
        Dropout(0.2),
        LSTM(100),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error')

    # 训练模型
    history = model.fit(trainX, trainY, epochs=50, batch_size=64, validation_data=(testX, testY), verbose=1)

    # 预测未来股价
    forecast_price = model.predict(testX)
    # 反归一化预测数据到原始范围
    forecast_price = scaler.inverse_transform(forecast_price)

    return forecast_price



def do():

    price_base = pd.read_csv("amzn_base.csv")
    forecast = LSTM_forecast(price_base, 60)
    print(forecast)

do()