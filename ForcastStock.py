"""
利用2016-2022年的全部股价数据，预测2023年全部日期的股票价格
1.用滑动窗口，每天逐步更新新一天的股价
2.从数据库中读取 并且将结果写入数据库中
3.预测5 20 60天的日期

连接基础数据库和预测结果保存数据库
读取基础股价数据
读取时间索引表

股票代码循环

    查询并在预测数据库内建立表格
    读取基础数据中的时间索引

    预测日期（时间索引表）循环2023.1.1-2023.12.31

        读取从开始至当前时间索引的股价表格

        根据股价表格数据 利用SARIMAX，LSTM，TimeGPT进行未来 5 20 60 天的股价预测

        将预测的结果写入预测数据库

"""
import numpy as np
import pandas as pd
import pymysql
from dbutils.pooled_db import PooledDB
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
import csv
import sys
from nixtlats import TimeGPT
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
import tensorflow as tf
import argparse
from pandas.tseries.offsets import BDay


########################################################################################################################
# 预测算法部分
#########################################################################################################################


# SARIMAX预测函数，分别预测未来5 20 60天的数据 也就是周 月 季度的股价数据
def SARIMAX_forecast(stock_base, predict_day):
    # TODO 这个地方还需要改进，未来的日期只是连续未来的日期，并没有考虑节假日，要设定为未来多少天的交易日比较好
    # TODO 然后也许可以添加一些经济数据作为预测的外生性变量

    # 拟合模型的过程
    model = SARIMAX(stock_base['Close'], order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), trend='t')
    results = model.fit(disp=False)

    # Make predictions for the next set days
    forecast = results.forecast(predict_day)
    return forecast



# TimeGPT预测函数，分别预测未来5 20 60天的数据 也就是周 月 季度的股价数据
def TimeGPT_forecast(stock_base, predict_day):
    # 创建TimeGPT对象，传入token参数，如果没有提供则默认使用环境变量中的TIMEGPT_TOKEN
    timegpt = TimeGPT(token='0tMWwfvjBodRpWuyY3DPtG5mqyge4ZtWDD6zVPFsMENwERSEueBA0mui8Mxdr9M4hUBBRCOvpdYRP8z49m90mRq2rWeVU7WYbmorSS7dV0DP7cfKLJcOQfgcYmW5Tr7laxzPa2iLK9uqobDfh5TBxRk9tnTEAxVnOrBvzgSaD8xshMQktjGbALzma4g3Kx6XzDLSgePzeMBqxN9w4v7uZ3h7Ea2EtJARS2CNEFt5kKIEAhICwYqVD3lEcawaYGUa')
    # 调用timegpt模块的validate_token()函数进行token验证
    timegpt.validate_token()

    forecast = timegpt.forecast(
        df=stock_base, h=predict_day, freq='B',
        time_col='Date', target_col='Close'
        # 可以有这个level得出区间概率
        # level=[80, 90],
    )

    forecast = forecast['TimeGPT']

    return forecast


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
    model.fit(trainX, trainY, epochs=50, batch_size=64, validation_data=(testX, testY), verbose=1)

    # 预测未来股价
    forecast_price = model.predict(testX)
    # 反归一化预测数据到原始范围
    forecast_price = scaler.inverse_transform(forecast_price)

    # 创建一个 DataFrame
    forecast = pd.DataFrame(forecast_price)

    return forecast


def GBM_forecast(price_base, predict_days=5, mu=0.05, sigma=0.2):
    # 获取基准价格
    base_price = price_base['Close'].iloc[-1]

    # 计算时间间隔
    dt = 1  # 假设时间间隔为1天

    # 模拟路径
    price_paths = []
    for _ in range(predict_days):
        # 生成随机数
        rand = np.random.normal(0, 1)
        # 计算价格变化
        delta_price = mu * base_price * dt + sigma * base_price * np.sqrt(dt) * rand
        # 更新股票价格
        base_price += delta_price
        # 将价格添加到路径中
        price_paths.append(base_price)

        price_forecast = pd.DataFrame(price_paths)

    return price_forecast


########################################################################################################################
# 数据库信息读写部分
#########################################################################################################################


# 读取基础数据库中的股价和日期数据
def read_db_base(cursor, stock_code, start_time, end_time):
    try:
        # 将输入的时间类型预先转换
        start_time = datetime.strptime(start_time, "%Y-%m-%d")
        end_time = datetime.strptime(end_time, "%Y-%m-%d")

        query = f"""
        #
        # 
            SELECT Date, Close
            FROM {stock_code}_base
            WHERE Date >= %s and Date <= %s       #读取时间开始前的数据
            ORDER BY Date ASC;
        """
        cursor.execute(query, (start_time, end_time,))  # 参数化查询 防止SQL注入
        result = cursor.fetchall()

        # 将查询结果转换为 DataFrame
        df = pd.DataFrame(result, columns=['Date', 'Close'])

        return df
    except Exception as e:
        print(f"Error fetching date for {stock_code}: {e}")
        return None


# TODO 目前不知道是要用这种365天动态的还是放一个列里面存字符串集合全部的价格，然后还是就60天的放5 20 60的数据，但是那个自增的表格我得研究一下，可能不能插入统样时间股价的数据
def create_tables(cursor, stock_code, predict_day):
    # 动态生成每一天的列名和数据类型
    day_columns = ', '.join([f'Day{i} FLOAT' for i in range(1, predict_day + 1)])

    create_table_query = f"""
   -- 创建预测数据表
   -- {predict_day}天的情况  
CREATE TABLE IF NOT EXISTS {stock_code}_predict_{predict_day}d (
    Date DATE DEFAULT NULL COMMENT '股价日期',
    Stock VARCHAR(255) DEFAULT NULL COMMENT '股票代码',
    PredictDate DATETIME DEFAULT NULL COMMENT '预测日期',
    PredictDay INT DEFAULT NULL COMMENT '预测未来的天数',
    Algorithm VARCHAR(255) DEFAULT NULL COMMENT '预测算法',
    Close FLOAT DEFAULT NULL COMMENT '收盘价格',
    PredictAvg FLOAT DEFAULT NULL COMMENT '预测股价的平均值',
    PredictMax FLOAT DEFAULT NULL COMMENT '预测股价的最大值',
    PredictMin FLOAT DEFAULT NULL COMMENT '预测股价的最小值',
    PredictStd FLOAT DEFAULT NULL COMMENT '预测股价的标准差',
    PredictProfit FLOAT DEFAULT NULL COMMENT '预测区间的最大收益率 %',
    PredictLoss FLOAT DEFAULT NULL COMMENT '预测区间的最大亏损率 %',
    PredictSlope FLOAT DEFAULT NULL COMMENT '预测序列的斜率',
    PredictIntercept FLOAT DEFAULT NULL COMMENT '预测序列的截距',
    {day_columns},
    UNIQUE KEY `date` (`Date` DESC) USING BTREE
);
    """
    cursor.execute(create_table_query)



# 将预测好的结果写入数据库中
def update_forecast(cursor, db_connection, stock_code, date_close, forecast, algorithm, predict_day):

    # 检测要预测的天数是否为5 20 60 过滤一下，才能写入数据库
    if predict_day == 60 or predict_day == 20 or predict_day == 5:
        print("")
    else:
        raise ValueError("Invalid forecast day. Please choose '60d', '20d', or '5d'.")

    # 解析thisday数据，也就是数据表中最后一天的时间和价格
    row = date_close.iloc[0]
    thisday_date = row['Date'].strftime('%Y-%m-%d')  # 确保日期格式正确
    thisday_close = row['Close']

    # 获取当前日期作为预测日期
    predict_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 将预测值的精度限制在小数点4位数，防止数据库精度不够不让写入
    forecast_values_list = [float(np.format_float_positional(f, precision=4)) for f in forecast.values]

    # 计算预测值的统计指标
    predict_max = np.max(forecast_values_list)
    predict_min = np.min(forecast_values_list)
    predict_avg = np.mean(forecast_values_list)
    predict_std = np.std(forecast_values_list)

    # 计算预测最大收益率和预测最大亏损率
    predict_profit = (predict_max - thisday_close) / thisday_close
    predict_loss = (predict_min - thisday_close) / thisday_close

    # 计算斜率和截距
    days = np.arange(1, predict_day + 1)
    predict_slope, predict_intercept = np.polyfit(days, forecast_values_list, 1)  # 1表示线性回归，返回斜率和截距

    # 动态构建SQL语句的列名部分 根据预定的时间动态生成SQL所要的天数
    columns = ', '.join([f'Day{i}' for i in range(1, predict_day + 1)])
    placeholders = ', '.join(['%s'] * predict_day)  # 为每一天的预测数据准备占位符

    # SQL语句模板，动态包括60天的字段
    sql = f"""
    #
    #
    INSERT INTO {stock_code}_predict_{predict_day}d
    (Date, Stock, PredictDate, PredictDay, Algorithm, Close, {columns}, PredictAvg, PredictMax, PredictMin, PredictStd,
    PredictProfit, PredictLoss, PredictSlope, PredictIntercept)
    VALUES (%s, %s, %s, %s, %s, %s, {placeholders}, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
    PredictDate = VALUES(PredictDate),
    PredictDay = VALUES(PredictDay),
    Algorithm = VALUES(Algorithm),
    Close = VALUES(Close),
    {', '.join([f'Day{i} = VALUES(Day{i})' for i in range(1, predict_day + 1)])},
    PredictAvg = VALUES(PredictAvg),
    PredictMax = VALUES(PredictMax),
    PredictMin = VALUES(PredictMin),
    PredictStd = VALUES(PredictStd),
    PredictProfit = VALUES(PredictProfit),
    PredictLoss = VALUES(PredictLoss),
    PredictSlope = VALUES(PredictSlope),
    PredictIntercept = VALUES(PredictIntercept);
    """

    # 准备SQL语句的参数，包括统计指标
    params = (
        thisday_date,
        stock_code,
        predict_date,
        predict_day,
        algorithm,
        thisday_close,
        *forecast_values_list[:predict_day],  # 确保列表长度不超过day
        predict_avg,
        predict_max,
        predict_min,
        predict_std,
        predict_profit,
        predict_loss,
        predict_slope,
        predict_intercept  # 添加截距
    )

    # 执行SQL语句
    cursor.execute(sql, params)

    # 提交到数据库
    db_connection.commit()


# 创建连接池函数 可以复用
def create_db_pool(host, port, user, password, db_name):
    return PooledDB(
        creator=pymysql,
        maxconnections=10,
        host=host,
        port=port,
        user=user,
        password=password,
        database=db_name
    )

########################################################################################################################
# 主要函数操作逻辑部分
#########################################################################################################################


# 根据输入算法类别输入预测数据调用相应的预测算法，并返回预测结果
def do_forecast(predict_day, rolling_stock, algorithm):
    # 此处是算法预测的封装
    if algorithm == "SARIMAX":
        forecast = SARIMAX_forecast(rolling_stock, predict_day)
    elif algorithm == "LSTM":
        forecast = LSTM_forecast(rolling_stock, predict_day)
    elif algorithm == "TimeGPT":
        forecast = TimeGPT_forecast(rolling_stock, predict_day)
    elif algorithm == "GBM":
        forecast = GBM_forecast(rolling_stock, predict_day)
    else:
        raise ValueError("Unsupported algorithm specified")
    return forecast

"""
从数据库中获取并准备股票数据。返回整个所需数据集的时间索引和股价数据
参数:
    - cursor: 数据库游标。
    - stock_code: 股票代码。
    - start_time: 数据获取的开始时间。
    - predict_time: 预测开始时间。
    - end_time: 数据获取的结束时间。
返回:
    - time_index: 时间索引数据，不包含股价。
    - all_stock_data: 所有股票数据，包含日期和股价。
"""
def fetch_prepare_data(cursor, stock_code, start_time, predict_time, end_time):

    # 从基础数据数据库中读取指定时间范围内的股票数据，返回时间索引数据
    time_index = read_db_base(cursor, stock_code, predict_time, end_time)
    time_index = pd.DataFrame(time_index)

    # 将获取的数据转换为pandas DataFrame，并移除'Close'列，因为这里只需要日期信息
    time_index.drop('Close', axis=1, inplace=True)
    time_index['Date'] = pd.to_datetime(time_index['Date'])

    # 读取指定时间范围内的全部股票数据
    all_stock_data = read_db_base(cursor, stock_code, start_time, end_time)
    all_stock_data = pd.DataFrame(all_stock_data)
    all_stock_data['Date'] = pd.to_datetime(all_stock_data['Date'])

    return time_index, all_stock_data


"""
本质上就是先读取整个时间范围内的股价和时间轴数据，然后就从预测开始时间到结束时间一天一天加入循环并生成预测数据的结果在写入到数据库中
"""
# 执行预测的do函数，输入predict_day，algorithm，还有时间自动拉取数据执行预测函数并讲预测结果存到数据库中
# predict_day 就是要预测的时间步长 5 20 60 分别代表5天 20天 60天
# algorithm 就是要使用的预测算法，SARIMAX LTSM TimeGPT GBM
# start_time 整个训练数据集开始的时间，predict_time 要预测的数据开始的时间，end_time 预测结束的时间
# 2016-01-01——2019-01-01——2023-12-31  即从2016年的数据开始读取，第一次预测是读取2016-2019的数据预测2019-01-01的结果，然后数据集和预测时间逐渐递增直到end_time
def do(predict_day, algorithm, start_time, predict_time, end_time, stock_code, basedata_pool, forecast_pool):
    # 打印当前正在执行的预测任务的相关信息，包括股票代码、使用的算法、预测的天数、预测的时间范围等。
    print(f"Predicting for {stock_code} using {algorithm} for {predict_day} days from {start_time} to {end_time} with prediction at {predict_time}")

    # 使用with语句管理数据库连接和游标，确保资源正确关闭
    with basedata_pool.connection() as db_connection_basedata, db_connection_basedata.cursor() as cursor_basedata:
        # 通过封装的函数获取并准备数据
        time_index, all_stock_data = fetch_prepare_data(cursor_basedata, stock_code, start_time, predict_time, end_time)

    # 对每个在时间索引中的日期进行循环处理。
    for index in range(len(time_index)):
        # 获取当前循环到的日期。
        rolling_day = time_index.iloc[index]['Date']
        # 根据当前日期筛选出到这一天为止的所有股票数据，用于模型的输入。
        rolling_stock = all_stock_data[all_stock_data['Date'] <= rolling_day]
        # 获取筛选数据的最后一行，包含最新的股票价格和日期，用于后续的数据记录和输出。
        date_close = rolling_stock.tail(1)
        # 打印该日期的股票数据。
        print("\n", date_close)

        # 初始化预测结果为一个指定长度的、元素为-1的序列，以处理可能的异常情况。
        forecast = pd.Series([-1] * predict_day, dtype='float')

        try:
            # 根据指定的算法调用对应的预测函数进行股价预测
            forecast = do_forecast(predict_day, rolling_stock, algorithm)

        except Exception as e:
            # 如果预测过程中发生异常，将异常信息记录到CSV文件中，并在控制台输出错误。
            with open('exceptions_log.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([stock_code, date_close, predict_day, str(e)])
            print(f"{algorithm} Exception for stock {stock_code} on date {date_close} for {predict_day} days forecast: {e}", file=sys.stderr)

        # 使用with语句连接预测结果数据库，将预测结果存储进去。
        with forecast_pool.connection() as db_connection_forecast, db_connection_forecast.cursor() as cursor_forecast:
            # 如果必要的表格不存在，则创建。
            create_tables(cursor_forecast, stock_code, predict_day)
            # 更新数据库，将预测结果写入。
            update_forecast(cursor_forecast, db_connection_forecast, stock_code, date_close, forecast, algorithm, predict_day)

########################################################################################################################
# 主函数参数提取部分
#########################################################################################################################

def parse_arguments():
    """解析命令行参数并返回解析后的参数."""
    parser = argparse.ArgumentParser(description='Stock prediction script')
    parser.add_argument('--days', type=int, help='Number of days for prediction')
    parser.add_argument('--algorithm', type=str, default='LSTM', help='Prediction algorithm')
    parser.add_argument('--start', type=str, help='Start date')
    parser.add_argument('--predict', type=str, help='Prediction date')
    parser.add_argument('--end', type=str, help='End date')
    parser.add_argument('--stock_code', type=str, help='Stock code')

    # 添加数据库连接参数
    parser.add_argument('--db_host', type=str, default='101.37.77.232', help='Database host address')
    parser.add_argument('--db_port', type=int, default=8903, help='Database port number')
    parser.add_argument('--db_user', type=str, default='lizhuolin', help='Database user name')
    parser.add_argument('--db_password', type=str, default='123456', help='Database password')
    parser.add_argument('--db_name_basedata', type=str, default='basedata', help='Database name for base data')
    parser.add_argument('--db_name_forecast', type=str, default='forecast_gbm', help='Database name for forecast data')

    return parser.parse_args()


if __name__ == "__main__":
    # 解析参数
    args = parse_arguments()

    # 创建数据库连接池
    basedata_pool = create_db_pool(args.db_host, args.db_port, args.db_user, args.db_password, args.db_name_basedata)
    forecast_pool = create_db_pool(args.db_host, args.db_port, args.db_user, args.db_password, args.db_name_forecast)

    # 执行主要的业务逻辑 do函数
    do(args.days, args.algorithm, args.start, args.predict, args.end, args.stock_code, basedata_pool, forecast_pool)

    # 关闭连接池
    basedata_pool.close()
    forecast_pool.close()