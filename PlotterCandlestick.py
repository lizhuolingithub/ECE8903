"""
GT-ECE8903 24spring StockPlot
Zhuolin Li

Plot绘图的脚本为收集到的股价信息绘制静态图像看板
1.连接并读取两个数据库基础信息和策略信息（输入时间和股票代码）
2.合并两个表当中的数据成为一个表
3.分割表格当中的所有列存在临时变量当中
4.使用这些临时变量绘制并输出静态的股票看板
"""
from apscheduler.schedulers.background import BackgroundScheduler
import yfinance as yf
import pandas as pd
import pymysql
import schedule
import time
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import mplfinance as mpf


#  读取预测数据库中的股价和对应数据
def read_db_forecast_day(cursor, stock_code, predict_day, algorithm, start_time, end_time):
    # 检测要预测的天数
    valid_days = ["60d", "20d", "5d"]
    if predict_day not in valid_days:
        raise ValueError(f"Invalid forecast day. Please choose from {valid_days}.")

    # 提取天数作为整数
    num_days = int(predict_day.replace('d', ''))

    # 使用提取的天数来生成列名
    day_columns = ', '.join([f'Day{i}' for i in range(1, num_days + 1)])

    try:
        # 更新SQL查询语句只查询对应时间最晚那一天的预测序列
        query = f"""
        -- 
        --  
        SELECT {day_columns}
        FROM {stock_code}_predict_{predict_day}
        WHERE Date >= %s AND Date <= %s AND Algorithm = %s
        ORDER BY Date DESC
        LIMIT 1;
        """

        cursor.execute(query, (start_time, end_time, algorithm))  # 参数化查询，防止SQL注入
        result = cursor.fetchall()

        # 将查询结果转换为 DataFrame 并转置
        df = pd.DataFrame(result, columns=[f'Day{i}' for i in range(1, num_days + 1)])
        df_transposed = df.T
        df_transposed.reset_index(inplace=True)
        df_transposed.columns = ['Date', 'Value']

        # 生成工作日序列
        business_days = pd.bdate_range(start=end_time, periods=num_days)

        # 将'Day'列的值设置为工作日序列
        df_transposed['Date'] = business_days.strftime('%Y-%m-%d')

        return df_transposed

    except Exception as e:
        print(f"Error fetching data for {stock_code}: {e}")
        return None


def read_db_combined(cursor, stock_code, start_time, end_time, algorithm='SARIMAX'):
    try:
        # 确保时间格式正确（防止SQL注入）
        start_time = start_time.strftime("%Y-%m-%d")
        end_time = end_time.strftime("%Y-%m-%d")

        # 确定需要查询的字段
        base_columns = ['Date', 'Stock', 'Algorithm',  'Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']

        strategy_columns = ['FastAvg', 'SlowAvg', 'MACD', 'SignalLine', 'MA', 'BollingerUp', 'BollingerDown',
                            'RSI', 'RSIchannel', 'Doji', 'ADX', 'MACDsign', 'OBV']

        """                   
        strategy_columns = ['FastAvg', 'SlowAvg', 'MACD', 'SignalLine', 'BollingerUp', 'BollingerDown', 
                            'RSI', 'RSIchannel', 'Doji', 'ADX', 'MACDsign', 'Channel', 'K', 'D', 'CCI', 'ROC', 
                            'WilliamsR', 'OBV', 'Klinger', 'CMF', 'CandleIndi']
                            
        forecast_columns5d = ['PredictMax', 'PredictMin', 'PredictAvg', 'PredictStd', 'PredictProfit', 'PredictLoss', 'PredictSlope', 'PredictIntercept']
        """

        forecast_columns5d = ['PredictMax5d', 'PredictMin5d', 'PredictAvg5d', 'PredictStd5d', 'PredictProfit5d',
                              'PredictLoss5d', 'PredictSlope5d', 'PredictIntercept5d']

        forecast_columns20d = ['PredictMax20d', 'PredictMin20d', 'PredictAvg20d', 'PredictStd20d', 'PredictProfit20d',
                               'PredictLoss20d', 'PredictSlope20d', 'PredictIntercept20d']

        forecast_columns60d = ['PredictMax60d', 'PredictMin60d', 'PredictAvg60d', 'PredictStd60d', 'PredictProfit60d',
                               'PredictLoss60d', 'PredictSlope60d', 'PredictIntercept60d']

        # 构建查询字段字符串
        selected_columns = ', '.join(
            base_columns + strategy_columns + forecast_columns5d + forecast_columns20d + forecast_columns60d)

        # 构建SQL查询
        query = f"""
        -- 
        -- 
        SELECT {selected_columns}
        FROM {stock_code}_combined
        WHERE Date >= %s AND Date <= %s AND Algorithm = %s
        ORDER BY Date ASC;
        """

        # 准备查询参数
        params = (start_time, end_time, algorithm)

        cursor.execute(query, params)  # 执行查询
        result = cursor.fetchall()  # 获取所有结果

        # 创建DataFrame
        all_columns = base_columns + strategy_columns + forecast_columns5d + forecast_columns20d + forecast_columns60d
        df = pd.DataFrame(result, columns=all_columns)

        return df
    except Exception as e:
        print(f"Error fetching data for {stock_code}: {e}")
        return None



"""
准备数据，将'Date'列转换为datetime格式并设置为索引，扩展表格以包括预测的日期，
本质上就是将历史时间和预测数据的时间轴填充并对齐，互相缺失的部分就赋空值，因为绘图函数里面的变量时间轴必须要统一
参数:
    table (DataFrame): 原始的股票数据表。
    forecast_day (DataFrame): 包含预测日期的表格。
    predict_day (str): 预测天数，例如 '5d', '20d', '60d'。
返回:
    tuple: 返回扩展后的table和forecast_day。
"""
def prepare_data(table, forecast_day, predict_day):

    # 确保'Date'列是datetime格式，并设置为索引
    table['Date'] = pd.to_datetime(table['Date'])
    table.set_index('Date', inplace=True)

    # 计算添加的天数，从predict_day字符串中提取天数部分
    days_to_add = int(predict_day[:-1])  # 从 "5d" 中提取 5

    # 生成新的日期范围，这些日期是表中最后一个日期后的工作日
    new_dates = pd.bdate_range(start=table.index.max() + pd.Timedelta(days=1), periods=days_to_add, freq='B')

    # 创建新数据行，初始化为 NaN
    new_data = pd.DataFrame(index=new_dates, columns=table.columns)
    new_data.bfill(inplace=True)  # 使用向后填充来处理缺失值

    # 合并原始表和新行
    extended_table = pd.concat([table, new_data])

    # 转换 forecast_day 的 'Date' 为 datetime，并设置为索引
    forecast_day['Date'] = pd.to_datetime(forecast_day['Date'])
    forecast_day.set_index('Date', inplace=True)
    forecast_day = forecast_day.reindex(extended_table.index)

    return extended_table, forecast_day




"""
使用mplfinance绘制一个全面的股票分析图表。
参数:
    table (DataFrame): 包含带指标的历史股票数据。
    forecast (DataFrame): 包含即将到来的日子的预测值。
    stock_name (str): 股票的名称或股票代码。
    start_date (datetime.date): 股票数据的开始日期。
    day (str): 预测范围（例如 '5d', '20d', '60d'），天数表示要添加的工作日数量。
    color (str): 预测线条的颜色方案。
"""
def plotter(table, forecast, stock_name, day, color):


    # 定义图表的额外元素
    apds = [
        mpf.make_addplot(table['MA'], panel=0, color='green', width=1.0),
        mpf.make_addplot(table['BollingerUp'], panel=0,color='red', width=1.0),
        mpf.make_addplot(table['BollingerDown'], panel=0, color='blue', width=1.0),
        mpf.make_addplot(table['MACD'], panel=1, color='green', width=0.75),
        mpf.make_addplot(table['SignalLine'], panel=1, color='orange', width=0.75),
        mpf.make_addplot(table['RSI'], panel=2, color='navy', ylabel='RSI', width=0.5, ylim=(0, 100)),
        mpf.make_addplot(table['OBV'], panel=3, color='darkred', secondary_y=False, ylabel='OBV', width=0.5),
        mpf.make_addplot(forecast['Value'], panel=0, color=color, width=1.25, linestyle='-.', ylabel='Forecast')
    ]

    # 添加预测的最小值和最大值线到主面板
    apds.extend([
        mpf.make_addplot(table[f'PredictMin{day}'], panel=0, color=color, width=0.75, linestyle='--'),
        mpf.make_addplot(table[f'PredictMax{day}'], panel=0, color=color, width=0.75, linestyle='--')
    ])

    # 设置图表样式
    style = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'figure.figsize': (24, 18)})

    # 设置标题
    title = f"Stock {stock_name} {start_date} to {end_date} with {algorithm} forecast-{day}ay"

    # 根据上述参数绘制图表
    fig, axes = mpf.plot(table, type='candle', addplot=apds, style=style, volume=True,
             figratio=(24, 18), figscale=1.5, title=title, returnfig=True, panel_ratios=(6, 3, 2, 2))

    # 添加图例
    axes[0].legend(['', 'Stock', 'MA', 'BollingerUp', 'BollingerDown', f'{day}Forecast', f'{day}_Min', f'{day}_Max'], loc='upper left')
    axes[1].legend(['MACD', 'SignalLine'], loc='upper left')

    # 保存图像到文件
    plt.savefig(f'PlotterCandlestick/stock_analysis_{day}.svg', dpi=1200)
    plt.show()


if __name__ == "__main__":

    # 选好的股票代码 16个精选的股票和ETF基金指数
    stock_codes = ['AMZN', 'BBH', 'FAS', 'GLD', 'GOOG', 'IVV', 'IWD', 'IWM', 'LABU', 'MS', 'QQQ', 'SPY', 'TQQQ', 'VGT',
                   'VUG', 'XLE']

    # 设置数据库的参数 连接数据库的信息（连接待生成策略数据的数据库）
    db_config = {"host": "10.5.0.11", "port": 3306, "user": "lizhuolin", "password": "123456", "database": "forecast"}

    # "host": "10.5.0.11", "port": 3306, "user": "lizhuolin", "password": "123456"
    # "host": "8.147.99.223", "port": 3306, "user": "lizhuolin", "password": "&a3sFD*432dfD!o0#3^dP2r2d!sc@"

    predict_days = ['5d', '20d', '60d']
    colors = ['blue', 'green', 'red']

    # 默认基础调试信息
    stock_code = "XLE"
    algorithm = ("LSTM")
    start_date = datetime.strptime("2020-1-1", "%Y-%m-%d").date()
    end_date = datetime.strptime("2020-6-30", "%Y-%m-%d").date()

    # 连接读取预测数据库信息
    db_connection_forecast = pymysql.connect(**db_config)  # 创建连接
    cursor_forecast = db_connection_forecast.cursor()  # 创建连接

    for predict_day, color in zip(predict_days, colors):

        # 读取融合视图的数据
        combined_table = read_db_combined(cursor_forecast, stock_code, start_date, end_date, algorithm)

        # 读取预测值的数据
        forecast_table = read_db_forecast_day(cursor_forecast, stock_code, predict_day, algorithm, start_date, end_date)

        # 调用数据准备函数，本质上就是将历史时间和预测数据的时间轴填充并对齐，互相缺失的部分就赋空值
        extended_table, forecast_table = prepare_data(combined_table, forecast_table, predict_day)

        # 将读取的数据输入进去绘图
        plotter(extended_table, forecast_table, stock_code, predict_day, color)

        print(f"\n  {predict_day}任务结束，已经将股票看板绘制完毕\n")


    # 关闭数据库连接
    cursor_forecast.close()
    db_connection_forecast.close()


    """
    print("\nstock_code :")
    for stock_code in stock_codes:
        print(",", stock_code)
    # 输入绘图界面参数基本交互
    stock_name = input("which stock wanna to plot：")
    day_before = int(input("how many the stock data to plot："))

    start_date = (date.today() - timedelta(days=day_before))
    """
