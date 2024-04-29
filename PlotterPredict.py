"""
读取预测数据库中的数据然后用于画图
"""
import argparse
import pandas as pd
import pymysql
from datetime import datetime
import matplotlib.pyplot as plt



"""
Fetches forecast data from the combined view for a specified stock and algorithm
between the start_time and end_time.
"""
#  读取预测数据库中的股价和对应数据
def read_db_forecast(cursor, stock_code, algorithm, start_time, end_time):

    try:
        # Convert the input time strings to datetime objects
        start_time = datetime.strptime(start_time, "%Y-%m-%d")
        end_time = datetime.strptime(end_time, "%Y-%m-%d")

        # SQL query to retrieve data from the combined view
        query = f"""
        -- 
        --  
        SELECT Date, Close, Stock, Algorithm,
               PredictAvg5d, PredictMax5d, PredictMin5d, PredictStd5d,
               PredictMax20d, PredictMin20d, PredictStd20d,
               PredictMax60d, PredictMin60d, PredictStd60d
        FROM {stock_code}_combined
        WHERE Date >= %s AND Date <= %s AND Algorithm = %s
        ORDER BY Date ASC;
        """
        cursor.execute(query, (start_time, end_time, algorithm))
        result = cursor.fetchall()

        # Convert query results into a DataFrame
        columns = ['Date', 'Close',  'Stock', 'Algorithm',
                   'PredictAvg5d', 'PredictMax5d', 'PredictMin5d', 'PredictStd5d',
                   'PredictMax20d', 'PredictMin20d', 'PredictStd20d',
                   'PredictMax60d', 'PredictMin60d', 'PredictStd60d']
        df = pd.DataFrame(result, columns=columns)
        return df
    except Exception as e:
        print(f"Error fetching data for {stock_code} using {algorithm}: {e}")
        return None



def plot_individual_forecast(df, predict_day, algorithm, color, stock_code):
    # 确保Date列是datetime类型，以便正确绘制
    df['Date'] = pd.to_datetime(df['Date'])

    # 设置图的尺寸
    plt.figure(figsize=(16, 8))

    # 绘制实际收盘价，去除点，线条更细
    plt.plot(df['Date'], df['Close'], label='Actual Close', color='black', linewidth=1, linestyle='-')

    # 绘制预测的最小值和最大值曲线
    plt.fill_between(df['Date'], df[f'PredictMin{predict_day}'], df[f'PredictMax{predict_day}'], color=color, alpha=0.3,label=f'{predict_day} Forecast Range')

    # 图表标题和图例
    plt.title(f'{algorithm} {predict_day} Forecast Min/Max Range for {stock_code}', fontsize=16)
    plt.legend(loc='upper left')

    # 设置x轴和y轴标签
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price', fontsize=14)

    # 美化x轴日期显示
    plt.gcf().autofmt_xdate()

    # 保存图片
    plt.savefig(f'PlotterPredict/{algorithm}_{stock_code}_{predict_day}.svg')

    plt.grid(True)
    plt.show()


def plotter(algorithm, start_time, predict_time, end_time, stock_code):

    # 连接预测数据库
    db_connection_forecast = pymysql.connect(**db_config_forecast)
    cursor_forecast = db_connection_forecast.cursor()

    # Reading forecast data 读取融合视图里面的数据
    stock_forecast = read_db_forecast(cursor_forecast, stock_code, algorithm, predict_time, end_time)
    if stock_forecast is None:
        print(f"Failed to retrieve forecast data for {stock_code} using {algorithm}.")
        return  # Exit the function if no data is fetched 做一个异常处理，如果里面为空的话就报错

    # 关闭连接
    cursor_forecast.close()
    db_connection_forecast.close()

    # 将 5 20 60 天的预测min max 范围图变为 蓝 绿 红
    predict_days = ['5d', '20d', '60d']
    colors = ['blue', 'green', 'red']

    # 用for循环准备画图
    for predict_day, color in zip(predict_days, colors):
        plot_individual_forecast(stock_forecast, predict_day, algorithm, color, stock_code)



def parse_arguments():
    """解析命令行参数并返回解析后的参数."""
    parser = argparse.ArgumentParser(description='Stock prediction script')
    parser.add_argument('--algorithm', type=str, default='SARIMAX', help='Prediction algorithm')
    parser.add_argument('--start', type=str, default='2016-01-01', help='Start date')
    parser.add_argument('--predict', type=str, default='2019-01-01', help='Prediction date')
    parser.add_argument('--end', type=str, default='2023-12-31', help='End date')
    parser.add_argument('--stock_code', type=str, default='xle', help='Stock code')

    return parser.parse_args()


if __name__ == "__main__":

    # 设置数据库的参数 连接数据库的信息（连接预测信息数据的数据库）
    db_config_forecast = {"host": "10.5.0.11", "port": 3306, "user": "lizhuolin", "password": "123456", "database": "forecast"}

    args = parse_arguments()
    plotter(args.algorithm, args.start, args.predict, args.end, args.stock_code)
