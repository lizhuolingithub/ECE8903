"""
读取预测数据库中的数据然后用于画图
"""
import argparse
import os
import pandas as pd
import pymysql
from datetime import datetime
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import matplotlib.dates as mdates


#  读取预测数据库中的Date，Close，Transaction字段
def read_db(cursor, stock_code, start_time, end_time):
    """
    Fetch Date, Close, and Transaction fields from the database for the specified stock_code and date range.
    """
    query = f"""
    SELECT `Date`, `Close`, `Transaction`
    FROM {stock_code}_trade
    WHERE `Date` BETWEEN '{start_time}' AND '{end_time}'
    ORDER BY `Date`;
    """
    cursor.execute(query)
    data = cursor.fetchall()
    return pd.DataFrame(data, columns=['Date', 'Close', 'Transaction'])

# 根据交易策略模拟交易计算收益率
def trade_simulation(df):
    """
    Simulate trades based on Transaction signals and calculate the cumulative return.

    Args:
    df (DataFrame): DataFrame that contains the Date, Close prices, and Transaction signals.

    Returns:
    DataFrame: Updated DataFrame with additional columns for portfolio values and returns.
    """
    initial_capital = 1000000  # Initial capital in dollars
    shares = 0
    capital = initial_capital
    portfolio_values = []

    # Ensure transactions are in lowercase to avoid case sensitivity issues
    df['Transaction'] = df['Transaction'].str.lower()

    for i, row in df.iterrows():
        print(f"Date: {row['Date']}, Close: {row['Close']}, Transaction: {row['Transaction']}")

        if row['Transaction'] == 'buy' and capital >= row['Close']:
            # Calculate the number of shares to buy
            shares_to_buy = capital // row['Close']
            shares += shares_to_buy
            capital -= shares_to_buy * row['Close']
            print(f"Bought {shares_to_buy} shares, remaining capital: {capital}")

        elif row['Transaction'] == 'sell' and shares > 0:
            # Sell all shares
            capital += shares * row['Close']
            print(f"Sold {shares} shares, new capital: {capital}")
            shares = 0

        # Record the portfolio value for this day
        portfolio_value = capital + shares * row['Close']
        portfolio_values.append(portfolio_value)

    df['Portfolio Value'] = portfolio_values
    df['Returns'] = df['Portfolio Value'] / initial_capital - 1  # Calculate returns relative to initial capital

    return df


"""
Plots and saves the returns over time for the given stock_code, both displaying and saving the plot as an SVG file.

Args:
    df (DataFrame): DataFrame containing the trading data including 'Date' and 'Returns'.
    stock_code (str): Stock code to label the plot and name the file.

Enhancements in this version:
    - Display the plot on screen for immediate viewing.
    - Save the plot in SVG format for high-quality reproduction.
    - Beautify the plot with grid, formatted date labels, and a descriptive legend.
 """
#  将得出来的结果绘制成收益率时间图像
def plotter(df, stock_code):
    save_dir = "PlotterTrade"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Setup the figure and axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')  # 设置整个画布的背景颜色为白色

    # Plotting the returns on the primary y-axis
    color = 'tab:red'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Returns (%)', color=color)
    ax1.plot(df['Date'], df['Returns'] * 100, color=color, label='Returns (%)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)  # 网格线更为细致、半透明

    # Create a second y-axis for the close prices
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Close Price', color=color)
    ax2.plot(df['Date'], df['Close'], color=color, label='Close Price')
    ax2.tick_params(axis='y', labelcolor=color)

    # Formatting the x-axis with date format
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    plt.xticks(rotation=45)

    # Adding title and legend
    plt.title(f'Stock Trading Performance for {stock_code}')
    fig.tight_layout()  # adjust subplots to give some room for the labels

    # Save the figure
    file_path = os.path.join(save_dir, f"{stock_code}_performance.svg")
    plt.savefig(file_path, format='svg', dpi=300)
    plt.show()

    # Close the plot to free up memory
    plt.close()
    print(f"Plot saved as {file_path}")




if __name__ == "__main__":

    # 选好的股票代码 16个精选的股票和ETF基金指数
    stock_codes = ['AMZN', 'BBH', 'FAS', 'GLD', 'GOOG', 'IVV', 'IWD', 'IWM', 'LABU', 'MS', 'QQQ', 'SPY', 'TQQQ', 'VGT',
                   'VUG', 'XLE']

    # 设置数据库的参数 连接数据库的信息（连接待生成策略数据的数据库）
    db_config = {"host": "10.5.0.11", "port": 3306, "user": "lizhuolin", "password": "123456", "database": "trade"}

    # "host": "10.5.0.11", "port": 3306, "user": "lizhuolin", "password": "123456"
    # "host": "8.147.99.223", "port": 3306, "user": "lizhuolin", "password": "&a3sFD*432dfD!o0#3^dP2r2d!sc@"


    start_date = datetime.strptime("2019-1-1", "%Y-%m-%d").date()
    end_date = datetime.strptime("2023-12-31", "%Y-%m-%d").date()

    # 连接读取预测数据库信息
    db_connection = pymysql.connect(**db_config)  # 创建连接
    cursor = db_connection.cursor()  # 创建连接

    for stock_code in stock_codes:
        df = read_db(cursor, stock_code, start_date, end_date)

        df['Date'] = pd.to_datetime(df['Date'])  # Ensure Date is in datetime format for plotting

        df_trade = trade_simulation(df)

        plotter(df, stock_code)


    # 关闭数据库连接
    cursor.close()
    db_connection.close()

