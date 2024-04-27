"""
GT-ECE8903 24spring
Zhuolin Li

"""
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
import yfinance as yf
import pandas as pd
import pymysql
import schedule
import time
from datetime import datetime, timedelta
import time
from apscheduler.schedulers.background import BackgroundScheduler

########################################################################################################################
# 数据库读写建表操作部分
#########################################################################################################################

# 初始的待生成策略数据各股票表建表函数，检测如果数据库内无对应表格则执行该函数
def create_tables(cursor, stock_code):
    create_table_query = \
        f"""
        CREATE TABLE IF NOT EXISTS {stock_code}_trade (
        `Date` DATE NOT NULL COMMENT '交易日期',
        `Close` FLOAT COMMENT '收盘价格',
        `Transaction` VARCHAR(50) COMMENT '最终交易决定：buy, sell, hold',
        `Returns` FLOAT COMMENT '根据买卖交易决定计算出来的收益率',
        `BuyComment` VARCHAR(255) COMMENT '买信号的位置',
        `SellComment` VARCHAR(255) COMMENT '卖信号的位置',
        `StateA` INT COMMENT '布林带状态：-3超出上端，-2 80-100%区间，-1 60-80%区间，0 40-60%区间，1 20-40%区间，2 0-20%区间，3超出下端',
        `StateB` INT COMMENT 'MACD信号：1 MACD下穿上信号线买，0 无穿越，-1 MACD上穿下信号线卖',
        `StateC` INT COMMENT 'Doji类型：1 蜻蜓Doji，0 普通或长脚Doji，-1 墓碑Doji',
        `StateD` INT COMMENT '复杂的Doji模式：2 晨曦之星且布林带低于40%买，1 三线打击，0 没有doji，-1 两只乌鸦，-2 黄昏之星或三只乌鸦且布林带高于60%卖',
        `StateE` INT COMMENT 'Stochastic Oscillator KD线：1 K线从下穿上D线买，0 无穿越，-1 K线从上穿下D线卖',
        `StateF` INT COMMENT 'RSI通道：2 0%-20%买，1 20%-40%，0 40%-60%，-1 60%-80%，-2 80%-100%卖',
        `StateG` INT COMMENT '商品通道指数CCI：2 CCI<-200买，1 CCI<-100，0 -100<CCI<100，-1 CCI>100，-2 CCI>200卖',
        `StateH` INT COMMENT '威廉姆斯%R：1 %R<-80，0 -80<%R<-20，-1 -20<%R',
        `Point` INT COMMENT '上述状态分数之和',
        `Buy` INT COMMENT '是否有买点 1表示有，0表示无',
        `Sell` INT COMMENT '是否有卖点 1表示有，0表示无',
        `ForecastAlgorithm` VARCHAR(255) COMMENT '预测未来价格的算法',
        `Profit5day` FLOAT COMMENT '5天的预测收益%',
        `Loss5day` FLOAT COMMENT '5天的预测损失%',
        `Profit20day` FLOAT COMMENT '20天的预测收益%',
        `Loss20day` FLOAT COMMENT '20天的预测损失%',
        `Profit60day` FLOAT COMMENT '60天的预测收益%',
        `Loss60day` FLOAT COMMENT '60天的预测损失%',
        PRIMARY KEY (`Date`),
        UNIQUE KEY `Date` (`Date`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT= '交易表记录各类技术指标和交易决策';
    """
    cursor.execute(create_table_query)


# 读取原始预测的数据库里面的视图的数据
def read_db(cursor, stock_code, start_time, algorithm):
    try:
        formatted_time = start_time.strftime("%Y-%m-%d")  # 确保时间格式正确（防止SQL注入）
        # SQL查询语句，选择特定字段和条件
        query = f"""
            SELECT 
                Date, Close, PredictProfit5d, PredictLoss5d, PredictProfit20d, PredictLoss20d, PredictSlope20d,
                PredictProfit60d, PredictLoss60d, PredictSlope60d, BollingerChannel, MACDsign, Doji, 
                RSIChannel, ComplexDoji, KDsign, CCI, WilliamsR, OBV, OBV20ma, OBV2de
            FROM 
                {stock_code}_combined
            WHERE 
                Date >= %s AND Algorithm = %s AND Stock = %s
            ORDER BY 
                Date ASC;
        """
        # 执行参数化查询，防止SQL注入
        cursor.execute(query, (formatted_time, algorithm, stock_code))  # 参数化查询
        result = cursor.fetchall()

        # 将查询结果转换为 DataFrame，并指定列名
        df = pd.DataFrame(result, columns=[
            'Date', 'Close', 'PredictProfit5d', 'PredictLoss5d', 'PredictProfit20d', 'PredictLoss20d', 'PredictSlope20d',
            'PredictProfit60d', 'PredictLoss60d', 'PredictSlope60d', 'BollingerChannel', 'MACDsign', 'Doji',
            'RSIChannel', 'ComplexDoji', 'KDsign', 'CCI', 'WilliamsR', 'OBV', 'OBV20ma', 'OBV2de'
        ])

        return df
    except Exception as e:
        print(f"Error fetching {stock_code}_combined: {e}")


# 将计算好的数据更新到交易数据表中
def update_db(stock_code, stock_table_df, gotime, cursor, db_connection):
    # 构建SQL插入语句，包括更新已存在记录的操作
    insert_query = f"""
        INSERT INTO {stock_code}_trade
        (Date, Close, StateA, StateB, StateC, StateD, StateE, StateF, StateG, StateH, Point, Buy, Sell, BuyComment, SellComment, ForecastAlgorithm, 
         Profit5day, Loss5day, Profit20day, Loss20day, Profit60day, Loss60day, Transaction, Returns)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        Close=VALUES(Close),
        StateA=VALUES(StateA),
        StateB=VALUES(StateB),
        StateC=VALUES(StateC),
        StateD=VALUES(StateD),
        StateE=VALUES(StateE),
        StateF=VALUES(StateF),
        StateG=VALUES(StateG),
        StateH=VALUES(StateH),
        Point=VALUES(Point),
        Buy=VALUES(Buy),
        Sell=VALUES(Sell),
        BuyComment=VALUES(BuyComment),
        SellComment=VALUES(SellComment),
        ForecastAlgorithm=VALUES(ForecastAlgorithm), 
        Profit5day=VALUES(Profit5day),
        Loss5day=VALUES(Loss5day),
        Profit20day=VALUES(Profit20day),
        Loss20day=VALUES(Loss20day),
        Profit60day=VALUES(Profit60day),
        Loss60day=VALUES(Loss60day),
        Transaction=VALUES(Transaction),
        Returns=VALUES(Returns)
    """

    # 准备数据列表用于批量插入
    data_to_insert = []
    for index, row in stock_table_df.iterrows():
        index_date = index.date() if isinstance(index, datetime) else index
        if index_date >= gotime:  # 只有日期在gotime之后的数据才被插入
            data_tuple = (index_date, row['Close'], row['A'], row['B'], row['C'], row['D'],
                          row['E'], row['F'], row['G'], row['H'], row['Point'], row['Buy'],
                          row['Sell'],row['BuyComment'], row['SellComment'], row['ForecastAlgorithm'], row['Profit5day'],
                          row['Loss5day'], row['Profit20day'], row['Loss20day'], row['Profit60day'], row['Loss60day'],
                          row['Transaction'], row['Returns'])
            data_to_insert.append(data_tuple)

    # 执行批量插入
    if data_to_insert:
        cursor.executemany(insert_query, data_to_insert)
        db_connection.commit()  # 一次性提交所有插入操作，提高效率


########################################################################################################################
# 数据时间控制部分
#########################################################################################################################

# 获得数据库当中最新和最老的日期，以备后续使用
def get_date_range(cursor, stock_code, suffix):
    try:
        # 构建查询最早和最晚日期的SQL语句
        query = f"SELECT MIN(Date), MAX(Date) FROM {stock_code}_{suffix}"

        # 执行查询
        cursor.execute(query)
        result = cursor.fetchone()
        if result:
            oldest_date = result[0] if result[0] is not None else None
            latest_date = result[1].strftime("%Y-%m-%d") if result[1] is not None else None
        else:
            oldest_date, latest_date = None, None
        # 返回一个包含最早和最晚日期的元组
        return oldest_date, latest_date

    except Exception as e:
        print(f"Error fetching date range for {stock_code}_{suffix}: {e}")
        return None, None


# 根据基础信息数据库和提取信息数据库的最老和最新时间来确定开始更新的时间
def get_update_time(cursor_read, cursor_update, read_suffix, update_suffix, stock_code, default):
    """
    根据基础信息数据库和提取信息数据库的最老和最新时间来确定更新提取信息的开始时间
    为了解决读取表和写入表的数据同步问题，防止多读取数据，也防止少读取数据
    更新提取信息的脚本要同时连接读取信息和写入信息两个数据库
    主要实现的逻辑：
    读取表最老时间 > 写入表最老时间（或者为空）---从基础最老时间开始更新
    读取表最新时间 < 写入表最新时间 - --从提取最新时间开始更新
    """

    oldest_read, latest_read = get_date_range(cursor_read, stock_code, read_suffix)
    oldest_update, latest_update = get_date_range(cursor_update, stock_code, update_suffix)


    if oldest_read is None:
        # 如果读取表的时间为空的话则返回默认时间，默认时间实际上就是人为设定的最早时间
        return datetime.strptime(default, "%Y-%m-%d").date()

    elif oldest_update is None or oldest_read > oldest_update:
        # 如果读取表的最老时间更旧，从读取表最老时间开始更新
        return oldest_read

    elif latest_update is not None and latest_read <= latest_update:
        # 如果写入表的最新时间不为空且早于读取表的最新时间，从写入表的最新时间减去一天开始更新
        return datetime.strptime(latest_update, "%Y-%m-%d").date() - timedelta(days=1)

    else:
        # 遇上其他情况，如读取表时间被写入表完全覆盖时，则从最新的时间谁最小减去一天开始更新
        return datetime.strptime(min(latest_update, latest_read), "%Y-%m-%d").date() - timedelta(days=1)


########################################################################################################################
# 主要功能数据计算处理部分
#########################################################################################################################

"""
计算并添加复杂Doji状态到DataFrame中。
参数-df : pd.DataFrame 需要包含 `ComplexDoji` 和 `BollingerPct` 列的DataFrame。
返回-df : pd.DataFrame  更新后的DataFrame，包含新的列 `D`。
"""
def calculate_complex_doji(df):

    # Defining conditions
    conditions = [
        (df['ComplexDoji'] == 5),  # 三线打击
        (df['ComplexDoji'] == 6) & (df['BollingerChannel'] < 2),  # 晨曦之星，布林带低于40%
        (df['ComplexDoji'] == 0),  # 没有doji
        ((df['ComplexDoji'] == 1) | (df['ComplexDoji'] == 4)) & (df['BollingerChannel'] > 3),  # 黄昏之星或两只乌鸦，布林带高于60%
        (df['ComplexDoji'] == 4),  # 三只乌鸦
    ]

    # Corresponding choices for each condition
    choices = [
        3,   # 三线打击
        2,   # 晨曦之星 and 布林带低于40% 买
        0,   # 没有doji
        -2,  # 黄昏之星 or 两只乌鸦
        -3   # 三只乌鸦
    ]

    # Apply conditions to determine the 'C' column
    df['C'] = np.select(conditions, choices, default=0)  # Default to 0 (no action) if no conditions are met

    return df['C']


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
        # print(f"Date: {row['Date']}, Close: {row['Close']}, Transaction: {row['Transaction']}")

        if row['Transaction'] == 'buy' and capital >= row['Close']:
            # Calculate the number of shares to buy
            shares_to_buy = capital // row['Close']
            shares += shares_to_buy
            capital -= shares_to_buy * row['Close']
            # print(f"Bought {shares_to_buy} shares, remaining capital: {capital}")

        elif row['Transaction'] == 'sell' and shares > 0:
            # Sell all shares
            capital += shares * row['Close']
            # print(f"Sold {shares} shares, new capital: {capital}")
            shares = 0

        # Record the portfolio value for this day
        portfolio_value = capital + shares * row['Close']
        portfolio_values.append(portfolio_value)

    df['Portfolio Value'] = portfolio_values
    df['Returns'] = df['Portfolio Value'] / initial_capital - 1  # Calculate returns relative to initial capital

    return df['Returns']

# 根据各个指标状态和设定好的买卖点条件计算出买卖点的个数
def add_trade_signals(df):
    # 定义买入条件
    buy_conditions = [
        df['A'] == 1,  # 1 macd下穿上信号线 买
        df['B'] == 1,  # 2 晨曦之星 and 布林带低于40% 买
        df['C'] == 1,  # 1 K线从下穿上D线 买
        df['D'] == 2,  # 2 RSIChannel 0%-20% 买
        df['E'] == 2   # 2 CCI<-200 买
    ]

    # 定义卖出条件
    sell_conditions = [
        df['A'] == -1,  # -1 macd上穿下信号线 卖
        df['B'] == -1,  # -2 黄昏之星 三只乌鸦 and 布林带高于60% 卖
        df['C'] == -1,  # -1 K线从上穿下D线 卖
        df['D'] == -2,  # -2 RSIChannel 80%-100% 卖
        df['E'] == -2  # -2 CCI>200 卖
    ]

    # 使用np.where对每个条件进行判断，生成为1的布尔值矩阵，然后沿着行的方向求和
    df['Buy'] = np.sum([np.where(condition, 1, 0) for condition in buy_conditions], axis=0)

    # 方法一样，是对卖的信号进行求和
    df['Sell'] = np.sum([np.where(condition, 1, 0) for condition in sell_conditions], axis=0)

    # 返回包含'Buy'和'Sell'的新DataFrame
    return df[['Buy', 'Sell']]

# 根据预定义的买入条件，计算并添加买入信号的注释字符串到DataFrame中。
def add_trade_signals_comments(df):
    """
    Update the DataFrame with comments based on trade signal conditions.

    Args:
    df (DataFrame): DataFrame containing the necessary columns for evaluating trade signals.

    Returns:
    DataFrame: Updated DataFrame with 'BuyComment' and 'SellComment' columns added.
    """

    # 定义买卖的信号条件
    buy_conditions = {
        'A': ("KD金叉布林带20%", 'A', 1),
        'B': ("MACD金叉黄金交叉布林带20%", 'B', 1),
        'C': ("三线打击或晨曦之星+小Doji布林带0-20%", 'C', 2),
        'D': ("RSIchannel 0%-20%", 'D', 1),
        'E': ("OBV当前值在20day avg之上且OBV 20天的二阶导数为正", 'E', 2)
    }

    sell_conditions = {
        'A': ("KD死叉布林带80%", 'A', -1),
        'B': ("MACD死叉死亡交叉布林带80%", 'B', -1),
        'C': ("黄昏之星+小Doji布林带80-100% 或三只乌鸦", 'C', -2),
        'D': ("RSIchannel 80%-100%", 'D', -1),
        'E': ("OBV当前值在20day avg之下且OBV 20天的二阶导数为负", 'E', -2)
    }

    def get_signals(row, conditions):
        """
        Internal helper to extract signal descriptions from the row based on defined conditions.
        """
        # Extracting all descriptions where the condition is met
        signals = [desc for desc, col, value in conditions.values() if row[col] == value]
        return "; ".join(signals)  # Join all descriptions into a single string

    # Apply the function to generate buy and sell comments
    df['BuyComment'] = df.apply(lambda row: get_signals(row, buy_conditions), axis=1)
    df['SellComment'] = df.apply(lambda row: get_signals(row, sell_conditions), axis=1)

    return df[['BuyComment', 'SellComment']]



# 结合得分买点和卖点以及预测的结果设置最终地买、卖或持有信号
def calculate_transaction_signals(df):

    # 定义买入信号的条件
    buy_signal = (
            (df['Point'] > 0) &
            # (df['Buy'] > 0) &
            (df['Loss5day'] > -5) &
            (df['Loss20day'] > -10) &
            (df['Loss60day'] > -20)
    )

    # TODO 这个卖出的信号还是很多，还是得想办法调整啊
    # 定义卖出信号的条件
    sell_signal = (
            (df['Point'] < 0) &
            # (df['Sell'] > 0) &
            ((df['Loss5day'] < -5) | (df['Loss20day'] < -10) & (df['Loss60day'] < -20)) |
            ((df['PredictSlope20d'] < 0) & (df['PredictSlope60d'] < 0))
    )

    # 使用numpy select选择条件
    df['Transaction'] = np.select(
        [buy_signal, sell_signal],
        ['Buy', 'Sell'],
        default='Hold'
    )

    return df['Transaction']


def update_kd_bollinger_state(df):
    """
    Update the DataFrame with a new column 'A' based on the combined state of KD line crossings and Bollinger Band percentiles.

    Args:
    df (DataFrame): DataFrame containing the columns 'KDsign' and 'BollingerPercent'.

    Returns:
    DataFrame: Updated DataFrame with a new column 'A'.
    """
    # Define conditions for buy and sell signals
    buy_condition = (df['KDsign'] == 1) & (df['BollingerChannel'] <= 1)
    sell_condition = (df['KDsign'] == 2) & (df['BollingerChannel'] >= 5)

    # 使用numpy select选择条件
    df['A'] = np.select(
        [buy_condition, sell_condition],
        [1, -1],
        default=0
    )

    return df['A']

def update_macd_bollinger_state(df):
    """
    Update the DataFrame with a new column 'A' based on the combined state of KD line crossings and Bollinger Band percentiles.

    Args:
    df (DataFrame): DataFrame containing the columns 'KDsign' and 'BollingerPercent'.

    Returns:
    DataFrame: Updated DataFrame with a new column 'A'.
    """
    # Define conditions for buy and sell signals
    buy_condition = (df['MACDsign'] == 1) & (df['BollingerChannel'] <= 1)
    sell_condition = (df['MACDsign'] == 2) & (df['BollingerChannel'] >= 5)

    # 使用numpy select选择条件
    df['B'] = np.select(
        [buy_condition, sell_condition],
        [1, -1],
        default=0
    )

    return df['B']

def update_obv_state(df):
    """
    Update the DataFrame with a new column 'A' based on the combined state of KD line crossings and Bollinger Band percentiles.

    Args:
    df (DataFrame): DataFrame containing the columns 'KDsign' and 'BollingerPercent'.

    Returns:
    DataFrame: Updated DataFrame with a new column 'A'.
    """
    # Define conditions for buy and sell signals
    buy_condition = (df['OBV'] > df['OBV20ma']) & (df['OBV2de'] > 0)
    sell_condition = (df['OBV'] < df['OBV20ma']) & (df['OBV2de'] < 0)

    # 使用numpy select选择条件
    df['E'] = np.select(
        [buy_condition, sell_condition],
        [2, -2],
        default=0
    )

    return df['E']

"""
从股票策略信息得出交易状态的信息并写入到表格中
参数-stock_table: DataFrame，股票交易信号数据。
返回-DataFrame: 增加了交易信号字段的股票交易数据。
"""
# 将策略数据库读取的表格数据计算交易信息字段并添加到表格右边
def signaling_trade(stock_table, algorithm):

    df = stock_table.copy()
    df.set_index('Date', inplace=True)

    # 状态A - KD线结合 布林带
    df['A'] = update_kd_bollinger_state(df)

    # 状态B - MACDsign 结合 布林带
    df['B'] = update_macd_bollinger_state(df)

    # 状态C - Complex Doji
    df['C'] = calculate_complex_doji(df)

    # 状态F - RSI
    df['D'] = -1 * df['RSIChannel'] + 3

    # 状态E - 随机振荡器 OBV OBV20ma OBV2de
    # df['E'] = df['KDsign'].map({1: -1, 2: 1, 0: 0})
    df['E'] = 0
    df['F'] = 0
    df['G'] = 0
    df['H'] = 0
    """
        # 状态A - 布林带
    df['A'] = -1 * df['BollingerChannel'] + 3

    # 状态B - MACDsign
    df['B'] = df['MACDsign'].map({1: -1, 2: 1, 0: 0})

    # 状态C - Doji
    df['C'] = df['Doji'].map({1: -1, 2: 1, 3: 0, 4: 0, 0: 0})

    # 状态D - complex doji
    df['D'] = calculate_complex_doji(df)

    # 状态E - 随机振荡器 Stochastic Oscillator KD线
    df['E'] = df['KDsign'].map({1: -1, 2: 1, 0: 0})

    # 状态F - RSI
    df['F'] = -1 * df['RSIChannel'] + 3
    
    # 状态G - 商品通道指数（Commodity Channel Index, CCI）
    df['G'] = np.where(df['CCI'] < -200, 2,
                       np.where(df['CCI'] < -100, 1,
                                np.where(df['CCI'] <= 100, 0,
                                         np.where(df['CCI'] <= 200, -1, -2))))

    # 状态H - 威廉姆斯%R
    df['H'] = np.where(df['WilliamsR'] < -80, 1,
                       np.where(df['WilliamsR'] < -20, 0, -1))  
                       
    # Point 也就是上述状态的点数之和
    df['Point'] = df[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']].sum(axis=1)                                
    """

    # Point 也就是上述状态的点数之和
    df['Point'] = 2 * df[['A', 'B', 'C']].sum(axis=1) * df[['D', 'E']].sum(axis=1)

    # 分别返回计算买和卖的信号数量
    df[['Buy', 'Sell']] = add_trade_signals(df)

    # 生成买入和卖出信号描述并以分号连接分别放置在同一个字符串内
    df[['BuyComment', 'SellComment']] = add_trade_signals_comments(df)

    # 将这里使用预测的算法写入数据库中
    df['ForecastAlgorithm'] = algorithm

    # 未来5天 20天 60天 预测最大收益
    df['Profit5day'] = df['PredictProfit5d']
    df['Profit20day'] = df['PredictProfit20d']
    df['Profit60day'] = df['PredictProfit60d']

    # 未来5天 20天 60天 预测最大损失
    df['Loss5day'] = df['PredictLoss5d']
    df['Loss20day'] = df['PredictLoss20d']
    df['Loss60day'] = df['PredictLoss60d']

    # 最终买卖持有信号判断
    df['Transaction'] = calculate_transaction_signals(df)

    # 根据上述的买卖信号计算出回报率
    df['Returns'] = trade_simulation(df)

    # 将 NaN 值替换为0
    df.fillna(0, inplace=True)

    return df



# 执行任务主函数，将上述的函数集合到一次任务当中执行
def doer():
    # 连接读取数据库信息
    db_connection_read = pymysql.connect(**db_config_forecast)
    cursor_read = db_connection_read.cursor()

    # 连接写入数据库信息
    db_connection_update = pymysql.connect(**db_config_trade)
    cursor_update = db_connection_update.cursor()

    # 循环遍历每张表
    for stock_code in stock_codes:

        create_tables(cursor_update, stock_code)  # 先在写入策略信息数据库创建对应表格

        # 根据基础信息数据库和提取信息数据库的最老和最新时间来确定开始更新的时间
        update_time = get_update_time(cursor_read, cursor_update, read_table_suffix, update_table_suffix, stock_code, default_time)
        print("\n针对股票代码为", stock_code, "的开始更新时间是", update_time)

        stock_data = read_db(cursor_read, stock_code, update_time, algorithm)  # 从求得的开始时间开始读取基础信息数据库里面的原始股价信息
        print("读取的原始股票信息为： \n", stock_data)

        stock_data_df = signaling_trade(stock_data, algorithm)  # 调用生成策略数据函数，在原有的股价信息表里面添加计算出来的生成策略数据
        print("计算的生成策略数据表为： \n", stock_data_df)

        # 将生成数据更新到数据库里面
        update_db(stock_code, stock_data_df, update_time, cursor_update, db_connection_update)

    print("\n本次任务结束，已经完全将策略数据更新完毕\n")

    # 关闭数据库连接，任务开始建立连接，任务结束就关闭连接
    cursor_read.close()
    db_connection_read.close()
    cursor_update.close()
    db_connection_update.close()



if __name__ == "__main__":

    # 选好的股票代码 16个精选的股票和ETF基金指数
    stock_codes = ['AMZN', 'BBH', 'FAS', 'GLD', 'GOOG', 'IVV', 'IWD', 'IWM', 'LABU', 'MS', 'QQQ', 'SPY', 'TQQQ', 'VGT',
                   'VUG', 'XLE']

    # 设置数据库的参数 连接读取数据库和更新数据库
    db_config_forecast = {"host": "10.5.0.11", "port": 3306, "user": "lizhuolin", "password": "123456", "database": "forecast"}
    db_config_trade = {"host": "10.5.0.11", "port": 3306, "user": "lizhuolin", "password": "123456", "database": "trade"}

    # "host": "10.5.0.11", "port": 3306, "user": "lizhuolin", "password": "123456"
    # "host": "8.147.99.223", "port": 3306, "user": "lizhuolin", "password": "&a3sFD*432dfD!o0#3^dP2r2d!sc@"

    # 设置读取和更新两个数据库表名的后缀
    read_table_suffix = 'combined'
    update_table_suffix = 'trade'

    # 默认的更新时间
    default_time = '2019-01-01'

    algorithm = 'LSTM'

    # 立即执行一次提取任务
    doer()

    # 使用APSchedule编排任务
    scheduler = BackgroundScheduler()

    # 设置定时任务（设置为每天早上八点半上班前半小时更新）
    days = ['tue', 'wed', 'thu', 'fri', 'sat']
    for day in days:
        scheduler.add_job(doer, 'cron', day_of_week=day, hour=8, minute=30)

    scheduler.start()

    # 运行其他程序逻辑或进入主事件循环，防止程序退出
    try:
        while True:
            time.sleep(30)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
