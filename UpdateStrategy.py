"""
GT-ECE8903 24spring
Zhuolin Li

主要目的为利用已经爬取好并存储在数据库里面的基础股价信息，然后计算出策略信息数据并存储到另一个数据库当中
1.连接本地的mySQL数据库，包括基础信息数据库和策略信息数据库
2.检测并建立策略信息数据库对应股价的表格，如已建表则跳过
3.如则读取原始基础信息数据库和策略信息数据库，对应表内最新的日期和最老的日期，然后确定开始更新的日期
4.读取原始基础信息数据库开始更新的日期前30天的数据，并存在临时变量当中
5.在临时变量表格后面计算并添加MACD对应的数据
6.根据日期信息来将计算好的MACD信息更新策略信息表格内（2014-1-1到系统当前日期）
7.设定为交易日收盘后定时更新，以及初次运行该脚本时更新一次
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
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
import talib


########################################################################################################################
# 数据库SQL读写建表部分
#########################################################################################################################

# 初始的待生成策略数据各股票表建表函数，检测如果数据库内无对应表格则执行该函数
def create_tables(cursor, stock_code):
    create_table_query = \
        f"""
        CREATE TABLE IF NOT EXISTS {stock_code}_strategy(
        `Date` date NOT NULL COMMENT '交易日期',
        `FastAvg` float DEFAULT NULL COMMENT '12天快速移动平均线',
        `SlowAvg` float DEFAULT NULL COMMENT '26天慢速移动平均线',
        `MACD` float DEFAULT NULL COMMENT '快线减慢线的差值',
        `SignalLine` float DEFAULT NULL COMMENT '信号线即MACD值的9天移动平均',
        `MA` float DEFAULT NULL COMMENT '布林带均值的9天移动平均',
        `BollingerUp` float DEFAULT NULL COMMENT '股价加上20天内价格的标准差',
        `BollingerDown` float DEFAULT NULL COMMENT '股价减去20天内价格的标准差',
        `BollingerChannel` float DEFAULT NULL COMMENT '股价的五个通道分类，把布林带分成5个部分从下到上值为0-7，7个状态，0在布林带下，6在布林带上',
        `RSI` float DEFAULT NULL COMMENT 'Relative Strength Index相对强度指标',
        `RSIChannel` float DEFAULT NULL COMMENT 'RSI 0-100 将其映射为 1-5',
        `Doji` float DEFAULT NULL COMMENT '检测是否十字星Doji，1 墓碑Doji  2 蜻蜓Doji  3 长脚Doji  4 普通Doji   0 其他不是Doji的情况',
        `ADX` float DEFAULT NULL COMMENT '平均方向指数（Average Directional Index, ADX)',
        `MACDsign` float DEFAULT NULL COMMENT 'MACD信号 1 2 分别为上穿下，下穿上，0即没有',
        `K` float DEFAULT NULL COMMENT '随机振荡器（Stochastic Oscillator ）的K线',
        `D` float DEFAULT NULL COMMENT '随机振荡器（Stochastic Oscillator ）的D线',
        `KDsign` float DEFAULT NULL COMMENT '随机振荡器（Stochastic Oscillator ）KD线的上下穿行，1 代表K线从下穿上D线，2 代表K线从上穿下D线，0即没有',
        `CCI` float DEFAULT NULL COMMENT '商品通道指数（Commodity Channel Index, CCI）75%的价格变动位于正负100之间的CCI值',
        `ROC` float DEFAULT NULL COMMENT 'Rate-of-Change, ROC）衡量价格变化幅度的指标',
        `WilliamsR` float DEFAULT NULL COMMENT '威廉姆斯%R（Williams %R）动量指标，识别超买和超卖条件',
        `OBV` float DEFAULT NULL COMMENT '均衡交易量（OBV） On Balance Volume 衡量买卖压力的技术指标，基于成交量的变化来预测价格趋势',
        `OBV20ma` float DEFAULT NULL COMMENT '均衡交易量（OBV）20天的移动平均值',
        `OBV2de` float DEFAULT NULL COMMENT '均衡交易量（OBV）2次导数',
        `Klinger` float DEFAULT NULL COMMENT '克林格指标 Klinger Indicator 判断价格趋势的强度和买卖信号',
        `CMF` float DEFAULT NULL COMMENT '查金资金流（CMF）Chaikin Money Flow 资金在股市中的流入和流出情况',
        `ComplexDoji` float DEFAULT NULL COMMENT '复杂蜡烛图指标 Candlestick Indicators 黄昏之星，弃婴，两只乌鸦，三只乌鸦，三线打击，晨曦之星 1-6',
        PRIMARY KEY (`Date` DESC),
        UNIQUE KEY `date` (`Date`) USING BTREE
        )
    """
    cursor.execute(create_table_query)


# 读取原始数据的数据库里面的数据
def read_db(cursor, stock_code, start_time):
    try:
        formatted_time = start_time.strftime("%Y-%m-%d")  # 确保时间格式正确（防止SQL注入）
        query = f"""
            SELECT Date, Open, High, Low, Close, AdjClose, Volume
            FROM {stock_code}_base
            WHERE Date >= DATE_SUB(%s, INTERVAL 60 DAY) #读取时间开始前30天的数据
            ORDER BY Date ASC;
        """
        cursor.execute(query, (formatted_time,))  # 参数化查询
        result = cursor.fetchall()

        # 将查询结果转换为 DataFrame
        df = pd.DataFrame(result, columns=['Date', 'Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume'])

        return df
    except Exception as e:
        print(f"Error fetching date for {stock_code}: {e}")


# 将添加好的策略信息字段写入到数据库当中
def update_stock_data(stock_code, stock_table_df, gotime, cursor, db_connection):
    # 准备批量插入的SQL语句，同时处理重复键的情况
    insert_query = f"""
        INSERT INTO {stock_code}_strategy 
        (Date, FastAvg, SlowAvg, MACD, SignalLine, MA, BollingerUp, BollingerDown, BollingerChannel, RSI, RSIChannel, Doji, ADX, MACDsign, 
         K, D, KDsign, CCI, ROC, WilliamsR, OBV, OBV20ma, OBV2de, Klinger, CMF, ComplexDoji) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        FastAvg=VALUES(FastAvg),
        SlowAvg=VALUES(SlowAvg),
        MACD=VALUES(MACD),
        SignalLine=VALUES(SignalLine),
        MA=VALUES(MA),
        BollingerUp=VALUES(BollingerUp),
        BollingerDown=VALUES(BollingerDown),
        BollingerChannel=VALUES(BollingerChannel),
        RSI=VALUES(RSI),
        RSIChannel=VALUES(RSIChannel),
        Doji=VALUES(Doji),
        ADX=VALUES(ADX),
        MACDsign=VALUES(MACDsign),
        K=VALUES(K),
        D=VALUES(D),
        KDsign=VALUES(KDsign),
        CCI=VALUES(CCI),
        ROC=VALUES(ROC),
        WilliamsR=VALUES(WilliamsR),
        OBV=VALUES(OBV),
        OBV20ma=VALUES(OBV20ma),
        OBV2de=VALUES(OBV2de),
        Klinger=VALUES(Klinger),
        CMF=VALUES(CMF),
        ComplexDoji=VALUES(ComplexDoji)
    """

    # 创建一个元组列表，包含所有要插入的数据
    data_to_insert = [
        (index.date() if isinstance(index, datetime) else index, row['FastAvg'], row['SlowAvg'], row['MACD'],
         row['SignalLine'], row['MA'],
         row['BollingerUp'], row['BollingerDown'], row['BollingerChannel'], row['RSI'], row['RSIChannel'], row['Doji'],
         row['ADX'], row['MACDsign'],
         row['K'], row['D'], row['KDsign'], row['CCI'], row['ROC'], row['WilliamsR'], row['OBV'], row['OBV20ma'],
         row['OBV2de'], row['Klinger'], row['CMF'], row['ComplexDoji'])
        for index, row in stock_table_df.iterrows()
        if (index.date() if isinstance(index, datetime) else index) >= gotime
    ]

    # 使用 executemany 进行批量插入操作
    if data_to_insert:  # 确保有数据插入
        cursor.executemany(insert_query, data_to_insert)
        db_connection.commit()  # 一次性提交所有插入操作


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
# 计算添加技术指标信息部分
#########################################################################################################################

"""
根据给定的股票交易数据行来分类Doji星形态。
参数-row: DataFrame的行，包含Open, AdjClose, High, Low。
返回-int: Doji星形态的类型编号。
"""


def classify_doji(row):
    # 计算实体大小和上下影线长度
    body_size = abs(row['Open'] - row['AdjClose'])
    upper_shadow = row['High'] - max(row['Open'], row['AdjClose'])
    lower_shadow = min(row['Open'], row['AdjClose']) - row['Low']

    # 定义Doji类型的条件
    normal = body_size <= (row['High'] - row['Low']) * 0.1
    gravestone = (upper_shadow > 2 * body_size) and normal
    dragonfly = (lower_shadow > 2 * body_size) and normal
    long_legged = (upper_shadow > 2 * body_size) and (lower_shadow > 2 * body_size)

    # 分类Doji
    if gravestone:
        return 1  # 墓碑Doji
    elif dragonfly:
        return 2  # 蜻蜓Doji
    elif long_legged:
        return 3  # 长脚Doji
    elif normal:
        return 4  # 普通Doji
    else:
        return 0  # 其他


"""
计算两条线的交叉信号
参数: fast_line: DataFrame列，代表快速线，如MACD线；slow_line: DataFrame列，代表慢速线，如信号线
返回: Series: 根据两线的交叉情况计算得到的交叉信号，
1表示快速线向上穿过慢速线，2表示快速线向下穿过慢速线，0表示无穿越。
"""


def calculate_crossover(fast_line, slow_line):
    crossover_signal = np.where(
        (fast_line > slow_line) &
        (fast_line.shift(1) < slow_line.shift(1)),
        1,  # 快速线向上穿过慢速线
        np.where(
            (fast_line < slow_line) &
            (fast_line.shift(1) > slow_line.shift(1)),
            2,  # 快速线向下穿过慢速线
            0  # 无穿越
        )
    )
    return crossover_signal


"""
将RSI指标映射到1-5的范围。
参数-rsi: float，原始的RSI值，范围在0到100之间。
返回-int: 映射到1-5范围的RSI值，1-5分别表示较低到较高的RSI水平。
"""


def map_rsi(rsi):
    if np.isnan(rsi):
        return 0  # 或者返回其他合适的数值
    rsi_min = 0
    rsi_max = 100
    mapped_rsi = int(np.ceil((rsi - rsi_min) / (rsi_max - rsi_min) * 4 + 1))
    return mapped_rsi


"""
计算股价相对于布林带的通道分类。
参数-df: DataFrame，包含'AdjClose', 'BollingerDown', 'BollingerUp'列的股票数据。
返回-int: 表示每日股价所在通道的分类（0-6）。
"""


def calculate_channel(row):
    # 计算布林带的中线和四分点
    bollinger_mid = (row['BollingerUp'] + row['BollingerDown']) / 2
    quarter1 = row['BollingerDown'] + (bollinger_mid - row['BollingerDown']) / 3
    quarter2 = row['BollingerDown'] + 2 * (bollinger_mid - row['BollingerDown']) / 3
    quarter3 = bollinger_mid + (row['BollingerUp'] - bollinger_mid) / 3
    quarter4 = bollinger_mid + 2 * (row['BollingerUp'] - bollinger_mid) / 3

    # 分类逻辑
    if row['AdjClose'] < row['BollingerDown']:
        return 0
    elif row['AdjClose'] < quarter1:
        return 1
    elif row['AdjClose'] < quarter2:
        return 2
    elif row['AdjClose'] < quarter3:
        return 3
    elif row['AdjClose'] < quarter4:
        return 4
    elif row['AdjClose'] < row['BollingerDown']:
        return 5
    else:
        return 6


"""
计算克林格指标 Klinger Indicator
参数-df: DataFrame，包含股票交易数据，'High', 'Low', 'Close'和'Volume'；fast: 快速EMA周期；slow: 慢速EMA周期。
返回-DataFrame: 包含克林格指标值的DataFrame。
"""


def calculate_klinger(df, fast=12, slow=26):
    # 计算真实流通量
    dm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    dm.fillna(0, inplace=True)
    volume_force = dm * df['Volume']

    # 计算快速和慢速EMA
    kvo = volume_force.ewm(span=fast).mean() - volume_force.ewm(span=slow).mean()
    return kvo


"""
计算查金资金流（CMF）Chaikin Money Flow.
参数-df: DataFrame，至少包含'High', 'Low', 'Close'和'Volume'，n: int，用于计算CMF的周期，默认为20天。
返回-Series: 包含CMF值的Series。
"""


def calculate_cmf(df, window=20):
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mfm.fillna(0, inplace=True)
    mf_volume = mfm * df['Volume']

    cmf = mf_volume.rolling(window=window).sum() / df['Volume'].rolling(window=window).sum()
    return cmf


"""
检测特定的蜡烛图形态并在新列中标注。
参数-df: DataFrame，包含股票交易数据，至少包含'Open', 'High', 'Low', 'Close'列。
返回-Series: 标注了蜡烛图指标的Series，1-6分别代表黄昏之星，弃婴，两只乌鸦，三只乌鸦，三线打击，晨曦之星
"""


def detect_candlestick_patterns(df):
    # 初始化蜡烛图指标列为0
    df['ComplexDoji'] = 0

    # 检测黄昏之星
    evening_star = talib.CDLEVENINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
    df.loc[evening_star != 0, 'ComplexDoji'] = 1  # 黄昏之星标记为1

    # 检测弃婴，仅当没有先前检测到形态时
    if not df['ComplexDoji'].any():  # 检查是否已经有任何非0值
        abandoned_baby = talib.CDLABANDONEDBABY(df['Open'], df['High'], df['Low'], df['Close'])
        df.loc[abandoned_baby != 0, 'ComplexDoji'] = 2  # 弃婴标记为2

    # 检测两只乌鸦，仅当没有先前检测到形态时
    if not df['ComplexDoji'].any():
        two_crows = talib.CDL2CROWS(df['Open'], df['High'], df['Low'], df['Close'])
        df.loc[two_crows != 0, 'ComplexDoji'] = 3  # 两只乌鸦标记为3

    # 检测三只乌鸦，仅当没有先前检测到形态时
    if not df['ComplexDoji'].any():
        three_black_crows = talib.CDL3BLACKCROWS(df['Open'], df['High'], df['Low'], df['Close'])
        df.loc[three_black_crows != 0, 'ComplexDoji'] = 4  # 三只乌鸦标记为4

    # 检测三线打击，仅当没有先前检测到形态时
    if not df['ComplexDoji'].any():
        three_line_strike = talib.CDL3LINESTRIKE(df['Open'], df['High'], df['Low'], df['Close'])
        df.loc[three_line_strike != 0, 'ComplexDoji'] = 5  # 三线打击标记为5

    # 检测晨曦之星，仅当没有先前检测到形态时
    if not df['ComplexDoji'].any():
        morning_star = talib.CDLMORNINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
        df.loc[morning_star != 0, 'ComplexDoji'] = 6  # 晨曦之星标记为6

    return df['ComplexDoji']


"""
从股票交易表格中计算并添加交易策略信息字段。
参数-stock_table: DataFrame，股票交易数据。
返回-DataFrame: 增加了策略信息字段的股票交易数据。
"""
# 将原始数据库读取的表格数据计算策略信息字段并添加到表格右边
def signaling_strategy(stock_table):
    df = stock_table.copy()
    df.set_index('Date', inplace=True)

    # 计算快速（12天）和慢速（26天）移动平均线
    df['FastAvg'] = df['AdjClose'].rolling(window=12).mean()
    df['SlowAvg'] = df['AdjClose'].rolling(window=26).mean()

    # 通常布林带的计算是基于收盘价的，这里使用调整后收盘价（AdjClose）
    df['MA'], df['BollingerUp'], df['BollingerDown'] = talib.BBANDS(
        df['AdjClose'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

    # 计算MACD（快线减慢线）以及其信号线（MACD的9天移动平均）
    df['MACD'], df['SignalLine'], df['MACDHistogram'] = talib.MACD(
        df['AdjClose'], fastperiod=12, slowperiod=26, signalperiod=9)

    # MACD信号
    df['MACDsign'] = calculate_crossover(df['MACD'], df['SignalLine'])

    # 计算RSI
    df['RSI'] = talib.RSI(df['AdjClose'], timeperiod=14)

    # 计算RSI channel 映射RSI到1-5范围
    df['RSIChannel'] = df['RSI'].apply(map_rsi)

    # Doji星计算 1墓碑Doji 2蜻蜓Doji 3长脚Doji 4普通Doji  应用定义的函数到DataFrame
    df['Doji'] = df.apply(classify_doji, axis=1)

    # 计算ADX
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)

    # 计算基于布林带的Channel通道
    df['BollingerChannel'] = df.apply(calculate_channel, axis=1)

    # 计算随机振荡器 Stochastic Oscillator
    df['K'], df['D'] = talib.STOCH(df['High'], df['Low'], df['Close'],
                                   fastk_period=5,
                                   slowk_period=3, slowk_matype=0,
                                   slowd_period=3, slowd_matype=0)

    # 计算K D线 是否上下交叉的信号
    df['KDsign'] = calculate_crossover(df['K'], df['D'])

    # 商品通道指数 计算CCI
    df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=20)

    # Momentum 动量指标 计算Rate-of-Change ROC
    df['ROC'] = talib.ROC(df['Close'], timeperiod=14)

    # 计算威廉姆斯 % R指标 Williams %R
    df['WilliamsR'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # 计算均衡交易量（OBV） On Balance Volume
    df['OBV'] = talib.OBV(df['Close'], df['Volume'])

    # 计算均衡交易量（OBV）的20天移动平均值
    df['OBV20ma'] = df['OBV'].rolling(window=20).mean()

    # 计算均衡交易量（OBV）的一阶导数
    df['OBV2de'] = df['OBV'].diff() / df['OBV'].rolling(window=1).mean()

    # 计算克林格指标 Klinger Indicator
    df['Klinger'] = calculate_klinger(df, fast=12, slow=26)

    # 计算查金资金流（CMF）Chaikin Money Flow
    df['CMF'] = calculate_cmf(df, window=20)

    # 计算复杂蜡烛图指标 Candlestick Indicators 黄昏之星，弃婴，两只乌鸦，三只乌鸦，三线打击
    df['ComplexDoji'] = detect_candlestick_patterns(df)

    # 将 NaN 值替换为0
    df.fillna(0, inplace=True)

    return df


########################################################################################################################
# 主函数执行部分
#########################################################################################################################


# 执行任务主函数，将上述的函数集合到一次任务当中执行
def doer():
    # 连接读取基础信息数据库信息
    db_connection_basedata = pymysql.connect(**db_config)
    cursor_basedata = db_connection_basedata.cursor()

    # 连接写入策略信息数据库信息
    db_connection_strategy = pymysql.connect(**db_config_strategy)
    cursor_strategy = db_connection_strategy.cursor()

    # 循环遍历每张表
    for stock_code in stock_codes:
        create_tables(cursor_strategy, stock_code)  # 先在写入策略信息数据库创建对应表格

        # 求得更新开始的时间
        update_time = get_update_time(cursor_basedata, cursor_strategy, read_suffix, update_suffix, stock_code, default)
        print("\n针对股票代码为", stock_code, "的开始更新时间是", update_time)

        stock_data = read_db(cursor_basedata, stock_code, update_time)  # 从求得的开始时间开始读取基础信息数据库里面的原始股价信息
        print("读取的原始股票信息为： \n", stock_data)

        stock_data_df = signaling_strategy(stock_data)  # 调用生成策略数据函数，在原有的股价信息表里面添加计算出来的生成策略数据
        print("计算的生成策略数据表为： \n", stock_data_df)

        # 将生成策略数据更新到数据库里面
        update_stock_data(stock_code, stock_data_df, update_time, cursor_strategy, db_connection_strategy)

    print("\n本次任务结束，已经完全将策略数据更新完毕\n")

    cursor_basedata.close()
    db_connection_basedata.close()
    cursor_strategy.close()
    db_connection_strategy.close()
    # 关闭上面的数据库连接，任务开始建立连接，任务结束就关闭连接


if __name__ == "__main__":

    # 选好的股票代码 16个精选的股票和ETF基金指数
    stock_codes = ['AMZN', 'BBH', 'FAS', 'GLD', 'GOOG', 'IVV', 'IWD', 'IWM', 'LABU', 'MS', 'QQQ', 'SPY', 'TQQQ', 'VGT',
                   'VUG', 'XLE']

    # 设置数据库的参数 连接数据库的信息（连接待基础数据的数据库）
    db_config = {"host": "101.37.77.232", "port": 8903, "user": "lizhuolin", "password": "123456",
                 "database": "basedata"}

    # 设置数据库的参数 连接数据库的信息（连接待生成策略数据的数据库）
    db_config_strategy = {"host": "101.37.77.232", "port": 8903, "user": "lizhuolin", "password": "123456",
                          "database": "strategy"}

    # "host": "10.5.0.11", "port": 3306, "user": "lizhuolin", "password": "123456"
    # "host": "8.147.99.223", "port": 3306, "user": "lizhuolin", "password": "&a3sFD*432dfD!o0#3^dP2r2d!sc@"

    # 读取表和更新表的后缀
    read_suffix = 'base'
    update_suffix = 'strategy'

    # 默认开始的更新时间
    default = '2014-01-01'

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
