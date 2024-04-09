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



# 初始的待生成策略数据各股票表建表函数，检测如果数据库内无对应表格则执行该函数
def create_tables(cursor, stock_code):
    create_table_query = \
        f"""
        #
        #
        CREATE TABLE IF NOT EXISTS {stock_code}_strategy(
        `Date` date NOT NULL COMMENT '交易日期',
        `FastAvg` float DEFAULT NULL COMMENT '12天快速移动平均线',
        `SlowAvg` float DEFAULT NULL COMMENT '26天慢速移动平均线',
        `MACD` float DEFAULT NULL COMMENT '快线减慢线的差值',
        `SignalLine` float DEFAULT NULL COMMENT '信号线即MACD值的9天移动平均',
        `MA` float DEFAULT NULL COMMENT '布林带均值的9天移动平均',
        `BollingerUp` float DEFAULT NULL COMMENT '股价加上20天内价格的标准差',
        `BollingerDown` float DEFAULT NULL COMMENT '股价减去20天内价格的标准差',
        `RSI` float DEFAULT NULL COMMENT 'Relative Strength Index相对强度指标',
        `RSIchannel` float DEFAULT NULL COMMENT 'RSIchannel RSI 0-100 将其映射为 1-5',
        `Doji` float DEFAULT NULL COMMENT '十字K线，1为是，0为否',
        `ADX` float DEFAULT NULL COMMENT '平均方向指数（Average Directional Index, ADX)',
        `MACDsign` float DEFAULT NULL COMMENT 'MACD信号 1 2 分别为上穿下，下穿上，0即没有',
        `Channel` float DEFAULT NULL COMMENT '股价的五个通道分类，把布林带分成5个部分从下到上值为0-7，7个状态，0在布林带下，6在布林带上',
        `K` float DEFAULT NULL COMMENT '随机振荡器（Stochastic Oscillator ）的K线',
        `D` float DEFAULT NULL COMMENT '随机振荡器（Stochastic Oscillator ）的D线',
        `CCI` float DEFAULT NULL COMMENT '商品通道指数（Commodity Channel Index, CCI）75%的价格变动位于正负100之间的CCI值',
        `ROC` float DEFAULT NULL COMMENT 'Rate-of-Change, ROC）衡量价格变化幅度的指标',
        `WilliamsR` float DEFAULT NULL COMMENT '威廉姆斯%R（Williams %R）动量指标，识别超买和超卖条件',
        `OBV` float DEFAULT NULL COMMENT '均衡交易量（OBV） On Balance Volume 衡量买卖压力的技术指标，基于成交量的变化来预测价格趋势',
        `Klinger` float DEFAULT NULL COMMENT '克林格指标 Klinger Indicator 判断价格趋势的强度和买卖信号',
        `CMF` float DEFAULT NULL COMMENT '查金资金流（CMF）Chaikin Money Flow 资金在股市中的流入和流出情况',
        `CandleIndi` float DEFAULT NULL COMMENT '复杂蜡烛图指标 Candlestick Indicators 黄昏之星，弃婴，两只乌鸦，三只乌鸦，三线打击 1-5',
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
                #
                #
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
    for index, row in stock_table_df.iterrows():
        # 确保 index_date 和 gotime 都是 datetime.date 类型进行比较
        index_date = index.date() if isinstance(index, datetime) else index
        if index_date >= gotime:  # 只有大于或等于开始更新的时间的数据才执行插入
            insert_query = f"""
            #RSIchannel
            #
           INSERT INTO {stock_code}_strategy 
           (Date, FastAvg, SlowAvg, MACD, SignalLine, MA, BollingerUp, BollingerDown, RSI, RSIchannel, Doji, ADX, MACDsign, Channel,
           K, D, CCI, ROC, WilliamsR, OBV, Klinger, CMF, CandleIndi) 
        VALUES ('{index_date}', {row['FastAvg']}, {row['SlowAvg']}, {row['MACD']}, {row['SignalLine']}, {row['MA']},  {row['BollingerUp']},
         {row['BollingerDown']}, {row['RSI']}, {row['RSIchannel']}, {row['Doji']}, {row['ADX']}, {row['MACDsign']}, {row['Channel']}, {row['K']}, 
         {row['D']}, {row['CCI']}, {row['ROC']}, {row['WilliamsR']}, {row['OBV']}, {row['Klinger']}, {row['CMF']}, {row['CandleIndi']} )
        ON DUPLICATE KEY UPDATE
            FastAvg={row['FastAvg']},
            SlowAvg={row['SlowAvg']},
            MACD={row['MACD']},
            SignalLine={row['SignalLine']},
            MA={row['MA']},
            BollingerUp={row['BollingerUp']},
            BollingerDown={row['BollingerDown']},
            RSI={row['RSI']},
            RSIchannel={row['RSIchannel']},
            Doji={row['Doji']},
            ADX={row['ADX']},
            MACDsign={row['MACDsign']},
            Channel={row['Channel']},
            K={row['K']}, 
            D={row['D']}, 
            CCI={row['CCI']}, 
            ROC={row['ROC']}, 
            WilliamsR={row['WilliamsR']}, 
            OBV={row['OBV']}, 
            Klinger={row['Klinger']}, 
            CMF={row['CMF']}, 
            CandleIndi={row['CandleIndi']}
            """
            cursor.execute(insert_query)
            db_connection.commit()


########################################################################################################################
# 数据时间控制部分
#########################################################################################################################

# TODO 这个还待改善，改成原始数据库和策略数据库的时间轴做减运算，最后得到一个待更新的时间轴序列，然后根据该序列进行计算策略指标并更新
# TODO 从而能解决假设策略数据库不连续的问题，算了，感觉好复杂，如果不连续的话就直接删了重来就行了，应该不会存在吧

# 获得数据库当中最老的日期，以备后续使用
def get_oldest_date(cursor, stock_code, dbname):
    try:
        query = f"SELECT MIN(Date) FROM {stock_code}_{dbname}"
        cursor.execute(query)
        result = cursor.fetchone()
        return result[0] if result and result[0] is not None else None
    except Exception as e:
        print(f"Error fetching oldest date for {stock_code}_{dbname}: {e}")
        return None


# 获得数据库当中最新的日期，以备后续使用
def get_latest_date(cursor, stock_code, dbname):
    try:
        query = f"SELECT MAX(Date) FROM {stock_code}_{dbname}"
        cursor.execute(query)
        result = cursor.fetchone()
        return result[0].strftime("%Y-%m-%d") if result and result[0] is not None else None
    except Exception as e:
        print(f"Error fetching latest date for {stock_code}_{dbname}: {e}")
        return None


# 根据基础信息数据库和提取信息数据库的最老和最新时间来确定开始更新的时间
def get_gotime(cursor_basedata, cursor_strategy, stock_code):
    """
    两个数据库，两个脚本，一个脚本更新基础信息，另一个脚本更新提取信息
    更新提取信息的脚本要同时连接两个数据库，读取基础信息，写入提取信息
    要根据两个脚本最老和最新时间的比较来更新提取信息
    基础最老时间 > 提取最老时间（或者为空）---从基础最老时间开始更新
    基础最新时间 < 提取最新时间 - --从提取最新时间开始更新
    """
    oldest_basedata = get_oldest_date(cursor_basedata, stock_code, "base")
    latest_basedata = get_latest_date(cursor_basedata, stock_code, "base")
    oldest_strategy = get_oldest_date(cursor_strategy, stock_code, "strategy")
    latest_strategy = get_latest_date(cursor_strategy, stock_code, "strategy")

    # 默认开始更新时间
    default_time = datetime.strptime("2014-01-03", "%Y-%m-%d").date()

    if oldest_basedata is None:
        # 如果基础数据的时间为空的话则返回默认时间，默认时间实际上就是基础数据库设定的最早时间
        return default_time
    elif oldest_strategy is None or oldest_basedata > oldest_strategy:
        # 如果基础数据的最老时间更旧，从基础数据最老时间开始更新
        return oldest_basedata
    elif latest_strategy is not None and latest_basedata <= latest_strategy:
        # 如果提取数据的最新时间不为空且早于基础数据的最新时间，从提取数据的最新时间减去一天开始更新
        return datetime.strptime(latest_strategy, "%Y-%m-%d").date() - timedelta(days=1)
    else:
        # 遇上其他情况，如基础数据库时间被策略数据库完全覆盖时，则从最新的时间谁最小减去一天开始更新
        return datetime.strptime(min(latest_strategy, latest_basedata), "%Y-%m-%d").date() - timedelta(days=1)


########################################################################################################################
# 计算添加技术指标信息部分
#########################################################################################################################


"""
计算平均方向移动指数（ADX）
参数-stock_data: DataFrame，window: int，计算ADX所用的滚动窗口大小，默认为14天。
返回- Series: 包含ADX值的Series。
"""
def calculate_adx(stock_data, window=14):
    def calculate_tr(high, low, close):
        # 计算真实范围
        previous_close = close.shift(1)
        tr = pd.DataFrame({'high_low': high - low,
                           'high_close': np.abs(high - previous_close),
                           'low_close': np.abs(low - previous_close)})
        return tr.max(axis=1)

    def calculate_dm(high, low):
        # 计算方向移动
        plus_dm = high.diff()
        minus_dm = low.diff()
        return plus_dm, minus_dm

    # 计算真实范围TR
    tr = calculate_tr(stock_data['High'], stock_data['Low'], stock_data['AdjClose'])
    tr_smooth = tr.rolling(window=window).sum()

    # 计算方向移动DM
    plus_dm, minus_dm = calculate_dm(stock_data['High'], stock_data['Low'])
    plus_dm_smooth = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0).rolling(window=window).sum()
    minus_dm_smooth = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0).rolling(window=window).sum()

    # 计算方向指标DI
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth

    # 计算动向指数DX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)

    # 计算平均动向指数ADX
    adx = dx.rolling(window=window).mean()

    return adx


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
计算相对强弱指数（Relative Strength Index, RSI）。
参数-stock_data: DataFrame
返回-Series: 计算得到的RSI值。
"""
def calculate_RSI(stock_data, window=14):
    change = stock_data['AdjClose'].diff()
    gain = change.apply(lambda x: x if x > 0 else 0)
    loss = change.apply(lambda x: -x if x < 0 else 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


"""
计算MACD交叉信号
参数-stock_data: DataFrame
返回-Series: 计算得到的MACD交叉信号，1表示MACD向上穿过信号线，2表示MACD向下穿过信号线，0表示无穿越。
"""
def calculate_macd_signal(stock_data):
    macd_sign = np.where(
        (stock_data['MACD'] > stock_data['SignalLine']) &
        (stock_data['MACD'].shift(1) < stock_data['SignalLine'].shift(1)),
        1,  # MACD向上穿过信号线
        np.where(
            (stock_data['MACD'] < stock_data['SignalLine']) &
            (stock_data['MACD'].shift(1) > stock_data['SignalLine'].shift(1)),
            2,  # MACD向下穿过信号线
            0  # 无穿越
        )
    )
    return macd_sign


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
计算随机振荡器指标。
参数:
df-DataFrame， k_period: int，，默认为14天。d_period: int，默认为3天。
返回-DataFrame: 包含'%K'和'%D'列的DataFrame。
"""
def calculate_stochastic_oscillator(df, k_period=14, d_period=3):
    # 计算%K线
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    df['%K'] = ((df['AdjClose'] - low_min) / (high_max - low_min)) * 100

    # 计算%D线（%K线的移动平均）
    df['%D'] = df['%K'].rolling(window=d_period).mean()

    return df[['%K', '%D']]


"""
计算商品通道指数（CCI）。
参数-df: DataFrame，包含股票交易数据，至少包括'High', 'Low', 和 'Close'列, n: int，，默认为20天。
返回-Series: 包含CCI值的Series。
"""


def calculate_cci(df, window=20):
    # 计算典型价格
    TP = (df['High'] + df['Low'] + df['Close']) / 3

    # 计算TP的n期SMA
    TP_SMA = TP.rolling(window=window).mean()

    # 计算平均偏差
    mean_deviation = TP.rolling(window=window).apply(lambda x: np.fabs(x - x.mean()).mean(), raw=True)

    # 计算CCI
    CCI = (TP - TP_SMA) / (0.015 * mean_deviation)

    return CCI


"""
计算均衡交易量（OBV） On Balance Volume
参数-df: DataFrame，包含股票交易数据，至少包含'Close'和'Volume'列。
返回-Series: 包含OBV值的Series。
"""
def calculate_obv(df):
    obv = pd.Series(index=df.index, data=0)
    obv.iloc[0] = df['Volume'].iloc[0]

    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] + df['Volume'].iloc[i]
        elif df['Close'].iloc[i] < df['Close'].iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] - df['Volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i - 1]
    return obv


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
返回-Series: 标注了蜡烛图指标的Series，1-5分别代表黄昏之星，弃婴，两只乌鸦，三只乌鸦，三线打击。
"""
def detect_candlestick_patterns(df):
    patterns = pd.Series(index=df.index, data=0)  # 初始化蜡烛图指标列为0

    # 简化的指标检测逻辑
    # 注意：以下逻辑是基于标准形态的简化版本，实际检测可能需要更复杂的条件判断

    for i in range(2, len(df)):
        # 黄昏之星
        if df['Close'].iloc[i - 2] > df['Open'].iloc[i - 2] and \
                df['Close'].iloc[i - 1] < df['Open'].iloc[i - 1] and df['Low'].iloc[i - 1] > df['High'].iloc[i - 2] and \
                df['Close'].iloc[i] < df['Open'].iloc[i] and df['Close'].iloc[i] < df['Close'].iloc[i - 2]:
            patterns.iloc[i] = 1  # 黄昏之星
        # 弃婴
        # 注意：弃婴的检测需要考虑间隔，这里仅作为示例
        elif df['Close'].iloc[i - 2] < df['Open'].iloc[i - 2] and \
                df['High'].iloc[i - 1] < df['Low'].iloc[i - 2] and df['High'].iloc[i - 1] < df['Low'].iloc[i] and \
                df['Close'].iloc[i] > df['Open'].iloc[i]:
            patterns.iloc[i] = 2  # 弃婴
        # 两只乌鸦
        # 注意：实际检测可能更复杂
        elif (df['Open'].iloc[i - 2] < df['Close'].iloc[i - 2] < df['Open'].iloc[i - 1] and df['Close'].iloc[i - 1] <
              df['Open'].iloc[i - 1] and df['Open'].iloc[i] > df['Close'].iloc[i - 1] and df['Close'].iloc[i] < df['Open'].iloc[i]):
            patterns.iloc[i] = 3  # 两只乌鸦
        # 三只乌鸦
        elif df['Close'].iloc[i - 2] > df['Open'].iloc[i - 2] and \
                df['Close'].iloc[i - 1] < df['Open'].iloc[i - 1] and df['Close'].iloc[i - 1] < df['Close'].iloc[
            i - 2] and \
                df['Close'].iloc[i] < df['Open'].iloc[i] and df['Close'].iloc[i] < df['Close'].iloc[i - 1]:
            patterns.iloc[i] = 4  # 三只乌鸦
        # 三线打击
        # 注意：这个模式通常更复杂，以下仅为示例
        elif df['Close'].iloc[i - 2] < df['Open'].iloc[i - 2] and \
                df['Close'].iloc[i - 1] < df['Open'].iloc[i - 1] and df['Close'].iloc[i - 1] < df['Close'].iloc[
            i - 2] and \
                df['Close'].iloc[i] > df['Open'].iloc[i - 2]:
            patterns.iloc[i] = 5  # 三线打击

    return patterns


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

    # 计算布林带
    df['MA'] = (df['High'] + df['Low'] + df['AdjClose']).rolling(window=20).mean() / 3
    df['BollingerUp'] = df['MA'] + 2 * df['AdjClose'].rolling(window=20).std()
    df['BollingerDown'] = df['MA'] - 2 * df['AdjClose'].rolling(window=20).std()

    # 计算MACD（快线减慢线）以及其信号线（MACD的9天移动平均）
    df['MACD'] = df['FastAvg'] - df['SlowAvg']
    df['SignalLine'] = df['MACD'].rolling(window=9).mean()

    # MACD信号
    df['MACDsign'] = calculate_macd_signal(df)

    # 计算RSI
    df['RSI'] = calculate_RSI(df, window=14)

    # 计算RSI channel 映射RSI到1-5范围
    df['RSIchannel'] = df['RSI'].apply(map_rsi)

    # Doji星计算 1墓碑Doji 2蜻蜓Doji 3长脚Doji 4普通Doji  应用定义的函数到DataFrame
    df['Doji'] = df.apply(classify_doji, axis=1)

    # 计算ADX
    df['ADX'] = calculate_adx(df, window=14)

    # 计算基于布林带的Channel通道
    df['Channel'] = df.apply(calculate_channel, axis=1)

    # 计算随机振荡器 Stochastic Oscillator
    df[['K', 'D']] = calculate_stochastic_oscillator(df)

    # 商品通道指数 计算CCI
    df['CCI'] = calculate_cci(df, window=20)

    # Momentum 动量指标 计算Rate-of-Change ROC
    df['ROC'] = ((df['Close'] / df['Close'].shift(14)) - 1) * 100

    # 计算威廉姆斯 % R指标 Williams %R
    highest_high = df['High'].rolling(window=14).max()
    lowest_low = df['Low'].rolling(window=14).min()
    df['WilliamsR'] = ((highest_high - df['Close']) / (highest_high - lowest_low)) * -100

    # 计算均衡交易量（OBV） On Balance Volume
    df['OBV'] = calculate_obv(df)

    # 计算克林格指标 Klinger Indicator
    df['Klinger'] = calculate_klinger(df, fast=12, slow=26)

    # 计算查金资金流（CMF）Chaikin Money Flow
    df['CMF'] = calculate_cmf(df, window=20)

    # 计算复杂蜡烛图指标 Candlestick Indicators 黄昏之星，弃婴，两只乌鸦，三只乌鸦，三线打击
    df['CandleIndi'] = detect_candlestick_patterns(df)

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

        gotime = get_gotime(cursor_basedata, cursor_strategy, stock_code)  # 求得更新开始的时间
        print("\n针对股票代码为", stock_code, "的开始更新时间是", gotime)

        stock_data = read_db(cursor_basedata, stock_code, gotime)  # 从求得的开始时间开始读取基础信息数据库里面的原始股价信息
        print("读取的原始股票信息为： \n", stock_data)

        stock_data_df = signaling_strategy(stock_data)  # 调用生成策略数据函数，在原有的股价信息表里面添加计算出来的生成策略数据
        print("计算的生成策略数据表为： \n", stock_data_df)

        update_stock_data(stock_code, stock_data_df, gotime, cursor_strategy,
                          db_connection_strategy)  # 这个df不知道什么问题 然后将生成策略数据更新到数据库里面

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
    db_config = {"host": "10.5.0.18", "port": 3306, "user": "lizhuolin", "password": "123456", "database": "basedata"}

    # 设置数据库的参数 连接数据库的信息（连接待生成策略数据的数据库）
    db_config_strategy = {"host": "10.5.0.18", "port": 3306, "user": "lizhuolin", "password": "123456", "database": "strategy"}

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