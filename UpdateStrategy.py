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

import logging
import numpy as np
from datetime import datetime, date, timedelta
import pandas as pd
import talib
from tenacity import retry, stop_after_attempt, wait_fixed
from BaseToolkit import ConnectionPoolDB, DateControl, Scheduler, LoggerManager

########################################################################################################################
# 数据库SQL读写建表类
#########################################################################################################################
class StrategyDataManager:
    # 初始化函数，传入配置管理器
    def __init__(self, connection_pool_db_basedata, connection_pool_db_strategy):  # 传入数据库连接池对象
        self.pool_db_basedata = connection_pool_db_basedata
        self.pool_db_strategy = connection_pool_db_strategy

    # 初始的待生成策略数据各股票表建表函数，检测如果数据库内无对应表格则执行该函数
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))  # 重试机制，最大尝试次数为3，每次尝试间隔为1秒
    def create_tables(self, stock_code: str):
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
        with self.pool_db_strategy.get_connection() as connection:  # 获取数据库连接
            with connection.cursor() as cursor:
                cursor.execute(create_table_query)
                connection.commit()

    # 读取原始数据的数据库里面的数据
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))  # 重试机制，最大尝试次数为3，每次尝试间隔为1秒
    def read_db(self, stock_code: str, start_time: datetime) -> pd.DataFrame:
        with self.pool_db_basedata.get_cursor() as cursor:  # 获取数据库连接
            try:

                formatted_time = start_time.strftime("%Y-%m-%d")  # 确保时间格式正确（防止SQL注入）
                query = f"""
                        SELECT Date, Open, High, Low, Close, AdjClose, Volume
                        FROM {stock_code}_base
                        WHERE Date >= DATE_SUB(%s, INTERVAL 30 DAY) #读取时间开始前30天的数据
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
    def update_stock_data(self, stock_code: str, stock_table_df: pd.DataFrame, gotime: datetime):
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
             row['BollingerUp'], row['BollingerDown'], row['BollingerChannel'], row['RSI'], row['RSIChannel'],
             row['Doji'],
             row['ADX'], row['MACDsign'],
             row['K'], row['D'], row['KDsign'], row['CCI'], row['ROC'], row['WilliamsR'], row['OBV'],
             row['OBV20ma'],
             row['OBV2de'], row['Klinger'], row['CMF'], row['ComplexDoji'])
            for index, row in stock_table_df.iterrows()
            if (index.date() if isinstance(index, datetime) else index) >= gotime
        ]

        # 使用 executemany 进行批量插入操作
        if data_to_insert:  # 确保有数据插入
            with self.pool_db_strategy.get_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.executemany(insert_query, data_to_insert)
                    connection.commit()  # 一次性提交所有插入操作


########################################################################################################################
# 计算添加技术指标信息类
#########################################################################################################################
class StrategyDataFetcher:
    def __init__(self):
        pass

    """
    根据给定的股票交易数据行来分类Doji星形态。
    参数-row: DataFrame的行，包含Open, AdjClose, High, Low。
    返回-int: Doji星形态的类型编号。
    """

    @staticmethod
    def classify_doji(row: pd.Series) -> int:
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

    @staticmethod
    def calculate_crossover(fast_line: int, slow_line: int) -> int:
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
    @staticmethod
    def map_rsi(rsi : float) -> int:
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

    @staticmethod
    def calculate_channel(row: pd.Series) -> int:
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
    @staticmethod
    def calculate_klinger(df: pd.DataFrame, fast=12, slow=26) -> pd.Series:
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
    @staticmethod
    def calculate_cmf(df: pd.DataFrame, window=20) -> pd.Series:
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
    @staticmethod
    def detect_candlestick_patterns(df: pd.DataFrame) -> pd.Series:
        # 初始化蜡烛图指标列为0
        df['ComplexDoji'] = 0

        # 检测黄昏之星
        evening_star = talib.CDLEVENINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
        df.loc[evening_star != 0, 'ComplexDoji'] = 1  # 黄昏之星标记为1

        # 检测弃婴，仅当没有先前检测到形态时
        if not df['ComplexDoji'].any():  # 检查是否已经有任何非零值
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
    def signaling_strategy(self, stock_table: pd.DataFrame) -> pd.DataFrame:
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
        df['MACDsign'] = self.calculate_crossover(df['MACD'], df['SignalLine'])

        # 计算RSI
        df['RSI'] = talib.RSI(df['AdjClose'], timeperiod=14)

        # 计算RSI channel 映射RSI到1-5范围
        df['RSIChannel'] = df['RSI'].apply(self.map_rsi)

        # Doji星计算 1墓碑Doji 2蜻蜓Doji 3长脚Doji 4普通Doji  应用定义的函数到DataFrame
        df['Doji'] = df.apply(self.classify_doji, axis=1)

        # 计算ADX
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)

        # 计算基于布林带的Channel通道
        df['BollingerChannel'] = df.apply(self.calculate_channel, axis=1)

        # 计算随机振荡器 Stochastic Oscillator
        df['K'], df['D'] = talib.STOCH(df['High'], df['Low'], df['Close'],
                                       fastk_period=5,
                                       slowk_period=3, slowk_matype=0,
                                       slowd_period=3, slowd_matype=0)

        # 计算K D线 是否上下交叉的信号
        df['KDsign'] = self.calculate_crossover(df['K'], df['D'])

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
        df['Klinger'] = self.calculate_klinger(df, fast=12, slow=26)

        # 计算查金资金流（CMF）Chaikin Money Flow
        df['CMF'] = self.calculate_cmf(df, window=20)

        # 计算复杂蜡烛图指标 Candlestick Indicators 黄昏之星，弃婴，两只乌鸦，三只乌鸦，三线打击
        df['ComplexDoji'] = self.detect_candlestick_patterns(df)

        # 将 NaN 值替换为0
        df.fillna(0, inplace=True)

        return df




if __name__ == "__main__":
    # 选好的股票代码 16个精选的股票和ETF基金指数
    # 读取要进行处理的股票代码
    with open('stock_codes.txt', 'r') as f:
        stock_codes = [line.strip() for line in f]

    # 配置日志并输出到文件
    logging.basicConfig(filename='logs/UpdateStrategy.log',
                        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    strategy_logger = LoggerManager('strategy_logger', 'logs/UpdateStrategy.log').get_logger()  # 创建策略日志对象

    # 读取数据连接信息的配置文件
    config_file = 'configs/configPolarDB.ini'

    # 数据库配置节名称
    section_basedata = 'basedata'
    section_strategy = 'strategy'
    section_forecast = 'forecast'
    section_trade = 'trade'

    # 读取表和更新表的后缀
    read_suffix = 'base'
    update_suffix = 'strategy'

    # 默认开始的更新时间
    default_time = '2014-01-01'

    # 实例化数据库连接池
    PoolDB_basedata = ConnectionPoolDB(section_basedata, config_file)  # 创建基础信息数据库连接池
    PoolDB_strategy = ConnectionPoolDB(section_strategy, config_file)  # 创建策略信息数据库连接池
    PoolDB_forecast = ConnectionPoolDB(section_forecast, config_file)  # 创建策略信息数据库连接池
    PoolDB_trade = ConnectionPoolDB(section_trade, config_file)  # 创建交易信息数据库连接池

    # 创建针对策略信息数据库的实例对象
    strategyDataManager = StrategyDataManager(PoolDB_basedata, PoolDB_strategy)  # 创建策略数据管理器
    dateControl = DateControl(PoolDB_basedata, PoolDB_strategy)  # 创建日期管理器# 创建日期管理器
    strategyDataFetcher = StrategyDataFetcher()  # 创建策略数据获取器
    scheduler = Scheduler()  # 创建调度器对象

    # 制定更新股票策略数据的任务
    def job_strategy(read_suffix: str, update_suffix: str, default_time: str):
        """
        任务函数用于更新股票策略数据到数据库中。
        功能:
        - 对每个股票执行以下操作：
            1. 在写入策略信息数据库中创建对应表格。
            2. 获取更新开始的时间。
            3. 从基础信息数据库中读取原始股价信息。
            4. 在原有的股价信息表中添加计算出的策略数据。
            5. 将生成的策略数据更新到数据库中。
        - 如果操作失败，则记录错误日志并打印错误信息。
        参数:
            - read_suffix: 读取策略数据所用的后缀。
            - update_suffix: 更新策略数据所用的后缀。
            - default_time: 默认时间。
        返回类型:
            无返回值。
        """
        # 循环遍历每张表
        for stock_code in stock_codes:
            try:
                strategyDataManager.create_tables(stock_code)  # 先在写入策略信息数据库创建对应表格

                update_time = dateControl.get_update_time(stock_code, read_suffix, update_suffix, default_time)  # 求得更新开始的时间

                stock_data = strategyDataManager.read_db(stock_code, update_time)  # 从求得的开始时间开始读取基础信息数据库里面的原始股价信息

                stock_data_df = strategyDataFetcher.signaling_strategy(
                    stock_data)  # 调用生成策略数据函数，在原有的股价信息表里面添加计算出来的生成策略数据

                strategyDataManager.update_stock_data(stock_code, stock_data_df, update_time)  # 将生成策略数据更新到数据库里面

                strategy_logger.info(
                    f"{stock_code} from {update_time} to {date.today()}  updated strategy successfully.")  # 打印日志信息
                print(f"{stock_code} from {update_time} to {date.today()}  updated strategy successfully.")  # 打印日志信息

            except Exception as e:

                strategy_logger.error(f"Error processing {stock_code}: {e}")  # 打印错误的日志信息
                print(f"Error processing {stock_code}: {e}")  # 打印错误的日志信息

    # 运行一次任务
    job_strategy(read_suffix, update_suffix, default_time)

    # 启动定时任务
    scheduler.Task(lambda: job_strategy(read_suffix, update_suffix, default_time))
