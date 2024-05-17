"""
GT-ECE8903 24spring
Zhuolin Li

"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from tenacity import retry, stop_after_attempt, wait_fixed
from BaseToolkit import ConnectionPoolDB, DateControl, Scheduler, LoggerManager


########################################################################################################################
# 数据库读写建表操作部分
#########################################################################################################################
class TradeDataManager:
    # 初始化函数，传入配置管理器
    def __init__(self, connection_pool_db_forecast, connection_pool_db_trade):  # 传入数据库连接池对象
        self.pool_db_forecast = connection_pool_db_forecast
        self.pool_db_trade = connection_pool_db_trade

    # 初始的待生成策略数据各股票表建表函数，检测如果数据库内无对应表格则执行该函数
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))  # 重试机制，最大尝试次数为3，每次尝试间隔为1秒
    def create_tables(self, stock_code: str):
        # 创建交易表
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
        with self.pool_db_trade.get_connection() as connection:  # 获取数据库连接
            with connection.cursor() as cursor:
                cursor.execute(create_table_query)
                connection.commit()

    # 读取原始预测的数据库里面的视图的数据
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))  # 重试机制，最大尝试次数为3，每次尝试间隔为1秒
    def read_db(self, stock_code: str, start_time: datetime, algorithm: str) -> pd.DataFrame:
        with self.pool_db_forecast.get_cursor() as cursor:  # 获取数据库连接
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
                    ORDER BY Date ASC;
                """
                # 执行参数化查询，防止SQL注入
                cursor.execute(query, (formatted_time, algorithm, stock_code))  # 参数化查询
                result = cursor.fetchall()

                # 将查询结果转换为 DataFrame，并指定列名
                df = pd.DataFrame(result, columns=[
                    'Date', 'Close', 'PredictProfit5d', 'PredictLoss5d', 'PredictProfit20d', 'PredictLoss20d',
                    'PredictSlope20d',
                    'PredictProfit60d', 'PredictLoss60d', 'PredictSlope60d', 'BollingerChannel', 'MACDsign', 'Doji',
                    'RSIChannel', 'ComplexDoji', 'KDsign', 'CCI', 'WilliamsR', 'OBV', 'OBV20ma', 'OBV2de'
                ])

                return df
            except Exception as e:
                print(f"Error fetching {stock_code}_combined: {e}")

    # 将计算好的数据更新到交易数据表中
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))  # 重试机制，最大尝试次数为3，每次尝试间隔为1秒
    def update_db(self, stock_code: str, stock_table_df: pd.DataFrame, gotime: datetime):
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
                              row['Sell'], row['BuyComment'], row['SellComment'], row['ForecastAlgorithm'],
                              row['Profit5day'],
                              row['Loss5day'], row['Profit20day'], row['Loss20day'], row['Profit60day'],
                              row['Loss60day'],
                              row['Transaction'], row['Returns'])
                data_to_insert.append(data_tuple)

        # 执行批量插入
        if data_to_insert:
            with self.pool_db_trade.get_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.executemany(insert_query, data_to_insert)
                    connection.commit()  # 一次性提交所有插入操作


########################################################################################################################
# 主要功能数据计算处理部分
#########################################################################################################################
class TradeDataFetcher:
    def __init__(self):
        pass

    """
    计算并添加复杂Doji状态到DataFrame中。
    参数-df : pd.DataFrame 需要包含 `ComplexDoji` 和 `BollingerPct` 列的DataFrame。
    返回-df : pd.DataFrame  更新后的DataFrame，包含新的列 `D`。
    """

    def calculate_complex_doji(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the 'D' column based on the 'ComplexDoji' and 'BollingerPct' columns.
        Args:
            - df (DataFrame): DataFrame that contains the 'ComplexDoji' and 'BollingerPct' columns.
        Returns:
            - DataFrame: Updated DataFrame with a new column 'D'.
        """
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
            3,  # 三线打击
            2,  # 晨曦之星 and 布林带低于40% 买
            0,  # 没有doji
            -2,  # 黄昏之星 or 两只乌鸦
            -3  # 三只乌鸦
        ]

        # Apply conditions to determine the 'C' column
        df['C'] = np.select(conditions, choices, default=0)  # Default to 0 (no action) if no conditions are met

        return df['C']

    # 根据交易策略模拟交易计算收益率

    def trade_simulation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate trades based on Transaction signals and calculate the cumulative return.
        Args:
            - df (DataFrame): DataFrame that contains the Date, Close prices, and Transaction signals.
        Returns:
            - DataFrame: Updated DataFrame with additional columns for portfolio values and returns.
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
    def add_trade_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Update the DataFrame with a new column 'Point' based on the combined state of technical indicators.
        Args:
            - df (DataFrame): DataFrame containing the necessary columns for evaluating trade signals.
        Returns:
            - DataFrame: Updated DataFrame with a new column 'Point'.
        """
        # 定义买入条件
        buy_conditions = [
            df['A'] == 1,  # 1 macd下穿上信号线 买
            df['B'] == 1,  # 2 晨曦之星 and 布林带低于40% 买
            df['C'] == 1,  # 1 K线从下穿上D线 买
            df['D'] == 2,  # 2 RSIChannel 0%-20% 买
            df['E'] == 2  # 2 CCI<-200 买
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
    def add_trade_signals_comments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Update the DataFrame with comments based on trade signal conditions.
        Args:
            - df (DataFrame): DataFrame containing the necessary columns for evaluating trade signals.
        Returns:
            - DataFrame: Updated DataFrame with 'BuyComment' and 'SellComment' columns added.
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

        def get_signals(row: pd.Series, conditions: dict) -> str:
            # Extracting all descriptions where the condition is met
            signals = [desc for desc, col, value in conditions.values() if row[col] == value]
            return "; ".join(signals)  # Join all descriptions into a single string

        # Apply the function to generate buy and sell comments
        df['BuyComment'] = df.apply(lambda row: get_signals(row, buy_conditions), axis=1)
        df['SellComment'] = df.apply(lambda row: get_signals(row, sell_conditions), axis=1)

        return df[['BuyComment', 'SellComment']]

    # 结合得分买点和卖点以及预测的结果设置最终地买、卖或持有信号

    def calculate_transaction_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Update the DataFrame with a new column 'Transaction' based on the combined state of technical indicators.
        Args:
            - df (DataFrame): DataFrame containing the necessary columns for evaluating trade signals.
        Returns:
            - DataFrame: Updated DataFrame with a new column 'Transaction'.
        """
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

    def update_kd_bollinger_state(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Update the DataFrame with a new column 'A' based on the combined state of KD line crossings and Bollinger Band percentiles.
        Args:
            - df (DataFrame): DataFrame containing the columns 'KDsign' and 'BollingerPercent'.
        Returns:
            - DataFrame: Updated DataFrame with a new column 'A'.
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

    def update_macd_bollinger_state(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Update the DataFrame with a new column 'A' based on the combined state of KD line crossings and Bollinger Band percentiles.
        Args:
            - df (DataFrame): DataFrame containing the columns 'KDsign' and 'BollingerPercent'.
        Returns:
            - DataFrame: Updated DataFrame with a new column 'A'.
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

    def update_obv_state(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Update the DataFrame with a new column 'A' based on the combined state of KD line crossings and Bollinger Band percentiles.
        Args:
            - df (DataFrame): DataFrame containing the columns 'KDsign' and 'BollingerPercent'.
        Returns:
            - DataFrame: Updated DataFrame with a new column 'A'.
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
    def signaling_trade(self, stock_table: pd.DataFrame, algorithm: str) -> pd.DataFrame:
        """
        Update the DataFrame with a new column 'Transaction' based on the combined state of technical indicators.
        Args:
            - stock_table (DataFrame): DataFrame containing the necessary columns for evaluating trade signals.
            - algorithm (str): 预测算法名称。
        Returns:
            - DataFrame: Updated DataFrame with a new column 'Transaction'.
        """
        df = stock_table.copy()
        df.set_index('Date', inplace=True)

        # 状态A - KD线结合 布林带
        df['A'] = self.update_kd_bollinger_state(df)

        # 状态B - MACDsign 结合 布林带
        df['B'] = self.update_macd_bollinger_state(df)

        # 状态C - Complex Doji
        df['C'] = self.calculate_complex_doji(df)

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
        df[['Buy', 'Sell']] = self.add_trade_signals(df)

        # 生成买入和卖出信号描述并以分号连接分别放置在同一个字符串内
        df[['BuyComment', 'SellComment']] = self.add_trade_signals_comments(df)

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
        df['Transaction'] = self.calculate_transaction_signals(df)

        # 根据上述的买卖信号计算出回报率
        df['Returns'] = self.trade_simulation(df)

        # 将 NaN 值替换为0
        df.fillna(0, inplace=True)

        return df


if __name__ == "__main__":
    # 读取要进行处理的股票代码
    with open('stock_codes.txt', 'r') as f:
        stock_codes = [line.strip() for line in f]

    # 配置日志并输出到文件
    logging.basicConfig(filename='logs/UpdateTrade.log',
                        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    trade_logger = LoggerManager('trade_logger', 'logs/UpdateTrade.log').get_logger()  # 创建交易日志记录器对象

    # 读取数据连接信息的配置文件
    config_file = 'configs/configPolarDB.ini'

    # 数据库配置节名称
    section_basedata = 'basedata'
    section_strategy = 'strategy'
    section_forecast = 'forecast'
    section_trade = 'trade'

    # 设置读取和更新两个数据库表名的后缀
    read_suffix = 'combined'
    update_suffix = 'trade'

    default_time = '2019-01-01'  # 默认的更新时间
    algorithm = 'LSTM'  # 默认的预测算法

    # 实例化数据库连接池
    PoolDB_basedata = ConnectionPoolDB(section_basedata, config_file)  # 创建基础信息数据库连接池
    PoolDB_strategy = ConnectionPoolDB(section_strategy, config_file)  # 创建策略信息数据库连接池
    PoolDB_forecast = ConnectionPoolDB(section_forecast, config_file)  # 创建策略信息数据库连接池
    PoolDB_trade = ConnectionPoolDB(section_trade, config_file)  # 创建交易信息数据库连接池

    # 创建针对交易信息数据库的实例对象
    tradeDataManager = TradeDataManager(PoolDB_forecast, PoolDB_trade)  # 创建交易数据管理器对象
    dateControl = DateControl(PoolDB_forecast, PoolDB_trade)  # 创建日期控制对象
    tradeDataFetcher = TradeDataFetcher()  # 创建交易数据获取器对象
    scheduler = Scheduler()  # 创建调度器对象

    # 制定更新股票交易数据的任务
    def job_trade(read_suffix: str, update_suffix: str, default_time: str, algorithm: str):
        """
        任务函数用于更新股票交易数据到数据库中。
        功能:
        - 对每个股票执行以下操作：
            1. 在写入交易信息数据库中创建对应表格。
            2. 根据基础信息数据库和提取信息数据库的最老和最新时间来确定开始更新的时间。
            3. 从基础信息数据库中读取原始股价信息。
            4. 在原有的股价信息表中添加计算出的交易数据。
            5. 将生成的交易数据更新到数据库中。
        - 如果操作失败，则记录错误日志并打印错误信息。
        参数:
            - read_suffix: 读取交易数据所用的后缀。
            - update_suffix: 更新交易数据所用的后缀。
            - default_time: 默认时间。
            - algorithm: 交易算法。
        返回类型:
            无返回值。
        """
        # 循环遍历每张表
        for stock_code in stock_codes:
            try:
                # 先在写入策略信息数据库创建对应表格
                tradeDataManager.create_tables(stock_code)

                # 根据基础信息数据库和提取信息数据库的最老和最新时间来确定开始更新的时间
                update_time = dateControl.get_update_time(stock_code, read_suffix, update_suffix, default_time)

                # 从求得的开始时间开始读取基础信息数据库里面的原始股价信息
                stock_data = tradeDataManager.read_db(stock_code, update_time, algorithm)

                # 调用生成策略数据函数，在原有的股价信息表里面添加计算出来的生成策略数据
                stock_data_df = tradeDataFetcher.signaling_trade(stock_data, algorithm)

                # 将生成数据更新到数据库里面
                tradeDataManager.update_db(stock_code, stock_data_df, update_time)

                trade_logger.info(
                    f"{stock_code} from {update_time} to {date.today()} updated trade successfully.")  # 打印日志信息
                print(f"{stock_code} from {update_time} to {date.today()}  updated trade successfully.")  # 打印日志信息

            except Exception as e:

                trade_logger.error(f"Error processing {stock_code}: {e}")  # 打印错误的日志信息
                print(f"Error processing {stock_code}: {e}")  # 打印错误的日志信息


    # 调用任务主函数
    job_trade(read_suffix, update_suffix, default_time, algorithm)

    # 启动定时任务
    scheduler.Task(lambda: job_trade(read_suffix, update_suffix, default_time, algorithm))
