"""
GT-ECE8903 24spring StockPlot
Zhuolin Li

建立预测表格的一个小脚本
1 连接数据库
2 循环遍历16个股并建立表格
3 可能还会需要触发器

要分别建3个表 5 20 60

然后触发器加上 avg min max std标准差

会有三种情况，下面是day=60的情况

预测股价数据库表结构设计：
name: {stock_code}_predict_{predict_day}
列名称       列类型   字段解释
mainID      BIGINT  主键ID序列号（自动递增）
Date        DATE    股价日期
Stock       VARCHAR 股票代码
PredictDate DATE    预测日期
Algorithm   VARCHAR 预测算法
PredictDay  INT     预测天数
Close       FLOAT   收盘价格
Day1        FLOAT   预测第1天的价格
Day2        FLOAT   预测第2天的价格
.......
Day59       FLOAT   预测第59天的价格
Day60       FLOAT   预测第60天的价格
PredictAvg  FLOAT   预测股价的平均值
PredictMax  FLOAT   预测股价的最大值
PredictMin  FLOAT   预测股价的最小值
PredictStd  FLOAT   预测股价的标准差
PredictProfit FLOAT 预测区间的最大收益
PredictLoss FLOAT   预测区间的最大亏损
PredictSlope FLOAT  预测序列的斜率
PredictIntercept FLOAT 预测序列的截距

--（Day1到Day60有60个字段，预测未来1到60天的价格）

"""
import pymysql



"""
Creates a single table for each stock and prediction day range that includes all predictive metrics and daily predictions.
This 'join' version creates a comprehensive table that can handle dynamic day-by-day prediction storage up to the specified day (e.g., 60 days).

Args:
    cursor (Cursor): Database cursor to execute the query.
    stock_code (str): Stock code for which the table is created.
    predict_day (int): Number of prediction days (e.g., 5, 20, 60).
"""
# 创建总数据库预测表函数
# TODO 目前不知道是要用这种365天动态的还是放一个列里面存字符串集合全部的价格，然后还是就60天的放5 20 60的数据，但是那个自增的表格我得研究一下，可能不能插入统样时间股价的数据
def create_tables_join(cursor, stock_code, predict_day):
    # 动态生成每一天的列名和数据类型
    day_columns = ', '.join([f'Day{i} FLOAT' for i in range(1, predict_day + 1)])

    create_table_query = f"""
   -- 创建预测数据表
   -- {predict_day}天的情况  
    CREATE TABLE IF NOT EXISTS {stock_code}_predict_{predict_day}d (
        mainID BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID序列号',
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
        UNIQUE KEY `mainID` (`mainID`) USING BTREE,
        KEY `date` (`Date` DESC) USING BTREE
    );
    """
    try:
        cursor.execute(create_table_query)
        db_connection.commit()
        print(f"Table_join {stock_code}_predict_{predict_day}d created successfully.")

    # 加上异常处理，如果有意外就回滚事务并且输出异常
    except Exception as e:
        print(f"Failed to create Table_join {stock_code}_predict_{predict_day}d : {e}")
        db_connection.rollback()


"""
This function creates individual tables for each prediction day range but does not include the full range of days in one table.
This 'split' version focuses on segregated storage, making it suitable for scenarios where predictions are stored and managed separately.

Args:
    cursor (Cursor): Database cursor to execute the query.
    stock_code (str): Stock code for which the table is created.
    predict_day (int): Number of prediction days (e.g., 5, 20, 60).
"""
# 创建分数据库预测表函数
def create_tables_split(cursor, stock_code, predict_day):
    # 动态生成每一天的列名和数据类型
    day_columns = ', '.join([f'Day{i} FLOAT' for i in range(1, predict_day + 1)])

    create_table_query = f"""
   -- 创建预测数据表
   -- {predict_day}天的情况  
    CREATE TABLE IF NOT EXISTS {stock_code}_predict_{predict_day}d (
        Date DATE PRIMARY KEY COMMENT '股价日期',
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
    try:
        cursor.execute(create_table_query)
        db_connection.commit()
        print(f"Table_split {stock_code}_predict_{predict_day}d created successfully.")

    # 加上异常处理，如果有意外就回滚事务并且输出异常
    except Exception as e:
        print(f"Failed to create Table_split {stock_code}_predict_{predict_day}d : {e}")
        db_connection.rollback()

"""
Creates or replaces a database view that combines data from multiple prediction period tables for a given stock.
This view facilitates easy access and analysis across different prediction periods by consolidating the data into a single view.

Args:
    cursor (Cursor): The database cursor to execute the query.
    stock_code (str): The stock symbol for which the view will be created.
    db_connection (Connection): The database connection to commit the transaction.
"""
# 创建视图的函数
def setview_tables(cursor, stock_code, db_connection):

    create_view_query = f"""
    -- 创建视图
    -- 创建将三个表的不同地方融合到一起来
    -- 创建视图，融合不同周期的预测数据和基本交易数据
    -- 创建视图，整合不同周期的预测数据、基本交易数据和策略指标数据
CREATE VIEW {stock_code}_combined AS
SELECT
    a.Stock,
    base.Date,
    base.Open,
    base.High,
    base.Low,
    base.Close,
    base.AdjClose,
    base.Volume,
    a.Algorithm,
    a.PredictMax AS PredictMax5d,
    a.PredictMin AS PredictMin5d,
    a.PredictAvg AS PredictAvg5d,
    a.PredictStd AS PredictStd5d,
    a.PredictProfit AS PredictProfit5d,
    a.PredictLoss AS PredictLoss5d,
    a.PredictSlope AS PredictSlope5d,
    a.PredictIntercept AS PredictIntercept5d,
    b.PredictMax AS PredictMax20d,
    b.PredictMin AS PredictMin20d,
    b.PredictAvg AS PredictAvg20d,
    b.PredictStd AS PredictStd20d,
    b.PredictProfit AS PredictProfit20d,
    b.PredictLoss AS PredictLoss20d,
    b.PredictSlope AS PredictSlope20d,
    b.PredictIntercept AS PredictIntercept20d,
    c.PredictMax AS PredictMax60d,
    c.PredictMin AS PredictMin60d,
    c.PredictAvg AS PredictAvg60d,
    c.PredictStd AS PredictStd60d,
    c.PredictProfit AS PredictProfit60d,
    c.PredictLoss AS PredictLoss60d,
    c.PredictSlope AS PredictSlope60d,
    c.PredictIntercept AS PredictIntercept60d,
    s.FastAvg,
    s.SlowAvg,
    s.MACD,
    s.SignalLine,
    s.MA,
    s.BollingerUp,
    s.BollingerDown,
    s.BollingerChannel,
    s.RSI,
    s.RSIChannel,
    s.Doji,
    s.ADX,
    s.MACDsign,
    s.K,
    s.D,
    s.KDsign,
    s.CCI,
    s.ROC,
    s.WilliamsR,
    s.OBV,
    s.OBV20ma,
    s.OBV2de,
    s.Klinger,
    s.CMF,
    s.ComplexDoji
FROM
    basedata.{stock_code}_base base
    RIGHT JOIN {stock_code}_predict_5d a ON base.Date = a.Date
    RIGHT JOIN {stock_code}_predict_20d b ON base.Date = b.Date AND a.Stock = b.Stock AND a.Algorithm = b.Algorithm
    RIGHT JOIN {stock_code}_predict_60d c ON base.Date = c.Date AND a.Stock = c.Stock AND a.Algorithm = c.Algorithm
    LEFT JOIN strategy.{stock_code}_strategy s ON base.Date = s.Date;

    """
    try:
        cursor.execute(create_view_query)
        db_connection.commit()
        print(f"View {stock_code}_forecast_combined created successfully.")

    # 加上异常处理，如果有意外就回滚事务并且输出异常
    except Exception as e:
        print(f"Failed to create view for {stock_code}: {e}")
        db_connection.rollback()


# 制定更新股票的任务
if __name__ == "__main__":

    # 选好的股票代码 16个精选的股票和ETF基金指数
    stock_codes = ['AMZN', 'BBH', 'FAS', 'GLD', 'GOOG', 'IVV', 'IWD', 'IWM', 'LABU', 'MS', 'QQQ', 'SPY', 'TQQQ', 'VGT',
                   'VUG', 'XLE']

    # 设置数据库的参数 连接数据库的信息（连接待生成策略数据的数据库）
    db_config = {"host": "10.5.0.11", "port": 3306, "user": "lizhuolin", "password": "123456", "database": "forecast"}

    # "host": "10.5.0.11", "port": 3306, "user": "lizhuolin", "password": "123456"
    # "host": "8.147.99.223", "port": 3306, "user": "lizhuolin", "password": "&a3sFD*432dfD!o0#3^dP2r2d!sc@"

    # 建立不同预测时间的预测表
    predict_days = [5, 20, 60]

    db_connection = pymysql.connect(**db_config)  # 创建连接
    cursor = db_connection.cursor()

    for stock_code in stock_codes:


        setview_tables(cursor, stock_code, db_connection)# 创建视图

        """
        for predict_day in predict_days:
            # 如果检测到连接的是forecast表的话那就执行join创建函数，如果不是则执行分开的创建函数

            create_tables_join(cursor, stock_code, predict_day)

            # create_tables_split(cursor, stock_code, predict_day)
        """

    cursor.close()
    db_connection.close()
