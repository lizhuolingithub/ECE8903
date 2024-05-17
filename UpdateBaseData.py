"""
GT-ECE8903 24spring
Zhuolin Li

主要目的为利用雅虎金融python拓展包，对给定的相关股票爬取每日的股价信息，具体操作如下
1.连接本地的mySQL数据库
2.检测是否已经建好对应股票的表格，若否则重建
3.如已经建好对应股票的表格，则读取表格内最新的日期和最老的日期
4.根据日期信息来更新股票表格内缺失的数据（2014-1-1到系统当前日期）
5.设定为交易日收盘后定时更新，以及初次运行该脚本时更新一次
"""

import logging
import yfinance as yf
from datetime import datetime, date, timedelta
from BaseToolkit import ConnectionPoolDB, DateControl, Scheduler, LoggerManager
from tenacity import retry, stop_after_attempt, wait_fixed
import pandas as pd

########################################################################################################################
# 基础数据库SQL读写建表类
########################################################################################################################
class BaseDataManager:
    # 初始化函数，传入配置管理器
    def __init__(self, connection_pool_db):  # 传入数据库连接池对象
        self.connection_pool_db = connection_pool_db

    # 初始的数据库各股票表建表函数，后续检测如果数据库内无对应表格则执行该函数
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))  # 重试机制，最大尝试次数为3，每次尝试间隔为1秒
    def create_tables(self, stock_code: str):
        """
        创建股票数据表格。
        参数:
            - stock_code (str): 股票代码
        该函数通过执行SQL语句，在数据库中创建与股票相关的数据表格。
        """
        create_table_query = f"""
        # 创建表格SQL语句
        # 
        CREATE TABLE IF NOT EXISTS {stock_code}_base (
        `Date` date NOT NULL COMMENT '交易日期',
        `Open` float DEFAULT NULL COMMENT '开盘价格',
        `High` float DEFAULT NULL COMMENT '当日最高价',
        `Low` float DEFAULT NULL COMMENT '当日最低价',
        `Close` float DEFAULT NULL COMMENT '收盘价格',
        `AdjClose` float DEFAULT NULL COMMENT '调整收盘价',
        `Dividends` float DEFAULT NULL COMMENT '股息',
        `StockSplits` float DEFAULT NULL COMMENT '股票拆分',
        `Volume` bigint DEFAULT NULL COMMENT '交易量',
        `BullVolume` bigint DEFAULT NULL COMMENT '买入交易量',
        `SellVolume` bigint DEFAULT NULL COMMENT '卖出交易量',
        PRIMARY KEY (`Date` DESC),
        UNIQUE KEY `date` (`Date`) USING BTREE
        )

        """
        with self.connection_pool_db.get_connection() as connection:  # 获取数据库连接
            with connection.cursor() as cursor:
                cursor.execute(create_table_query)
                connection.commit()

    # 更新股票的数据
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    # 重试机制，最大尝试次数为3，每次尝试间隔为1秒
    def update_stock_data(self, stock_code: str, stock_data: pd.DataFrame):
        """
        更新股票数据。
        参数:
            - stock_code (str): 股票代码
            - stock_data (DataFrame): 股票数据DataFrame
        该函数将给定的股票数据插入或更新到对应股票的数据表格中。如果股票数据表格已存在，将数据插入；如果不存在，先创建表格再插入数据。
        """
        # 准备批量插入的SQL语句
        insert_query = f"""
        INSERT INTO {stock_code}_base (Date, Open, High, Low, Close, Volume, Dividends, StockSplits) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        Open=VALUES(Open),
        High=VALUES(High),
        Low=VALUES(Low),
        Close=VALUES(Close),
        Volume=VALUES(Volume),
        Dividends=VALUES(Dividends),
        StockSplits=VALUES(StockSplits)
        """
        # 创建一个元组列表，包含所有要插入的数据
        data_to_insert = [
            (
                index.strftime('%Y-%m-%d'), row['Open'], row['High'], row['Low'], row['Close'], row['Volume'],
                row['Dividends'],
                row['Stock Splits'])
            for index, row in stock_data.iterrows()
        ]

        # 使用executemany进行批量插入操作
        if data_to_insert:  # 确保有数据插入
            with self.connection_pool_db.get_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.executemany(insert_query, data_to_insert)
                    connection.commit()  # 一次性提交所有插入操作


########################################################################################################################
# 主要业务逻辑获取股票信息类
#########################################################################################################################
class BaseDataFetcher:
    def __init__(self):
        pass

    @staticmethod
    def get_stock_data(stock_code: str, start_date: str) -> pd.DataFrame:
        """
        获取股票数据。
        参数:
            - stock_code (str): 股票代码
            - start_date (str): 开始日期，格式为"YYYY-MM-DD"，默认为None
        返回:
            - stock_data (DataFrame): 股票数据DataFrame
        """
        stock = yf.Ticker(stock_code)
        end_date = date.today().strftime("%Y-%m-%d")

        if not start_date or start_date == end_date:  # 检查开始时间是否为空或者与结束时间相等
            start_date = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
            # 如果相等则设为结束时间前一天

        stock_data = stock.history(start=start_date, end=end_date)

        return stock_data


########################################################################################################################
# 主函数
########################################################################################################################


if __name__ == "__main__":

    # 选好的股票代码 16个精选的股票和ETF基金指数
    # 读取要进行处理的股票代码
    with open('stock_codes.txt', 'r') as f:
        stock_codes = [line.strip() for line in f]

    # 读取数据连接信息的配置文件
    config_file = 'configs/configPolarDB.ini'

    # 数据库配置节名称
    section_basedata = 'basedata'
    section_strategy = 'strategy'
    section_forecast = 'forecast'
    section_trade = 'trade'

    # 配置日志并输出到文件
    logging.basicConfig(filename='logs/UpdateBaseData.log',
                        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    base_logger = LoggerManager('base_logger', 'logs/UpdateBaseData.log', ).get_logger()

    # 实例化数据库连接池
    PoolDB_basedata = ConnectionPoolDB(section_basedata, config_file)  # 创建基础信息数据库连接池
    PoolDB_strategy = ConnectionPoolDB(section_strategy, config_file)  # 创建策略信息数据库连接池
    PoolDB_forecast = ConnectionPoolDB(section_forecast, config_file)  # 创建策略信息数据库连接池
    PoolDB_trade = ConnectionPoolDB(section_trade, config_file)  # 创建交易信息数据库连接池

    # 创建针对基础信息数据库的实例对象
    baseDatatableManager = BaseDataManager(PoolDB_basedata)  # 创建数据库表格管理器
    dateControl = DateControl(PoolDB_basedata, PoolDB_basedata)  # 创建日期管理器
    baseDataFetcher = BaseDataFetcher()  # 创建股票数据获取器
    scheduler = Scheduler()  # 创建调度器对象

    # 制定更新股票基础数据的任务
    def job_basedata():
        """
        任务函数用于更新股票基础数据到数据库中。
        功能:
        - 对每个股票执行以下操作：
            1. 创建股票数据表格。
            2. 获取起始日期。
            3. 获取股票数据。
            4. 将股票数据更新到数据库中。
        - 如果操作失败，则记录错误日志并打印错误信息。
        参数:
            无参数。
        返回类型:
            无返回值。
        """
        for stock_code in stock_codes:

            try:
                baseDatatableManager.create_tables(stock_code)  # 创建股票数据表格

                update_time = dateControl.get_start_date(stock_code)  # 获取起始日期

                stock_data = baseDataFetcher.get_stock_data(stock_code, update_time)  # 获取股票数据

                baseDatatableManager.update_stock_data(stock_code, stock_data)  # 更新股票数据到数据库中

                base_logger.info(
                    f"{stock_code} from {update_time} to {date.today()}  updated basedata successfully.")  # 打印日志信息
                print(f"{stock_code} from {update_time} to {date.today()}  updated basedata successfully.")

            except Exception as e:

                base_logger.error(f"Error processing {stock_code}: {e}")  # 打印错误的日志信息
                print(f"Error processing {stock_code}: {e}")  # 打印错误的日志信息


    # 立即更新数据
    job_basedata()

    # 启动定时任务
    scheduler.Task(lambda: job_basedata)
