"""
GT-ECE8903 24spring
Zhuolin Li

基础工具库类
class ConnectionPoolDB: 用于读取配置文件和获取数据库配置。
    - 初始化配置管理器
    - 获取数据库配置
    - 获取数据库连接池的连接
    - 获取数据库连接池的游标

class DateControl: 用于控制数据更新的日期范围。
    - 获得数据库当中最新和最老的日期，以备后续使用
    - 根据数据库中最早和最新的日期确定数据更新的开始日期
    - 根据数据库中最早和最新的日期确定数据更新的结束日期
    - 获得指定日期范围内的交易日列表

class Scheduler: 用于定时任务调度。
    - 启动定时任务调度器

"""
import pymysql
from datetime import datetime, date, timedelta  # 改变这里的导入方式
import logging
import time
import configparser
from apscheduler.schedulers.background import BackgroundScheduler
from contextlib import contextmanager
from tenacity import retry, stop_after_attempt, wait_fixed
from dbutils.pooled_db import PooledDB


########################################################################################################################
# 数据库连接池管理器类，用于读取配置文件和获取数据库配置。
########################################################################################################################
class ConnectionPoolDB:
    def __init__(self, section: str, config_file: str):
        """
         初始化配置管理器。
         参数:
            - section (str): 配置节名称
            - config_file (str): 配置文件路径
         """
        self.config = configparser.ConfigParser()
        self.section = section
        self.config.read(config_file)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))  # 重试机制，最大尝试次数为3，每次尝试间隔为1秒
    def get_db_config(self) -> dict:
        """
        获取数据库配置。
        返回:
            - db_config (dict): 数据库配置字典
        """
        section_name = f'{self.section}'
        if section_name in self.config:
            return {
                'host': self.config[section_name]['host'],
                'port': int(self.config[section_name]['port']),
                'user': self.config[section_name]['user'],
                'password': self.config[section_name]['password'],
                'database': self.config[section_name]['database']
            }
        else:
            raise ValueError(f"Invalid environment specified: {self.section}")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))  # 重试机制，最大尝试次数为3，每次尝试间隔为1秒
    @contextmanager  # 上下文管理器，用于自动释放数据库连接
    def get_connection(self) -> pymysql.Connection:
        """
        获取数据库连接池的连接
        返回:
            - connection (Connection): 数据库连接
        """
        self.db_config = self.get_db_config()
        self.pool = PooledDB(
            creator=pymysql,
            mincached=1,
            maxcached=20,
            maxconnections=100,
            blocking=True,
            host=self.db_config['host'],
            port=self.db_config['port'],
            user=self.db_config['user'],
            password=self.db_config['password'],
            database=self.db_config['database']
        )
        connection = self.pool.connection()
        try:
            yield connection
        finally:
            try:
                connection.close()  # 释放数据库连接
            except Exception as e:
                # 处理关闭过程中的异常
                logging.error(f"Failed to close connection: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))  # 重试机制，最大尝试次数为3，每次尝试间隔为1秒
    @contextmanager  # 上下文管理器，用于自动释放数据库连接
    def get_cursor(self) -> pymysql.cursors:
        """
        获取数据库连接池的游标
        返回:
            - cursor (Cursor): 数据库游标
        """
        with self.get_connection() as connection:
            cursor = connection.cursor()
            try:
                yield cursor
            finally:
                try:
                    cursor.close()  # 释放数据库连接
                except Exception as e:
                    # 处理游标关闭过程中的异常
                    logging.error(f"Failed to close cursor: {e}")


########################################################################################################################
# 数据时间控制部分
#########################################################################################################################

class DateControl:
    def __init__(self, connection_pool_db_read, connection_pool_db_update):  # 传入数据库连接池对象
        self.pool_db_read = connection_pool_db_read
        self.pool_db_update = connection_pool_db_update

    # 获得数据库当中最新和最老的日期，以备后续使用
    @staticmethod
    def get_date_range(cursor, stock_code: str, suffix: str) -> tuple:
        try:
            # 构建查询最早和最晚日期的SQL语句
            query = f"SELECT MIN(Date), MAX(Date) FROM {stock_code}_{suffix}"

            # 执行查询
            cursor.execute(query)
            result = cursor.fetchone()
            if result:
                oldest_date = result[0] if result[0] is not None else None
                latest_date = result[1] if result[1] is not None else None
            else:
                oldest_date, latest_date = None, None
            # 返回一个包含最早和最晚日期的元组
            return oldest_date, latest_date

        except Exception as e:
            print(f"Error fetching date range for {stock_code}_{suffix}: {e}")
            return None, None

    # 根据数据库中最早和最新的日期确定数据更新的开始日期
    def get_start_date(self, stock_code: str) -> str:
        """
        获取股票数据更新的起始日期。
        参数:
            - stock_code (str): 股票代码
        返回:
            - start_date (str): 数据更新的起始日期，格式为"YYYY-MM-DD"
        该函数根据数据库中最早和最新的日期确定数据更新的起始日期。如果数据库中不存在数据，
        或者最早日期晚于特定日期（如2014-01-03），则起始日期设定为特定日期。如果股票代码是"LABU"，
        则起始日期为2015-05-28。
        """
        with self.pool_db_read.get_cursor() as cursor_read:
            # 获取数据库中最老的日期
            oldest_date, latest_date = self.get_date_range(cursor_read, stock_code, 'base')

        # 如果获取的最早日期为空，或者该日期晚于 "2014-01-03" 则设定起始日期为 "2014-01-03" LABU除外
        if oldest_date is None or (oldest_date > date(2014, 1, 3) and stock_code != "LABU"):
            return "2014-01-03"
        elif stock_code == "LABU" and oldest_date > date(2015, 5, 28):  # 专门为了LABU这个股票进行了优化
            return "2015-05-28"  # 这个股票只在2015年的5月28日上市

        if latest_date:
            # 设定起始日期为最晚日期的次日
            next_date = latest_date + timedelta(days=1)
            return next_date.strftime("%Y-%m-%d")

        # 如果无法获取最晚日期，则起始日期设为今天的前一天
        return (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    # 根据基础信息数据库和提取信息数据库的最老和最新时间来确定开始更新的时间
    def get_update_time(self, stock_code: str, read_suffix: str, update_suffix: str, default: str) -> date:
        """
        确定更新提取信息的开始时间，通过比较基础信息数据库和提取信息数据库的时间范围。
        参数:
            - stock_code (str): 股票代码，用于标识特定的股票。
            - read_suffix (str): 读取表后缀，用于识别读取信息的数据库表。
            - update_suffix (str): 写入表后缀，用于识别写入信息的数据库表。
            - default (str): 默认起始日期（格式为"YYYY-MM-DD"），当读取表中没有数据时使用。
        返回:
            - datetime.date: 确定的更新开始日期。这是一个日期对象，表示应从何时开始更新信息。
        主要逻辑：
            - 读取表最老时间 > 写入表最老时间（或为空）时，从基础信息最老时间开始更新。
            - 读取表最新时间 <= 写入表最新时间时，从写入表的最新时间减去一天开始更新。
            - 其他情况，如读取表时间被写入表完全覆盖时，则从最新的时间中谁最小减去一天开始更新。
        """
        with self.pool_db_read.get_cursor() as cursor_read:
            oldest_read, latest_read = self.get_date_range(cursor_read, stock_code, read_suffix)

        with self.pool_db_update.get_cursor() as cursor_update:
            oldest_update, latest_update = self.get_date_range(cursor_update, stock_code, update_suffix)

        default_date = datetime.strptime(default, "%Y-%m-%d").date()

        if isinstance(oldest_read, datetime):
            oldest_read = oldest_read.date()
        if isinstance(latest_read, datetime):
            latest_read = latest_read.date()
        if isinstance(oldest_update, datetime):
            oldest_update = oldest_update.date()
        if isinstance(latest_update, datetime):
            latest_update = latest_update.date()

        if oldest_read is None:
            return default_date
        elif oldest_update is None or oldest_read > oldest_update:
            return oldest_read
        elif latest_read is not None and (latest_update is None or latest_read <= latest_update):
            return latest_update - timedelta(days=1)
        else:
            return min(latest_read, latest_update) - timedelta(days=1)



########################################################################################################################
# 定时任务调度类
########################################################################################################################
class Scheduler:
    def __init__(self):
        """
        调度器，用于调度任务的执行。
        参数:
            - task (Task): 任务对象，包含要执行的任务逻辑
        该类包含一个 `DownloadTask` 方法，用于设置和启动任务调度。调度器会根据设定的每周特定日期和时间，执行任务对象中的 `job` 方法。
        """
        self.scheduler = BackgroundScheduler()  # 调度器对象

    def Task(self, job: callable):
        """
        任务函数，用于定义任务逻辑。
        参数:
            - job (callable): 任务逻辑，是一个可调用对象
        该类包含一个 `job` 方法，用于执行具体的任务逻辑。
        """
        days = ['tue', 'wed', 'thu', 'fri', 'sat']  # 周二到周五的每天的8:30执行一次任务
        for day in days:
            self.scheduler.add_job(job, 'cron', day_of_week=day, hour=8, minute=30)
        self.scheduler.start()

        try:
            while True:
                time.sleep(30)  # 每隔30秒检查一次 是否有任务需要执行
        except (KeyboardInterrupt, SystemExit):
            self.scheduler.shutdown()

########################################################################################################################
# 日志管理器类
########################################################################################################################
class LoggerManager:
    def __init__(self, name: str, log_file: str, level: int = logging.INFO):
        """
        初始化日志管理器
        参数:
            - name (str): 日志记录器的名称
            - log_file (str): 日志文件路径
            - level (int): 日志记录级别
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        handler = logging.FileHandler(log_file)
        handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def get_logger(self):
        """获取日志记录器"""
        return self.logger
