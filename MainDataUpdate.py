from BaseToolkit import ConnectionPoolDB, DateControl, Scheduler, LoggerManager
from UpdateBaseData import BaseDataManager, BaseDataFetcher
from UpdateStrategy import StrategyDataManager, StrategyDataFetcher
from UpdateTrade import TradeDataManager, TradeDataFetcher
from datetime import datetime, date, timedelta


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

            base_logger.info(f"{stock_code} from {update_time} to {date.today()}  updated basedata successfully.")  # 打印日志信息
            print(f"{stock_code} from {update_time} to {date.today()}  updated basedata successfully.")

        except Exception as e:

            base_logger.error(f"Error processing {stock_code}: {e}")  # 打印错误的日志信息
            print(f"Error processing {stock_code}: {e}")  # 打印错误的日志信息


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

            stock_data_df = strategyDataFetcher.signaling_strategy(stock_data)  # 调用生成策略数据函数，在原有的股价信息表里面添加计算出来的生成策略数据

            strategyDataManager.update_stock_data(stock_code, stock_data_df, update_time)  # 将生成策略数据更新到数据库里面

            strategy_logger.info(f"{stock_code} from {update_time} to {date.today()}  updated strategy successfully.")  # 打印日志信息
            print(f"{stock_code} from {update_time} to {date.today()}  updated strategy successfully.")  # 打印日志信息

        except Exception as e:

            strategy_logger.error(f"Error processing {stock_code}: {e}")  # 打印错误的日志信息
            print(f"Error processing {stock_code}: {e}")  # 打印错误的日志信息


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

            trade_logger.info(f"{stock_code} from {update_time} to {date.today()} updated trade successfully.")  # 打印日志信息
            print(f"{stock_code} from {update_time} to {date.today()}  updated trade successfully.")  # 打印日志信息

        except Exception as e:

            trade_logger.error(f"Error processing {stock_code}: {e}")  # 打印错误的日志信息
            print(f"Error processing {stock_code}: {e}")  # 打印错误的日志信息


########################################################################################################################
# 主函数
########################################################################################################################


if __name__ == "__main__":
    # 读取要进行处理的股票代码 选好的股票代码 16个精选的股票和ETF基金指数
    with open('stock_codes.txt', 'r') as f:
        stock_codes = [line.strip() for line in f]  # 读取股票代码列表

    # 读取数据连接信息的配置文件
    config_file = 'configs/configPolarDB.ini'

    # 数据库配置节名称
    section_basedata = 'basedata'  # 基础信息数据库配置节名称
    section_strategy = 'strategy'  # 策略信息数据库配置节名称
    section_forecast = 'forecast'  # 预测信息数据库配置节名称
    section_trade = 'trade'  # 交易信息数据库配置节名称

    default_time_strategy = '2014-01-01'  # 策略数据库初始默认的更新时间
    default_time_trade = '2019-01-01'  # 交易数据库初始默认的更新时间
    algorithm = 'LSTM'  # 默认的预测算法

    # 策略数据库 读取表和更新表的后缀
    strategy_read_suffix = 'base'
    strategy_update_suffix = 'strategy'

    # 交易信息数据库 读取和更新两个数据库表名的后缀
    trade_read_suffix = 'combined'
    trade_update_suffix = 'trade'

    # 创建不同的日志记录器
    base_logger = LoggerManager('base_logger', 'logs/UpdateBaseData.log').get_logger()  # 配置基础信息数据库日志
    strategy_logger = LoggerManager('strategy_logger', 'logs/UpdateStrategy.log').get_logger()  # 配置策略信息数据库日志
    trade_logger = LoggerManager('trade_logger', 'logs/UpdateTrade.log').get_logger()  # 配置交易信息数据库日志

    # 实例化数据库连接池
    PoolDB_basedata = ConnectionPoolDB(section_basedata, config_file)  # 创建基础信息数据库连接池
    PoolDB_strategy = ConnectionPoolDB(section_strategy, config_file)  # 创建策略信息数据库连接池
    PoolDB_forecast = ConnectionPoolDB(section_forecast, config_file)  # 创建策略信息数据库连接池
    PoolDB_trade = ConnectionPoolDB(section_trade, config_file)  # 创建交易信息数据库连接池

    # 创建基础工具包文件的实例对象
    dateControl = DateControl(PoolDB_basedata, PoolDB_basedata)  # 创建日期管理器
    scheduler = Scheduler()  # 创建调度器对象

    # 创建针对基础信息数据库的实例对象
    baseDatatableManager = BaseDataManager(PoolDB_basedata)  # 创建数据库表格管理器
    baseDataFetcher = BaseDataFetcher()  # 创建股票数据获取器

    # 创建针对策略信息数据库的实例对象
    strategyDataManager = StrategyDataManager(PoolDB_basedata, PoolDB_strategy)  # 创建策略数据管理器
    strategyDataFetcher = StrategyDataFetcher()  # 创建策略数据获取器

    # 创建针对交易信息数据库的实例对象
    tradeDataManager = TradeDataManager(PoolDB_forecast, PoolDB_trade)  # 创建交易数据管理器对象
    tradeDataFetcher = TradeDataFetcher()  # 创建交易数据获取器对象

    # 启动定时基础信息更新任务
    scheduler.Task(lambda: job_basedata())

    # 启动定时策略信息更新任务
    scheduler.Task(lambda: job_strategy(strategy_read_suffix, strategy_update_suffix, default_time_strategy))

    # 启动定时交易信息更新任务
    scheduler.Task(lambda: job_trade(trade_read_suffix, trade_update_suffix, default_time_trade, algorithm))

#  TODO: 改善一下这个里面的日志的用法，看看怎么复用
#  TODO: 三个不同job的注释也要去写一下，让别人看懂


#  TODO: 看看是否需要把job给类与对象封装了（继承什么的）然后在这个文件里面引用
#  TODO: forecast也可以改成类与对象，简单一点先， git要上传了，然后github code为啥不知道不能用，得找下原因
#  TODO: {date.today()} 这个逻辑有点问题，看看是否需要改一下，尽量不要在这里用from datetime import datetime, date, timedelta
#  TODO: 在主函数这里创建对象，然后不经过函数参数传递的方式，以全局变量的方式调用是否合理的，看看是否需要修改一下（job）
#  TODO：看看是否需要添加类型提示，这样比较工程化一点
