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

from apscheduler.schedulers.background import BackgroundScheduler
import yfinance as yf
import pandas as pd
import pymysql
import schedule
import time
import datetime




# 初始的数据库各股票表建表函数，后续检测如果数据库内无对应表格则执行该函数
def create_tables(cursor, stock_code):
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
    cursor.execute(create_table_query)


    # 设置触发器的执行语句
    create_trigger_query = f"""
     # 设置触发器，更新调整后的价格
     # 
    CREATE TRIGGER `update_adjclose_{stock_code}` BEFORE INSERT ON `{stock_code}_base` FOR EACH ROW BEGIN

    DECLARE adjCloseCalc FLOAT;
    
     IF NEW.StockSplits = 0 THEN
        SET adjCloseCalc = NEW.Close - NEW.Dividends;
     ELSE
        SET adjCloseCalc = (NEW.Close - NEW.Dividends);
     END IF;
        SET NEW.AdjClose = adjCloseCalc;
     END;   
    """
    # cursor.execute(create_trigger_query)


# 获得数据库当中最老的日期，以备后续使用
def get_oldest_date(cursor, stock_code):
    try:
        query = f"""
        SELECT MIN(Date) FROM {stock_code}_base
         """
        cursor.execute(query)
        result = cursor.fetchone()
        return result[0] if result and result[0] is not None else None
    except Exception as e:
        print(f"Error fetching oldest date for {stock_code}: {e}")
        return None

# 获得数据库当中最新的日期，以备后续使用
def get_latest_date(cursor, stock_code):
    try:
        query = f"""
        SELECT MAX(Date) FROM {stock_code}_base
        """
        cursor.execute(query)
        result = cursor.fetchone()
        return result[0].strftime("%Y-%m-%d") if result and result[0] is not None else None
    except Exception as e:
        print(f"Error fetching latest date for {stock_code}: {e}")
        return None

# 求数据库当中最老的时间，从而确定开始更新的时间
def get_start_date(cursor, stock_code):
    oldest_date_str = get_oldest_date(cursor, stock_code)

    if oldest_date_str is None or (oldest_date_str > datetime.date(2014, 1, 3) and stock_code != "LABU"):
        start_date = "2014-01-03"
    elif stock_code == "LABU" and oldest_date_str > datetime.date(2015, 5, 28):  # 专门为了LABU这个股票进行了优化
        start_date = "2015-05-28"  # 这个股票只在2015年的5月28日上市
    else:
        latest_date_str = get_latest_date(cursor, stock_code)

        if latest_date_str:
            latest_date = datetime.datetime.strptime(latest_date_str, "%Y-%m-%d").date()
            next_date = latest_date + datetime.timedelta(days=1)
            start_date = next_date.strftime("%Y-%m-%d")
        else:
            start_date = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    return start_date


# 更新股票的数据
def update_stock_data(cursor,db_connection, stock_code, start_date):

    stock = yf.Ticker(stock_code)
    end_date = datetime.date.today().strftime("%Y-%m-%d")

    if not start_date or start_date == end_date:  # 检查开始时间是否为空或者与结束时间相等
        start_date = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        # 相等则设为结束时间前一天

    print("\n\n",stock_code,"startDate",start_date,"endDate",end_date)

    stock_data = stock.history(start=start_date, end=end_date)

    print(stock_data)

    for index, row in stock_data.iterrows():
        insert_query = f"""
        # 
        # 
        INSERT INTO {stock_code}_base (Date, Open, High, Low, Close, Volume, Dividends, StockSplits) 
    VALUES ('{index.date()}', {row['Open']}, {row['High']}, {row['Low']}, {row['Close']}, {row['Volume']}, {row['Dividends']}, {row['Stock Splits']})
    ON DUPLICATE KEY UPDATE
        Open={row['Open']},
        High={row['High']},
        Low={row['Low']},
        Close={row['Close']},
        Volume={row['Volume']},
        Dividends={row['Dividends']},
        StockSplits={row['Stock Splits']}
        """
        cursor.execute(insert_query)
        db_connection.commit()






# 制定更新股票的任务
def job():
    db_connection = pymysql.connect(**db_config)
    cursor = db_connection.cursor()

    for stock_code in stock_codes:

        create_tables(cursor, stock_code)

        start_date = get_start_date(cursor, stock_code)

        update_stock_data(cursor, db_connection, stock_code, start_date)

    cursor.close()
    db_connection.close()


if __name__ == "__main__":

    # 选好的股票代码 16个精选的股票和ETF基金指数
    stock_codes = ['AMZN', 'BBH', 'FAS', 'GLD', 'GOOG', 'IVV', 'IWD', 'IWM', 'LABU', 'MS', 'QQQ', 'SPY', 'TQQQ', 'VGT',
                   'VUG', 'XLE']

    # 设置数据库的参数 连接数据库的信息（连接待生成策略数据的数据库）
    db_config = {"host": "10.5.0.18", "port": 3306, "user": "lizhuolin", "password": "123456", "database": "basedata"}

    # host="8.147.99.223", port="3306", user="lizhuolin", password="&a3sFD*432dfD!o0#3^dP2r2d!sc@"
    # host="10.5.0.18", user="lizhuolin", password="123456"

    # 立即更新数据
    job()

    # 使用APSchedule编排任务
    scheduler = BackgroundScheduler()

    # 设置定时任务（设置为每天早上八点半上班前半小时更新）
    days = ['tue', 'wed', 'thu', 'fri', 'sat']
    for day in days:
        scheduler.add_job(job, 'cron', day_of_week=day, hour=8, minute=30)
    scheduler.start()

    # 运行其他程序逻辑或进入主事件循环，防止程序退出
    try:
        while True:
            time.sleep(30)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()