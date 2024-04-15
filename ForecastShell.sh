#!/bin/bash

# 定义数据库连接参数
DB_NAME_FORECAST="forecast_gbm"

# 定义共享参数
ALGORITHM="GBM"
START="2016-01-01"
PREDICT="2023-01-01"
END="2023-12-31"

STOCK_CODES5=("AMZN" "BBH" "FAS" "GLD" "GOOG" "IVV" "IWD" "IWM" "LABU" "MS" "QQQ" "SPY" "TQQQ" "VGT" "VUG" "XLE")

STOCK_CODES20=("AMZN" "BBH" "FAS" "GLD" "GOOG" "IVV" "IWD" "IWM" "LABU" "MS" "QQQ" "SPY" "TQQQ" "VGT" "VUG" "XLE")

STOCK_CODES60=("AMZN" "BBH" "FAS" "GLD" "GOOG" "IVV" "IWD" "IWM" "LABU" "MS" "QQQ" "SPY" "TQQQ" "VGT" "VUG" "XLE")

pip install pymysql
pip install nixtlats
pip install dbutils

# 执行 5 天预测
for STOCK_CODE in "${STOCK_CODES5[@]}"
do
    python ForcastStock.py --days 5 --algorithm $ALGORITHM --start $START --predict $PREDICT --end $END --stock_code $STOCK_CODE --db_name_forecast $DB_NAME_FORECAST
done


# 执行 20 天预测
for STOCK_CODE in "${STOCK_CODE20[@]}"
do
    python ForcastStock.py --days 20 --algorithm $ALGORITHM --start $START --predict $PREDICT --end $END --stock_code $STOCK_CODE --db_name_forecast $DB_NAME_FORECAST
done

# 执行 60 天预测
for STOCK_CODE in "${STOCK_CODES60[@]}"
do
    python ForcastStock.py --days 60 --algorithm $ALGORITHM --start $START --predict $PREDICT --end $END --stock_code $STOCK_CODE --db_name_forecast $DB_NAME_FORECAST
done


# 改变这个sh脚本的权限，让其能写和调用
# chmod +x ForecastShell.sh

# 直接去运行这个sh脚本
# ./ForecastShell.sh
