#!/bin/bash

# 定义共享参数
ALGORITHM="LSTM"
START="2016-01-01"
PREDICT="2019-01-01"
END="2023-12-31"

STOCK_CODES=("AMZN")

# STOCK_CODES=("AMZN" "BBH" "FAS" "GLD" "GOOG" "IVV" "IWD" "IWM" "LABU" "MS" "QQQ" "SPY" "TQQQ" "VGT" "VUG" "XLE")


# 绘图的预测
for STOCK_CODE in "${STOCK_CODES[@]}"
do
    python PredicterPlot.py --algorithm $ALGORITHM --start $START --predict $PREDICT --end $END --stock_code $STOCK_CODE
done


# 改变这个sh脚本的权限，让其能写和调用
# chmod +x run_sp.sh

# 直接去运行这个sh脚本
# ./run_sp.sh
