
from nixtlats import TimeGPT
import pandas as pd


# Read the first excel file
amzn_base = pd.read_excel("amzn_base.xlsx")

# Convert the second column to datetime format
amzn_base['Date'] = pd.to_datetime(amzn_base['Date'], format="%Y-%m-%d")

# Read the second excel file
amzn_predict = pd.read_excel("amzn_predict.xlsx")

# Convert the second column to datetime format
amzn_predict['Date'] = pd.to_datetime(amzn_predict['Date'], format="%Y-%m-%d")

# Print both dataframes
print(amzn_base)
print(amzn_predict)

amzn_baseX = amzn_base[['Date', 'Open', 'High', 'Low', 'Volume']]

# 创建TimeGPT对象，传入token参数，如果没有提供则默认使用环境变量中的TIMEGPT_TOKEN
timegpt = TimeGPT(token='0tMWwfvjBodRpWuyY3DPtG5mqyge4ZtWDD6zVPFsMENwERSEueBA0mui8Mxdr9M4hUBBRCOvpdYRP8z49m90mRq2rWeVU7WYbmorSS7dV0DP7cfKLJcOQfgcYmW5Tr7laxzPa2iLK9uqobDfh5TBxRk9tnTEAxVnOrBvzgSaD8xshMQktjGbALzma4g3Kx6XzDLSgePzeMBqxN9w4v7uZ3h7Ea2EtJARS2CNEFt5kKIEAhICwYqVD3lEcawaYGUa')

# 调用timegpt模块的validate_token()函数进行token验证
timegpt.validate_token()

timegpt.plot(amzn_base, time_col='Date', target_col='Close')

# 调用timegpt模块中的forecast函数，对pltr_long_df数据进行预测
# 参数df表示要进行预测的数据框，pltr_long_df为待预测的数据框
# 参数h表示预测的时间步数，这里设置为14，即预测未来14个时间步的值
# 参数freq表示数据的频率，这里设置为'B'，表示工作日频率
# 参数id_col表示数据框中表示序列ID的列名，这里设置为'series_id'
# 参数time_col表示数据框中表示时间的列名，这里设置为'date'
# 参数target_col表示数据框中表示目标变量的列名，这里设置为'value'
timegpt_forecast_df = timegpt.forecast(
    df=amzn_base, h=30, freq='B',
    time_col='Date', target_col='Close',
    level=[80, 90],
    # 微调可以让他跑多10次，拟合程度更高
    # finetune_steps=10,
    # 添加历史就是让他从历史数据开始滑动窗口方法把整个数据集的预测跑一遍
    # add_history=True
)




# 打印预测结果的前几行
print(timegpt_forecast_df)

timegpt.plot(timegpt_forecast_df, time_col='Date', target_col='TimeGPT')