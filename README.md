### README - Stock data analysis and prediction system

#### System Overview

This system is a comprehensive stock data processing and analysis platform designed to provide full-process automation solutions from data crawling to trading decision-making. The system consists of four main modules: stock data download, data indicator update, stock price prediction, and trading strategy generation. It is also supplemented by related drawing and execution scripts to ensure that users can obtain real-time and accurate stock trading information and predictions.

#### Module function

##### 1. DownloaderStockData.py - Stock data download

- **Main functions**: Download stock data from Yahoo Finance interface and store it in the MySQL database.
- **Operating procedures**:
   1. Connect to the MySQL database.
   2. Check and create a stock data table. If the table already exists, read the latest and earliest data dates in the table.
   3. Update the missing stock data (from 2014-01-01 to the current system date) based on the date range of the existing data.
   4. Set up scheduled update tasks, including data update after the end of the trading day and full update during the first run.

##### 2. UpdateStrategy.py - Calculate indicator data

- **Main functions**: Based on stored stock data, calculate the technical indicators required for various trading strategies and update them to the strategy information database.
- **Operating procedures**:
   1. Connect to the MySQL database containing raw and policy data.
   2. Check and create the policy data table. If the table already exists, skip creation.
   3. Read the data date range that needs to be updated, and use the data 30 days in advance to calculate indicators such as MACD.
   4. Update relevant data in the policy database and set up scheduled update tasks.

##### 3. ForecastStock.py - Stock Price Forecast

- **Main functions**: Use historical data to predict future stock prices, using SARIMAX, LSTM and TimeGPT models for prediction.
- **Operating procedures**:
   1. Connect to the database and read basic stock price data.
   2. Analyze the data from 2016 to 2022 and predict the stock price in 2023.
   3. Update the forecast results on a daily basis and store the forecast data back to the database.

##### 4. UpdateTrade.py - Trading information update

- **Main functions**: Generate and update trading decisions and related data based on forecast results and strategy indicators.
- **Operating procedures**:
   1. Connect to the database and read prediction and strategy data.
   2. Update the stock trading data table based on the trading strategy and forecast results.
   3. Set up regular update tasks to ensure the real-time and accuracy of data.

#### Auxiliary script

- **BuilderPredict.py**: Create database tables and views required for prediction results.
- **PlotterCandlestick.py**: Draws K-line charts of stocks, which is the main visual display tool.
- **PlotterPredict.py**: Plots a comparison chart between the stock price prediction results and the actual stock price.
- **PlotterTrade.py**: Displays the relationship between the return rate of the trading strategy and the stock price.
- **Auxiliary Shell Scripts**: Includes `ForecastShell.sh` and `PlotterPredict_Shell.sh`, used to execute Python prediction and plotting scripts.

#### Instructions for use

Please ensure that Python and related libraries have been installed on the system, the MySQL server is running normally, and the database connection parameters have been configured according to the actual environment. For specific running commands and parameter settings, please refer to the instructions in each script.

#### Development and maintenance

This project is sponsored by Zhuolin Li

Development and maintenance, if you have any questions or suggestions, please contact zhuolin@gatech.edu.

#### License

This project adopts the MIT license. For detailed terms, please refer to the LICENSE file attached to the project.
