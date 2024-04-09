import pymysql
import numpy as np

def connect_db():
    """ Connects to the forecast database. """
    return pymysql.connect(host="10.5.0.18", user="lizhuolin", password="123456", database="forecast_sarimax")

def calculate_values(cursor, stock_code, predict_day):
    """ Calculates and updates the predictive stats in the database for given stock and predict_day. """
    # Updated SQL query to select the required columns directly
    query = f"""
        SELECT Date, Close, PredictMin, PredictMax {', '.join([f'Day{i}' for i in range(1, predict_day + 1)])}
        FROM {stock_code}_predict_{predict_day}d
    """
    cursor.execute(query)
    rows = cursor.fetchall()

    for row in rows:
        date = row[0]
        close_price = row[1]
        predict_min = row[2]
        predict_max = row[3]
        prices = np.array(row[4:], dtype=float)  # Adjusting the index to skip the first three columns

        # Ensure no NaNs in prices for calculations
        if np.isnan(prices).any() or close_price == 0:
            continue

        # Calculate predictive stats based on the fetched data
        predict_profit = 100 * (predict_max - close_price) / close_price if close_price != 0 else 0
        predict_loss = 100 * (predict_min - close_price) / close_price if close_price != 0 else 0
        days = np.arange(1, len(prices) + 1)
        slope, intercept = np.polyfit(days, prices, 1)

        # Update the table with calculated values
        update_query = f"""
            UPDATE {stock_code}_predict_{predict_day}d
            SET
                PredictProfit = %s,
                PredictLoss = %s,
                PredictSlope = %s,
                PredictIntercept = %s
            WHERE Date = %s
        """
        cursor.execute(update_query, (predict_profit, predict_loss, slope, intercept, date))



def update_predictive_stats():
    db_connection = connect_db()
    cursor = db_connection.cursor()

    stock_codes = ['AMZN', 'BBH', 'FAS', 'GLD', 'GOOG', 'IVV', 'IWD', 'IWM', 'LABU', 'MS', 'QQQ', 'SPY', 'TQQQ', 'VGT', 'VUG', 'XLE']
    predict_days = [5, 20, 60]

    try:
        for stock_code in stock_codes:
            for days in predict_days:
                calculate_values(cursor, stock_code, days)
                db_connection.commit()
                print(f"Updated predictive stats for {stock_code} for {days} days forecast.")

    except Exception as e:
        print(f"Failed to update predictive stats: {e}")
        db_connection.rollback()
    finally:
        cursor.close()
        db_connection.close()

if __name__ == "__main__":
    update_predictive_stats()