import pymysql
import pandas as pd
from flask import request, Flask, jsonify

# 设置Flask程序接口
app = Flask(__name__)

def dataframe_to_json(df):
    """
    Convert a pandas DataFrame to JSON format, with the 'Month' column formatted as 'yyyy-mm-dd'.

    Parameters:
    df (pd.DataFrame): The DataFrame to convert, expected to have a 'Month' column in datetime format.

    Returns:
    str: JSON string representing the DataFrame.
    """
    try:
        # Ensure 'Month' column is in 'yyyy-mm-dd' format if it exists and is in datetime type
        if 'Month' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Month']):
            df['Month'] = df['Month'].dt.strftime('%Y-%m-%d')

        # Convert the DataFrame to a JSON string
        json_result = df.to_json(orient='records', date_format='iso')
        return json_result
    except Exception as e:
        print(f"Error converting DataFrame to JSON: {e}")
        return None

@app.route('/price')
def query_price():
    # 获取查询字符串中的省份参数，如果没有提供，则默认为 'Cherkaska'
    province = request.args.get('province', 'Cherkaska')
    Items = ['Beetroots', 'Bread', 'Butter', 'Chicken', 'Fuel', 'Potatoes']
    item_columns = ', '.join([f"MAX(CASE WHEN Item = '{item}' THEN Price END) AS `{item}`" for item in Items])
    try:
        connection = pymysql.connect(**db_config)  # 创建连接
        cursor = connection.cursor()
        # 构造一个SQL查询，聚合每个月的不同消费品价格
        query = f"""
        SELECT DATE_FORMAT(Date, '%Y-%m') AS `Month`,
        {item_columns}
        FROM food_view
        WHERE Province = '{province}'
        GROUP BY DATE_FORMAT(Date, '%Y-%m')
        ORDER BY DATE_FORMAT(Date, '%Y-%m')
        """
        cursor.execute(query)
        result = cursor.fetchall()  # 获取所有结果行
        cursor.close()
        connection.close()
        columns = ['Month'] + Items
        price = pd.DataFrame(result, columns=columns) if result else pd.DataFrame(columns=columns)
        dataframe_to_json(price)

    except Exception as e:
        print(f"Error fetching data for {province}: {e}")
        return jsonify({'error': str(e)})


"""
函数2-难民查询函数
输入 省名称
输出 按照月份变化的难民数量序列
"""


@app.route('/refugee')
def query_refugee():
    # 获取查询字符串中的省份参数，如果没有提供，则默认为 'Cherkaska'
    province = request.args.get('province', 'Cherkaska')
    try:
        connection = pymysql.connect(**db_config)  # 创建连接
        cursor = connection.cursor()
        # 确保省份名被正确地包含在单引号中
        query = f"SELECT Date AS Month, Refugee FROM people_view WHERE Province = '{province}'"
        cursor.execute(query)
        result = cursor.fetchall()
        # 如果有结果则创建DataFrame，否则创建空的DataFrame
        cursor.close()
        connection.close()
        refugee = pd.DataFrame(result, columns=['Month', 'Refugee'])
        return dataframe_to_json(refugee)
    except Exception as e:
        print(f"Error fetching {province}: {e}")
        return jsonify({'error': str(e)})


"""
函数3-战争数量和死亡人数查询函数
输入 省名称
输出 按照月份变化的战争数量和死亡人数序列
"""


@app.route('/war')
def query_war():
    # 获取查询字符串中的省份参数，如果没有提供，则默认为 'Cherkaska'
    province = request.args.get('province', 'Cherkaska')
    try:
        connection = pymysql.connect(**db_config)  # 创建连接
        cursor = connection.cursor()
        query = f"""
        SELECT
            Date AS Month,
            SUM(fatalities) AS Fatalities,
            COUNT(*) AS Conflicts
        FROM
            war_view
        WHERE
            Province = '{province}'
        GROUP BY
            Date
        ORDER BY
            Month;
        """
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        connection.close()
        war = pd.DataFrame(result, columns=['Month', 'Fatalities', 'Conflicts'])
        return dataframe_to_json(war)
    except Exception as e:
        print(f"Error fetching {province} {e}")
        return jsonify({'error': str(e)})


"""
函数4 - 冲突类型查询函数
输入 省名称
输出 该省全部的冲突及其数量
"""


@app.route('/conflict')
def query_conflict():
    # 获取查询字符串中的省份参数，如果没有提供，则默认为 'Cherkaska'
    province = request.args.get('province', 'Cherkaska')
    try:
        connection = pymysql.connect(**db_config)  # 创建连接
        cursor = connection.cursor()
        query = f"""
        SELECT DISTINCT(EVENT),
               COUNT(EVENT) AS Times
        FROM 
               war_view
        WHERE 
               Province = '{province}'
        GROUP BY 
               EVENT
        ORDER BY 
               EVENT;
        """
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        connection.close()
        conflict = pd.DataFrame(result, columns=['Event', 'Times'])
        return dataframe_to_json(conflict)
    except Exception as e:
        print(f"Error fetching {province} {e}")
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    # 乌克兰标准的24个州
    Provinces = ['Cherkaska', 'Chernihivska', 'Chernivetska', 'Dnipropetrovska', 'Donetska',
                 'Ivano-Frankivska', 'Kharkivska', 'Khersonska', 'Khmelnytska', 'Kirovohradska', 'Kyivska',
                 'Luhanska', 'Lvivska', 'Mykolaivska', 'Odeska', 'Poltavska', 'Rivnenska', 'Sumska',
                 'Ternopilska', 'Vinnytska', 'Volynska', 'Zakarpatska', 'Zaporizka', 'Zhytomyrska']

    # 保留的六种重要消费品，红菜头，面包，黄油，鸡肉，汽油，土豆
    Items = ['Beetroots', 'Bread', 'Butter', 'Chicken', 'Fuel', 'Potatoes']

    # 设置数据库的参数 连接数据库的信息（连接待生成策略数据的数据库）
    db_config = {"host": "8.147.99.223", "port": 3306, "user": "lizhuolin",
                 "password": "&a3sFD*432dfD!o0#3^dP2r2d!sc@", "database": "cse6242"}

    app.run(debug=True, host='0.0.0.0', port=5001)
