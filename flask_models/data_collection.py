import pandas as pd
import mysql.connector

def connect_to_db():
    """Hàm tạo kết nối đến cơ sở dữ liệu."""
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='website_nsports_backup'
        )
        return connection
    except mysql.connector.Error as err:
        print(f"Lỗi kết nối MySQL: {err}")
        return None


def fetch_data_from_db():
    """Lấy dữ liệu user ratings từ database."""
    connection = connect_to_db()
    if not connection:
        return None

    try:
        query = "SELECT user_id, product_id, rating FROM user_ratings WHERE rating > 0;"
        df = pd.read_sql(query, connection)
        print("Dữ liệu từ DB:", df.head())
        return df
    except mysql.connector.Error as err:
        print(f"Lỗi truy vấn MySQL: {err}")
        return None
    finally:
        connection.close()

        
def fetch_product_info(product_id):
    """Hàm lấy thông tin sản phẩm từ cơ sở dữ liệu."""
    connection = connect_to_db()  # Sử dụng hàm kết nối
    if connection is None:
        return None
    
    cursor = connection.cursor(dictionary=True)
    
    try:
        product_id = int(product_id)
        query = "SELECT product_id, product_img, product_name, product_price, product_price_new FROM tbl_product WHERE product_id = %s"
        cursor.execute(query, (product_id,))  # Truyền tham số an toàn để tránh SQL injection
        result = cursor.fetchone()
    except mysql.connector.Error as err:
        print(f"Lỗi truy vấn MySQL: {err}")
        result = None
    finally:
        # Đảm bảo đóng cursor và kết nối dù có lỗi hay không
        cursor.close()
        connection.close()
    
    return result


def fetch_trend_data():
    """Truy vấn dữ liệu xu hướng từ cơ sở dữ liệu."""
    connection = connect_to_db()
    if connection is None:
        return None
    
    query = """
    SELECT product_id, month, year, 
           SUM(purchase_count) AS total_purchases, 
           SUM(view_count) AS total_views, 
           SUM(added_to_cart) AS total_cart_additions
    FROM product_activity
    GROUP BY product_id, month, year
    """
    
    try:
        df = pd.read_sql(query, connection)
    except mysql.connector.Error as err:
        print(f"Lỗi truy vấn MySQL: {err}")
        df = None
    finally:
        connection.close()
    
    return df