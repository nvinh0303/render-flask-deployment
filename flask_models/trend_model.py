from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

def train_trend_model(df):
    """Huấn luyện mô hình hồi quy dự đoán xu hướng."""
    if df is None or df.empty:
        print("DataFrame không có dữ liệu.")
        return None
    
    # Chuẩn bị dữ liệu
    X = df[['total_views', 'total_cart_additions']].values
    y = df['total_purchases'].values

    # Huấn luyện mô hình
    model = LinearRegression()
    model.fit(X, y)
    
    print("Mô hình hồi quy đã được huấn luyện.")
    return model

def predict_trending_products(df, model):
    """Dự đoán sản phẩm nổi bật dựa trên mô hình hồi quy."""
    if df is None or df.empty:
        print("DataFrame không có dữ liệu.")
        return None

    # Chuẩn bị dữ liệu đầu vào
    X = df[['total_views', 'total_cart_additions']].values

    # Dự đoán số lượng mua hàng
    df['predicted_purchases'] = model.predict(X)

    # Sắp xếp theo giá trị dự đoán và trả về top 10 sản phẩm
    trending_products = df.sort_values(by='predicted_purchases', ascending=False).head(10)
    
    return trending_products
