# import pandas as pd
# from prophet import Prophet

# def build_sales_forecasting_model(df, product_id):
#     # Lọc dữ liệu cho sản phẩm cụ thể
#     product_data = df[df['product_id'] == product_id][['date', 'sales']]
    
#     # Đổi tên các cột theo định dạng Prophet yêu cầu: 'ds' và 'y'
#     product_data = product_data.rename(columns={'date': 'ds', 'sales': 'y'})
    
#     # Khởi tạo và huấn luyện mô hình Prophet
#     model = Prophet()
#     model.fit(product_data)
    
#     return model

# def predict_future_sales(model, periods=30):
#     # Tạo dataframe chứa ngày dự báo trong tương lai
#     future = model.make_future_dataframe(periods=periods)
    
#     # Thực hiện dự báo
#     forecast = model.predict(future)
    
#     return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# if __name__ == "__main__":
#     # Giả sử bạn có dữ liệu bán hàng cho từng sản phẩm
#     df = pd.read_csv('path_to_sales_data.csv')  # Thay đổi đường dẫn cho phù hợp
    
#     product_id = 102  # ID của sản phẩm cần dự báo
#     model = build_sales_forecasting_model(df, product_id)
    
#     # Dự báo 30 ngày tới
#     forecast = predict_future_sales(model, periods=30)
#     print("Dự báo xu hướng bán hàng cho sản phẩm", product_id, ":\n", forecast)
