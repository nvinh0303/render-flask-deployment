def preprocess_data(df):
    """
    Tiền xử lý dữ liệu: Loại bỏ giá trị thiếu và chuyển đổi kiểu dữ liệu
    """
    # Loại bỏ các giá trị thiếu
    df.dropna(inplace=True)

    # Chuyển đổi kiểu dữ liệu
    df['user_id'] = df['user_id'].astype(int)
    df['product_id'] = df['product_id'].astype(int)
    df['rating'] = df['rating'].astype(float)

    return df
