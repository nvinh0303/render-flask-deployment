import pandas as pd
from sklearn.neighbors import NearestNeighbors
from flask_models.data_collection import fetch_data_from_db, fetch_product_info

def build_recommendation_model(df):
    """Tạo mô hình KNN từ dữ liệu."""
    user_item_matrix = df.pivot(index='user_id', columns='product_id', values='rating').fillna(0)
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(user_item_matrix.values)
    return model_knn, user_item_matrix


def get_recommendations(product_id, model, user_item_matrix, n_recommendations=5):
    """
    Hàm lấy gợi ý sản phẩm sử dụng mô hình NearestNeighbors.
    :param product_id: ID của sản phẩm cần tìm gợi ý
    :param model: Mô hình NearestNeighbors
    :param user_item_matrix: Ma trận người dùng - sản phẩm
    :param n_recommendations: Số lượng gợi ý sản phẩm
    :return: Danh sách các gợi ý sản phẩm
    """
    # Lấy vector đặc trưng của sản phẩm cần tìm
    product_vector = user_item_matrix.loc[product_id].values.reshape(1, -1)
    
    # Tìm các sản phẩm gần nhất
    distances, indices = model.kneighbors(product_vector, n_neighbors=n_recommendations + 1)  # +1 vì bao gồm sản phẩm hiện tại
    
    recommendations = []
    
    # Lọc các sản phẩm gần nhất (bỏ qua sản phẩm chính)
    for idx in indices[0][1:]:
        # Tạo đối tượng sản phẩm từ chỉ mục
        recommended_product = {
            "product_id": user_item_matrix.index[idx],
            "similarity_score": 1 / (1 + distances[0][idx-1])  # Độ tương đồng (nghịch đảo của khoảng cách)
        }
        recommendations.append(recommended_product)
    
    return recommendations


if __name__ == "__main__":
    df = fetch_data_from_db()
    if df is not None:
        model, user_item_matrix = build_recommendation_model(df)
        product_id = 10  # ID sản phẩm cần gợi ý
        recommendations = get_recommendations(product_id, model, user_item_matrix)
        print("Gợi ý sản phẩm:", recommendations)
    else:
        print("Không thể lấy dữ liệu từ cơ sở dữ liệu.")
