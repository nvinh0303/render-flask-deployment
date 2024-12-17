import pandas as pd
from sklearn.neighbors import NearestNeighbors
from flask_models.data_collection import fetch_data_from_db, fetch_product_info
import numpy as np

def build_recommendation_model(df):
    # Xóa các dòng trùng lặp trong dữ liệu (nếu có) và giữ lại dòng cuối cùng
    df = df.drop_duplicates(subset=['user_id', 'product_id'], keep='last')

    """Tạo mô hình KNN từ dữ liệu.
    :param df: DataFrame chứa dữ liệu người dùng - sản phẩm
    :return: Mô hình KNN và ma trận người dùng - sản phẩm
    """
    user_item_matrix = df.pivot(index='user_id', columns='product_id', values='rating').fillna(0)

    # print("Ma trận người dùng - sản phẩm:\n", user_item_matrix)

    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(user_item_matrix.values)
    return model_knn, user_item_matrix


def get_recommendations(product_id, model, user_item_matrix, n_recommendations=9):
    """
    Hàm lấy gợi ý sản phẩm sử dụng mô hình NearestNeighbors.
    :param product_id: ID của sản phẩm cần tìm gợi ý
    :param model: Mô hình NearestNeighbors
    :param user_item_matrix: Ma trận người dùng - sản phẩm
    :param n_recommendations: Số lượng gợi ý sản phẩm
    :return: Danh sách các gợi ý sản phẩm
    """

    """
    Ý tưởng:
    - Đầu tiên, kiểm tra xem sản phẩm có tồn tại trong ma trận không.
    - Sau đó, xác định vector đặc trưng của sản phẩm cần gợi ý.
    - Kiểm tra xem vector đặc trưng có rỗng không.
    - Đảm bảo rằng số lượng đặc trưng của sản phẩm phù hợp với ma trận người dùng - sản phẩm.
    - Tìm các sản phẩm gợi ý.
    - Đảm bảo rằng số lượng sản phẩm gợi ý đủ lớn.
    - Nếu số lượng sản phẩm gợi ý ít hơn số lượng yêu cầu, giảm số lượng yêu cầu.
    - Lặp qua các sản phẩm gợi ý và lấy thông tin sản phẩm.
    - Nếu không tìm thấy gợi ý nào, in ra thông báo.
    """

    # Kiểm tra xem sản phẩm có tồn tại trong ma trận không
    if product_id not in user_item_matrix.columns:
        print(f"Product {product_id} not found in matrix!")
        return []

    # Xác định vector đặc trưng của sản phẩm cần gợi ý
    product_vector = user_item_matrix[product_id].values.reshape(1, -1)

    # Kiểm tra xem vector đặc trưng có rỗng không
    if not product_vector.any():
        print(f"Product {product_id} has no ratings or the vector is empty!")
        return []

    print(f"Product vector for {product_id}: {product_vector}")

    # Đảm bảo rằng số lượng đặc trưng của sản phẩm phù hợp với ma trận người dùng - sản phẩm
    if product_vector.shape[1] != user_item_matrix.shape[1]:
        print(f"Feature mismatch: Product vector has {product_vector.shape[1]} features, but model expects {user_item_matrix.shape[1]} features.")
        
        # Nếu số lượng đặc trưng của sản phẩm ít hơn, thêm các giá trị 0 vào vector
        missing_features_count = user_item_matrix.shape[1] - product_vector.shape[1]
        if missing_features_count > 0:

            # Tạo vector chứa các giá trị 0
            missing_features = [0] * missing_features_count

            # Thêm các giá trị 0 vào vector
            product_vector = np.append(product_vector, [missing_features], axis=1)

        print(f"Updated product vector: {product_vector}")

    # Tìm các sản phẩm gợi ý
    distances, indices = model.kneighbors(product_vector, n_neighbors=n_recommendations + 1)

    # Đảm bảo rằng số lượng sản phẩm gợi ý đủ lớn
    num_neighbors = len(indices[0]) - 1  # Đây bao gồm cả sản phẩm đầu tiên (chính nó)
    print(f"Found {num_neighbors} neighbors for product {product_id}")

    # Nếu số lượng sản phẩm gợi ý ít hơn số lượng yêu cầu, giảm số lượng yêu cầu
    if num_neighbors < n_recommendations:
        print(f"Warning: Only {num_neighbors} neighbors available, adjusting to match.")
        n_recommendations = num_neighbors # Đặt số lượng gợi ý bằng số lượng sản phẩm gợi ý thực sự

    recommendations = [] 
    
    # Lặp qua các sản phẩm gợi ý và lấy thông tin sản phẩm
    for idx in indices[0][1:n_recommendations + 1]:
        recommended_product_id = user_item_matrix.columns[idx]
        similarity_score = 1 / (1 + distances[0][idx - 1])  # Điểm tương đồng (similarity score) được tính dựa trên khoảng cách cosine 
        
        recommendations.append({
            "product_id": recommended_product_id,
            "similarity_score": similarity_score
        })
    
    # Nếu không tìm thấy gợi ý nào
    if not recommendations:
        print(f"No recommendations found for product {product_id}")

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
