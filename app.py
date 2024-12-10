from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_models.data_collection import fetch_data_from_db, fetch_product_info, fetch_trend_data
from flask_models.data_preprocessing import preprocess_data
from flask_models.recommendation_model import build_recommendation_model, get_recommendations
from flask_models.trend_model import train_trend_model, predict_trending_products

app = Flask(__name__)
CORS(app)  # Cho phép các domain truy cập API


@app.route('/api/analyze', methods=['GET'])
def analyze():
    """Endpoint phân tích dữ liệu."""
    try:
        # Lấy dữ liệu từ database
        data = fetch_data_from_db()
        if data is None or data.empty:
            return jsonify({"error": "Không thể truy cập dữ liệu hoặc dữ liệu trống."}), 500

        # Tiền xử lý dữ liệu
        processed_data = preprocess_data(data)
        return processed_data.to_json(orient='records')  # Trả kết quả dưới dạng JSON

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/recommend/<int:product_id>', methods=['GET'])
def recommend(product_id):
    """Endpoint gợi ý sản phẩm dựa trên product_id."""
    try:
        # Lấy dữ liệu từ database
        data = fetch_data_from_db()
        if data is None or data.empty:
            return jsonify({"error": "Không thể truy cập dữ liệu hoặc dữ liệu trống."}), 500

        # Tiền xử lý dữ liệu
        processed_data = preprocess_data(data)

        # Xây dựng mô hình gợi ý
        model, user_item_matrix = build_recommendation_model(processed_data)

        # Lấy gợi ý
        recommendations = get_recommendations(product_id, model, user_item_matrix)
        product_recommendations = []

        # Gắn thông tin chi tiết sản phẩm vào kết quả
        for recommendation in recommendations:
            product_info = fetch_product_info(recommendation['product_id'])
            if product_info:
                product_recommendations.append({
                    "product_id": str(recommendation['product_id']),
                    "link": f"index_chitiet.php?product_id={recommendation['product_id']}",
                    "image": f"../admin/uploads/{product_info['product_img']}",
                    "name": product_info['product_name'],
                    "price_old": str(product_info['product_price']),
                    "price_new": str(product_info['product_price_new']),
                })

        return jsonify({"product": product_recommendations})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/trending-products', methods=['GET'])
def get_trending_products():
    """Endpoint dự đoán sản phẩm xu hướng."""
    try:
        # Lấy dữ liệu từ database
        df = fetch_trend_data()
        if df is None or df.empty:
            return jsonify({"error": "Không có dữ liệu xu hướng."}), 500

        # Huấn luyện mô hình xu hướng
        model = train_trend_model(df)
        if model is None:
            return jsonify({"error": "Không thể huấn luyện mô hình xu hướng."}), 500

        # Dự đoán sản phẩm xu hướng
        trending_products = predict_trending_products(df, model)
        if trending_products is None or trending_products.empty:
            return jsonify({"error": "Không thể tính toán sản phẩm xu hướng."}), 500

        # Trả kết quả
        result = trending_products.to_dict(orient='records')
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
