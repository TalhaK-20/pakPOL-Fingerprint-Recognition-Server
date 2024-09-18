from flask import Flask, request, jsonify
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from skimage.feature import hog

app = Flask(__name__)


def extract_fingerprint_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    img = cv2.GaussianBlur(img, (5, 5), 0)

    features, _ = hog(img, block_norm='L2-Hys', pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)

    return features


@app.route('/extract-fingerprint-features', methods=['POST'])
def extract_fingerprint_features_endpoint():
    data = request.json
    image_path = data['image_path']

    try:
        features = extract_fingerprint_features(image_path)
        return jsonify({'features': features.tolist()}), 200
    except Exception as e:
        print(f"Error extracting fingerprint features: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/compare-fingerprints', methods=['POST'])
def compare_fingerprints():
    data = request.json
    query_features = np.array(data['query_features'], dtype=np.float32)
    database_features = data['database_features']

    try:
        valid_database_features = []

        for i, features in enumerate(database_features):
            features = np.array(features, dtype=np.float32)

            if features.shape[0] != query_features.shape[0]:
                print(f"Skipping database feature at index {i} due to shape mismatch")
                continue

            valid_database_features.append(features)

        if len(valid_database_features) == 0:
            raise ValueError("No valid fingerprint features found in the database.")

        valid_database_features = np.array(valid_database_features, dtype=np.float32)

        neighbors = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(valid_database_features)
        distance, index = neighbors.kneighbors([query_features])

        similarity_threshold = 1.0

        if distance[0][0] < similarity_threshold:
            return jsonify({'match': True, 'distance': distance[0][0]}), 200
        else:
            return jsonify({'match': False, 'distance': distance[0][0]}), 200

    except ValueError as ve:
        print(f"Feature length mismatch: {ve}")
        return jsonify({'error': str(ve)}), 500
    except Exception as e:
        print(f"Error comparing fingerprints: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5002)
