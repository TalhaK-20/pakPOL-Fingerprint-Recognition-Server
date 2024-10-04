from flask import Flask, request, jsonify
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)


def extract_minutiae(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Step 1: Preprocess the image (Binarization and Thinning)
    # Binarize the image using Otsu's thresholding
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply morphological thinning to get a skeleton of the fingerprint
    thin_img = cv2.ximgproc.thinning(binary_img)

    # Step 2: Minutiae detection (simplified for ridge endings and bifurcations)
    minutiae = []
    rows, cols = thin_img.shape

    # Define the 8-connected neighborhood for pixel traversal
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):

            if thin_img[i, j] == 255:
                # Get 8-neighbors
                neighbors = [
                    thin_img[i - 1, j - 1], thin_img[i - 1, j], thin_img[i - 1, j + 1],
                    thin_img[i, j - 1], thin_img[i, j + 1],
                    thin_img[i + 1, j - 1], thin_img[i + 1, j], thin_img[i + 1, j + 1]
                ]

                # Count non-zero neighbors (ridge endings have 1 neighbor, bifurcations have 3)
                non_zero_neighbors = np.count_nonzero(neighbors)

                if non_zero_neighbors == 1:
                    minutiae.append({'type': 'ending', 'position': (i, j)})

                elif non_zero_neighbors == 3:
                    minutiae.append({'type': 'bifurcation', 'position': (i, j)})

    return minutiae


@app.route('/extract-fingerprint-features', methods=['POST'])
def extract_fingerprint_features_endpoint():
    data = request.json
    image_path = data['image_path']

    try:
        # Extract minutiae points from the fingerprint
        minutiae = extract_minutiae(image_path)
        return jsonify({'minutiae': minutiae}), 200

    except Exception as e:
        print(f"Error extracting minutiae points: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/compare-fingerprints', methods=['POST'])
def compare_fingerprints():
    data = request.json
    query_minutiae = data['query_minutiae']
    database_minutiae = data['database_minutiae']

    try:
        valid_database_minutiae = []

        # Convert the minutiae to a usable format
        query_positions = np.array([m['position'] for m in query_minutiae], dtype=np.float32)
        for i, minutiae_list in enumerate(database_minutiae):
            positions = np.array([m['position'] for m in minutiae_list], dtype=np.float32)

            if positions.shape[0] != query_positions.shape[0]:
                print(f"Skipping database entry at index {i} due to mismatch in minutiae count")
                continue

            valid_database_minutiae.append(positions)

        if len(valid_database_minutiae) == 0:
            raise ValueError("No valid minutiae data found in the database.")

        valid_database_minutiae = np.array(valid_database_minutiae, dtype=np.float32)

        # Use nearest neighbors to find the closest match based on minutiae positions
        neighbors = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(valid_database_minutiae)
        distance, index = neighbors.kneighbors([query_positions])

        similarity_threshold = 10.0  # You can adjust this threshold as per your accuracy requirements

        if distance[0][0] < similarity_threshold:
            return jsonify({'match': True, 'distance': distance[0][0]}), 200
        else:
            return jsonify({'match': False, 'distance': distance[0][0]}), 200

    except ValueError as ve:
        print(f"Feature mismatch: {ve}")
        return jsonify({'error': str(ve)}), 500

    except Exception as e:
        print(f"Error comparing fingerprints: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5002)