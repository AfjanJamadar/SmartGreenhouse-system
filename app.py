import os
from flask import Flask, render_template, request
import random

app = Flask(__name__)

# Ensure 'uploads' directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Function to load CSV files
def load_csv(filepath, has_header=True, crop_file=False):
    data = []
    try:
        with open(filepath, 'r') as file:
            for idx, line in enumerate(file):
                if has_header and idx == 0:
                    continue  # Skip header
                values = line.strip().split(',')

                if not crop_file:  # For sensor data
                    try:
                        data.append([float(values[1]), float(values[2]), float(values[3])])
                    except (ValueError, IndexError):
                        print(f"Invalid data in sensor file at line {idx + 1}: {line}")
                        continue  # Skip invalid rows

                else:  # For crop data
                    try:
                        data.append([float(values[0]), float(values[1]), float(values[2]), values[3]])
                    except (ValueError, IndexError):
                        print(f"Invalid data in crop file at line {idx + 1}: {line}")
                        continue  # Skip invalid rows
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None
    print(f"Successfully loaded {len(data)} rows from {filepath}")
    return data

# Split features and labels for crop data
def split_features_labels(data, label_index):
    X = [row[:label_index] for row in data]
    y = [row[label_index] for row in data]
    print(f"Split data into {len(X)} features and {len(y)} labels")
    return X, y

# Random Forest Classifier (Mock implementation)
class RandomForest:
    def __init__(self, n_trees, max_depth=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            subset_X, subset_y = self.bootstrap_sample(X, y)
            tree = self.build_tree(subset_X, subset_y, depth=0)
            self.trees.append(tree)

    def bootstrap_sample(self, X, y):
        n_samples = len(X)
        sampled_indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
        sampled_X = [X[i] for i in sampled_indices]
        sampled_y = [y[i] for i in sampled_indices]
        return sampled_X, sampled_y

    def build_tree(self, X, y, depth):
        if len(set(y)) == 1 or (self.max_depth and depth >= self.max_depth):
            return {'prediction': max(set(y), key=y.count)}  # Majority class prediction

        n_features = len(X[0])
        features_to_split = random.sample(range(n_features), k=random.randint(1, n_features))

        best_feature, best_value = self.find_best_split(X, y, features_to_split)

        if best_feature is None:
            return {'prediction': max(set(y), key=y.count)}

        left_X, left_y, right_X, right_y = self.split_dataset(X, y, best_feature, best_value)

        left_tree = self.build_tree(left_X, left_y, depth + 1)
        right_tree = self.build_tree(right_X, right_y, depth + 1)

        return {'feature': best_feature, 'value': best_value, 'left': left_tree, 'right': right_tree}

    def find_best_split(self, X, y, features):
        best_feature, best_value = None, None
        best_score = float('inf')

        for feature in features:
            unique_values = sorted(set(row[feature] for row in X))
            for value in unique_values:
                left_y = [y[i] for i, row in enumerate(X) if row[feature] <= value]
                right_y = [y[i] for i, row in enumerate(X) if row[feature] > value]
                score = self.gini_impurity(left_y, right_y)

                if score < best_score:
                    best_score = score
                    best_feature, best_value = feature, value

        return best_feature, best_value

    def gini_impurity(self, left_y, right_y):
        def gini(y):
            class_counts = [y.count(c) for c in set(y)]
            n_samples = len(y)
            return 1 - sum((count / n_samples) ** 2 for count in class_counts)

        n_left, n_right = len(left_y), len(right_y)
        total_samples = n_left + n_right
        gini_left, gini_right = gini(left_y), gini(right_y)

        weighted_avg_gini = (n_left / total_samples) * gini_left + (n_right / total_samples) * gini_right
        return weighted_avg_gini

    def split_dataset(self, X, y, feature, value):
        left_X, left_y, right_X, right_y = [], [], [], []
        for i, row in enumerate(X):
            if row[feature] <= value:
                left_X.append(row)
                left_y.append(y[i])
            else:
                right_X.append(row)
                right_y.append(y[i])
        return left_X, left_y, right_X, right_y

    def predict_one(self, row, tree):
        if 'prediction' in tree:
            return tree['prediction']

        feature = tree['feature']
        value = tree['value']

        if row[feature] <= value:
            return self.predict_one(row, tree['left'])
        else:
            return self.predict_one(row, tree['right'])

    def predict_proba(self, X):
        tree_predictions = []
        for row in X:
            predictions = [self.predict_one(row, tree) for tree in self.trees]
            tree_predictions.extend(predictions)

        proba = {}
        for pred in tree_predictions:
            proba[pred] = proba.get(pred, 0) + 1
        total = len(tree_predictions)
        for key in proba:
            proba[key] = proba[key] / total  # Convert counts to probabilities
        print(f"Prediction probabilities: {proba}")
        return proba

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        csv_file = request.files['csvfile']

        if csv_file:
            # Save the uploaded user sensor data CSV
            user_filepath = os.path.join('uploads', 'modified_sensor_data.csv')
            crop_filepath = os.path.join('uploads', 'crop_suggestion_data_final.csv')  # Update path if needed
            csv_file.save(user_filepath)

            user_data = load_csv(user_filepath)  # Load sensor data
            crop_data = load_csv(crop_filepath, crop_file=True)  # Load crop data

            # Check for valid data before proceeding
            if user_data is None or crop_data is None:
                print("Data loading failed. Check the file structure.")
                return "Error loading data. Please check the uploaded file format."

            # Split crop data into features and labels
            X, y = split_features_labels(crop_data, label_index=3)

            # Train the RandomForest model
            rf_model = RandomForest(n_trees=10, max_depth=10)
            rf_model.fit(X, y)

            # Get the latest environmental data
            latest_data = user_data[-1][:3]  # Last row's temperature, moisture, and humidity

            # Predict probabilities for each crop
            crop_proba = rf_model.predict_proba([latest_data])

            # Expanded crop mapping dictionary
            crop_mapping = {
    'strawberry': 'Strawberry',
    'rice': 'Rice',
    'corn': 'Corn',
    'wheat': 'Wheat',
    'potato': 'Potato',
    'tomato': 'Tomato',
    'onion': 'Onion',
    'barley': 'Barley',
    'mustard': 'Mustard',
    'garlic': 'Garlic',
    'millet': 'Millet',
    'pulses': 'Pulses',
    'chickpea': 'Chickpea',
    'fenugreek': 'Fenugreek',
    'jowar': 'Jowar (Sorghum)',
    'bajra': 'Bajra (Pearl Millet)',
    'groundnut': 'Groundnut (Peanut)',
    'ragi': 'Ragi (Finger Millet)',
    'tapioca': 'Tapioca',
    'lentils': 'Lentils',
    'arhar': 'Arhar (Pigeon Pea)',
    'kharif': 'Kharif Crops',
    'rubber': 'Rubber',
    'sesame': 'Sesame',
    'maize': 'Maize',
    'kidney beans': 'Kidney Beans',
    'pigeon beans': 'Pigeon Beans',
    'moth beans': 'Moth Beans',
    'mung beans': 'Mung Beans',
    'black beans': 'Black Beans',
    'pomegranate': 'Pomegranate',
    'banana': 'Banana',
    'mango': 'Mango',
    'grapes': 'Grapes',
    'watermelon': 'Watermelon',
    'muskmelon': 'Muskmelon',
    'apple': 'Apple',
    'orange': 'Orange',
    'papaya': 'Papaya',
    'coconut': 'Coconut',
    'cotton': 'Cotton',
    'jute': 'Jute',
    'asparagus': 'Asparagus'
}

            

            # Preparing results with probabilities
            result = []
            for crop_code, probability in crop_proba.items():
                crop_name = crop_mapping.get(crop_code, "Unknown Crop")
                result.append((crop_name, probability * 100))  # Convert to percentage

            # Sort by highest probability
            result.sort(key=lambda x: x[1], reverse=True)

            # Log results to console for debugging
            print(f"Results: {result}")

            return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)


      