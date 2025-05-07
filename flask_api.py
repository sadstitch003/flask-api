from flask import Flask, request, jsonify
import pandas as pd
import os
import joblib
from sklearn.neighbors import NearestNeighbors
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__)

# Configuration
MODEL_FOLDER = 'models'
DATA_FOLDER = 'data'
KNN_MODEL_PATH = os.path.join(MODEL_FOLDER, 'knn_model.joblib')
CSV_DATA_PATH = os.path.join(DATA_FOLDER, 'transaction-engineered.csv')

# Helper function to create pivot table
def create_pivot_table(df):
    basket = df.pivot_table(
        index='Transaction_ID',
        columns='Deskripsi Barang',
        values='Jml',
        aggfunc='sum',
        fill_value=0
    )
    basket = (basket >= 1).astype(int)
    return basket

# Load id_barang mapping
id_barang = pd.read_csv(os.path.join(DATA_FOLDER, 'IdBarang.csv'))  # Columns: ['id', 'product']

def get_product_by_id(product_id):
    try:
        return id_barang.iloc[product_id, 1]
    except (IndexError, ValueError):
        return None

def load_data():
    return pd.read_csv(CSV_DATA_PATH)

def train_apriori():
    df = load_data()
    basket = create_pivot_table(df)
    basket_binary = basket > 0
    frequent_itemsets = apriori(basket_binary, min_support=0.001, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)
    return rules

# Load model and data once
df = load_data()
basket = create_pivot_table(df)
model_knn = joblib.load(KNN_MODEL_PATH)
df_apriori = train_apriori()

@app.route('/api/recommend', methods=['GET'])
def recommend():
    try:
        product_id = request.args.get('id')
        if product_id is None:
            return jsonify({'error': 'Missing product ID'}), 400

        try:
            product_id = int(product_id)
        except ValueError:
            return jsonify({'error': 'Product ID must be an integer'}), 400

        number_recommendation = int(request.args.get('number_recommendation', 5))

        product = get_product_by_id(product_id)
        if not product or product not in basket.columns:
            return jsonify({'error': f'Product ID {product_id} not found'}), 404

        # --- KNN Recommendation ---
        product_index = list(basket.columns).index(product)
        distances, indices = model_knn.kneighbors(
            [basket.T.values[product_index]],
            n_neighbors=min(number_recommendation + 1, len(basket.columns))
        )

        knn_recommendations = {
            basket.columns[indices[0][i]]: 1 - distances[0][i]
            for i in range(1, len(distances[0]))
        }

        # --- Apriori Recommendation ---
        antecedents = [list(x)[0] for x in df_apriori['antecedents'] if len(x) == 1]
        indices = [i for i, a in enumerate(antecedents) if a == product]

        apriori_recommendations = {}
        for i in indices:
            conseq = list(df_apriori.iloc[i]['consequents'])
            if len(conseq) == 1:
                apriori_recommendations[conseq[0]] = df_apriori.iloc[i]['lift']

        # --- Merge Recommendations ---
        combined = []
        for prod in set(knn_recommendations.keys()).union(apriori_recommendations.keys()):
            combined.append({
                'product': prod,
                'similarity_score': knn_recommendations.get(prod),
                'lift_score': apriori_recommendations.get(prod),
                'priority': (
                    1 if prod in knn_recommendations and prod in apriori_recommendations else
                    2 if prod in knn_recommendations else 3
                )
            })

        combined_sorted = sorted(combined, key=lambda x: (x['priority'], -(x['similarity_score'] or 0), -(x['lift_score'] or 0)))
        top_recommendations = combined_sorted[:number_recommendation]

        return jsonify({
            'product': product,
            'recommendations': top_recommendations
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500