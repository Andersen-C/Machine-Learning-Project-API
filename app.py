from flask import Flask, request, jsonify
import joblib
from recommender import recommend_songs_artists, recommend_songs_artists_batch
from flask_cors import CORS
from flask_caching import Cache
import pandas as pd
import os

app = Flask(__name__)
CORS(app)
cache = Cache(config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 300})
cache.init_app(app)

# Load data
with open("model/recommender_systems.pkl", "rb") as f:
    model_data = joblib.load(f)

data = pd.read_csv('model/dataset.csv')
data = data.drop_duplicates(subset=['track_name'])
data = data.dropna()
data = data.drop(columns=['Unnamed: 0'], errors='ignore')
data = data.reset_index(drop=True)
data['artists'] = data['artists'].str.replace(';', ", ")

combined_features = model_data['combined_features']

@app.route("/", methods=['GET'])
def home():
    return "The Recommender API is running"

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        content = request.json
        queries = content.get("query", [])
        if isinstance(queries, list):
            if len(queries) != 1:
                return jsonify({"error": "'query' must be a list with exactly one string"}), 400
            query = queries[0]
        elif isinstance(queries, str):
            query = queries
        else:
            return jsonify({"error": "'query' must be a single string or a list with one string"}), 400

        search_by = content.get("search_by", "song")
        filter_by = content.get("filter_by", None)
        top_k = content.get("top_k", 5)
        recommend_type = content.get("recommend_type", "song")

        result = recommend_songs_artists(
            query=query,
            data=data,
            combined_features=combined_features,
            number_of_recommendation=top_k,
            search_by=search_by,
            filter_by=filter_by,
            recommend_type=recommend_type
        )

        normalized = []
        seen = set()
        for r in result:
            key = f"{r['track_name'].lower()}::{r['artists'].lower()}"
            if key not in seen:
                normalized.append({
                    "track_id": r["track_id"],
                    "track_name": r["track_name"],
                    "artist": r["artists"],
                    "album": r["album_name"],
                    "genre": r["track_genre"],
                    "popularity": r.get("popularity", 0)
                })
                seen.add(key)

        normalized = sorted(normalized, key=lambda x: x["popularity"], reverse=True)
        return jsonify({"recommendations": normalized[:top_k]}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/recommend/batch", methods=["POST"])
def recommend_batch():
    content = request.json
    queries = content.get("queries", [])  
    
    if not isinstance(queries, list) or len(queries) < 2:
            return jsonify({"error": "'queries' must be a list with at least two strings for batch recommendation"}), 400
    
    search_by = content.get("search_by", "song")
    filter_by = content.get("filter_by", None)
    top_k = content.get("number_of_recommendation", 5)
    recommend_type = content.get("recommend_type", "song")

    if not isinstance(queries, list) or not queries:
        return jsonify({"error": "queries must be a non-empty list"}), 400

    results = recommend_songs_artists_batch(
        queries=queries,
        data=data,
        combined_features=combined_features,
        number_of_recommendation=top_k,
        search_by=search_by,
        filter_by=filter_by,
        recommend_type=recommend_type
    )

    if isinstance(results, str):
        return jsonify({"message": results}), 200

    return jsonify({"recommendations": results.to_dict(orient="records")}), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "API is healthy"}), 200

@app.route("/info", methods=["GET"])
def info():
    return jsonify({
        "model_version": "v1.0",
        "dataset_size": len(data),
        "last_updated": "2025-06-06",
        "available_filters": ["artist", "album", "genre"],
        "available_search_by": ["song", "artist", "album"]
    }), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)