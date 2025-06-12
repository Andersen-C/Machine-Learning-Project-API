from flask import Flask, request, jsonify
# from flask import render_template

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
    return "Success"


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        content = request.json
        query = content.get("query")
        if isinstance(query, list):
            if len(query) != 1:
                return jsonify({"error": "'query' must be a list with exactly one string"}), 400
            query = query[0]
        elif not isinstance(query, str):
            return jsonify({"error": "'query' must be a single string or a list with one string"}), 400

        search_by = content.get("search_by", "song")
        filter_by = content.get("filter_by", None)
        top_k = content.get("top_k", 5)
        recommend_type = content.get("recommend_type", "artist")

        result = recommend_songs_artists(
            query=query,
            data=data,
            combined_features=combined_features,
            number_of_recommendation=top_k,
            search_by=search_by,
            filter_by=filter_by,
            recommend_type=recommend_type
        )

        if isinstance(result, str):
            return jsonify({"error": result}), 404

        normalized = []
        seen = set()
        print(f"Result: {result}")
        for r in result:
            if recommend_type == "song":
                key = f"{r['track_name'].lower()}::{r['artists'].lower()}"
                if key not in seen:
                    normalized.append({
                        "track_id": r.get("track_id", ""),
                        "track_name": r.get("track_name", ""),
                        "artist": r.get("artists", ""),
                        "album": r.get("album_name", ""),
                        "genre": r.get("track_genre", ""),
                        "popularity": r.get("popularity", 0)
                    })
                    seen.add(key)

            elif recommend_type == "artist":
                artist_name = r.get("artists", "")
                if artist_name not in seen:
                    normalized.append({
                        "artist": artist_name,
                        "popularity": r.get("popularity", 0)
                    })
                    seen.add(artist_name)

            elif recommend_type == "album":
                album_name = r.get("album_name", "")
                artist_name = r.get("artists", "")
                key = f"{album_name}::{artist_name}"
                if key not in seen:
                    normalized.append({
                        "album": album_name,
                        "artist": artist_name,
                        "popularity": r.get("popularity", 0)
                    })
                    seen.add(key)

        normalized = sorted(normalized, key=lambda x: x.get("popularity", 0), reverse=True)
        return jsonify({"recommendations": normalized[:top_k]}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/recommend/batch", methods=["POST"])
def recommend_batch():
    try:
        content = request.json
        queries = content.get("queries", [])
        search_by = content.get("search_by", "song")
        filter_by = content.get("filter_by", None)
        top_k = content.get("number_of_recommendation", 5)
        recommend_type = content.get("recommend_type", "song")

        if not isinstance(queries, list) or not queries:
            return jsonify({"error": "queries must be a non-empty list"}), 400

        result = recommend_songs_artists_batch(
            queries=queries,
            data=data,
            combined_features=combined_features,
            number_of_recommendation=top_k,
            search_by=search_by,
            filter_by=filter_by,
            recommend_type=recommend_type
        )

        if isinstance(result, str):
            return jsonify({"error": result}), 404

        normalized = []
        seen = set()

        for r in result:
            if recommend_type == "song":
                key = f"{r['track_name'].lower()}::{r['artists'].lower()}"
                if key not in seen:
                    normalized.append({
                        "track_id": r.get("track_id", ""),
                        "track_name": r.get("track_name", ""),
                        "artist": r.get("artists", ""),
                        "album": r.get("album_name", ""),
                        "genre": r.get("track_genre", ""),
                        "popularity": r.get("popularity", 0)
                    })
                    seen.add(key)

            elif recommend_type == "artist":
                artist_name = r.get("artists", "")
                if artist_name not in seen:
                    normalized.append({
                        "artist": artist_name,
                        "popularity": r.get("popularity", 0)
                    })
                    seen.add(artist_name)

            elif recommend_type == "album":
                album_name = r.get("album_name", "")
                artist_name = r.get("artists", "")
                key = f"{album_name}::{artist_name}"
                if key not in seen:
                    normalized.append({
                        "album": album_name,
                        "artist": artist_name,
                        "popularity": r.get("popularity", 0)
                    })
                    seen.add(key)

        normalized = sorted(normalized, key=lambda x: x.get("popularity", 0), reverse=True)
        return jsonify({"recommendations": normalized[:top_k]}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


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
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host="0.0.0.0", port=port)