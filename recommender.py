from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

def recommend_songs_artists(
    query, data, combined_features, 
    number_of_recommendation=5, 
    search_by="song", filter_by=None, 
    recommend_type="song"
):
    if isinstance(query, str):
        query = [query.strip().lower()]
    else:
        query = [q.strip().lower() for q in query]

    data = data.reset_index(drop=True)

    matched_indices = []

    for q in query:
        if search_by == 'song':
            match = data[data['track_name'].str.lower().str.strip() == q]
        elif search_by == 'artist':
            match = data[data['artists'].str.lower().str.contains(q)]
        elif search_by == 'album':
            match = data[data['album_name'].str.lower().str.strip() == q]
        else:
            return "Invalid search type. Choose 'song', 'artist', or 'album'."

        if not match.empty:
            matched_indices.extend(match.index.tolist())

    if not matched_indices:
        return f"No matches found for query: {query}"

    matched_indices = list(set(matched_indices))

    query_vector = np.mean([combined_features[i] for i in matched_indices], axis=0)

    filtered_data = data.copy()
    filtered_features = combined_features.copy()

    reference_artist = data.loc[matched_indices[0], 'artists'].strip().lower()
    reference_genre = data.loc[matched_indices[0], 'track_genre']
    reference_album = data.loc[matched_indices[0], 'album_name']

    if isinstance(filter_by, list):
        if 'artist' in filter_by:
            mask = filtered_data['artists'].str.lower().str.contains(reference_artist)
            filtered_data = filtered_data[mask]
            filtered_features = filtered_features[mask.values]

        if 'genre' in filter_by:
            mask = filtered_data['track_genre'] == reference_genre
            filtered_data = filtered_data[mask]
            filtered_features = filtered_features[mask.values]

        if 'album' in filter_by:
            mask = filtered_data['album_name'] == reference_album
            filtered_data = filtered_data[mask]
            filtered_features = filtered_features[mask.values]


    if len(filtered_data) <= 1:
        return f"Not enough data with the filter '{filter_by}' to make recommendations."

    # KNN for recommendations
    knn = NearestNeighbors(n_neighbors=min(len(filtered_data), number_of_recommendation + len(matched_indices)), metric='cosine')
    knn.fit(filtered_features)

    distances, indices = knn.kneighbors([query_vector])
    recommended_indices = indices.flatten()

    original_local_indices = [filtered_data.index[filtered_data.index == idx][0] for idx in matched_indices if idx in filtered_data.index]
    recommended_indices = [i for i in recommended_indices if i not in original_local_indices]

    recommendations = filtered_data.iloc[recommended_indices].copy()

    if recommendations.empty:
        return f"No recommendations found with the filter '{filter_by}'. Try removing or changing the filter."

    if 'track_name' in recommendations.columns:
        recommendations['track_name_lower'] = recommendations['track_name'].str.lower().str.strip()
        recommendations = recommendations.drop_duplicates(subset='track_name_lower')
        recommendations.drop(columns='track_name_lower', inplace=True)

    columns = ['track_id', 'track_name', 'artists', 'album_name', 'track_genre', 'popularity']
    recommendations = recommendations[columns]

    if 'popularity' in recommendations.columns:
        if recommend_type == 'song':
            recommendations = recommendations.sort_values(by='popularity', ascending=False)
        elif recommend_type == 'artist':
            recommendations = (
                recommendations.groupby('artists', as_index=False)['popularity']
                .max()
                .sort_values(by='popularity', ascending=False)
                .reset_index(drop=True)
            )
        elif recommend_type == 'album':
            recommendations = (
                recommendations.groupby(['album_name', 'artists'], as_index=False)['popularity']
                .max()
                .sort_values(by='popularity', ascending=False)
                .reset_index(drop=True)
            )

    return recommendations.head(number_of_recommendation).to_dict(orient="records")

def recommend_songs_artists_batch(
    queries, data, combined_features,
    number_of_recommendation=5,
    search_by="song", filter_by=None,
    recommend_type="song"
):
    if not isinstance(queries, list) or not queries:
        return "Query must be a non-empty list."

    queries = [q.strip().lower() for q in queries if isinstance(q, str) and q.strip()]
    data = data.reset_index(drop=True)

    matched_indices = []
    for q in queries:
        if search_by == 'song':
            match = data[data['track_name'].str.lower().str.strip() == q]
        elif search_by == 'artist':
            match = data[data['artists'].str.lower().str.contains(q)]
        elif search_by == 'album':
            match = data[data['album_name'].str.lower().str.strip() == q]
        else:
            return "Invalid search type. Choose 'song', 'artist', or 'album'."

        if not match.empty:
            matched_indices.extend(match.index.tolist())

    if not matched_indices:
        return f"No matches found for queries: {queries}"

    matched_indices = list(set(matched_indices))

    query_vector = np.mean([combined_features[i] for i in matched_indices], axis=0)

    filtered_data = data.copy()
    filtered_features = combined_features.copy()

    reference_artist = data.loc[matched_indices[0], 'artists'].strip().lower()
    reference_genre = data.loc[matched_indices[0], 'track_genre']
    reference_album = data.loc[matched_indices[0], 'album_name']

    if filter_by == 'artist':
        mask = data['artists'].str.lower().str.contains(reference_artist)
        filtered_data = data[mask].reset_index(drop=True)
        filtered_features = combined_features[mask.values]
    elif filter_by == 'genre':
        mask = data['track_genre'] == reference_genre
        filtered_data = data[mask].reset_index(drop=True)
        filtered_features = combined_features[mask.values]
    elif filter_by == 'album':
        mask = data['album_name'] == reference_album
        filtered_data = data[mask].reset_index(drop=True)
        filtered_features = combined_features[mask.values]

    if len(filtered_data) <= 1:
        return f"Not enough data with the filter '{filter_by}' to make recommendations."

    knn = NearestNeighbors(n_neighbors=min(len(filtered_data), number_of_recommendation + len(matched_indices)), metric='cosine')
    knn.fit(filtered_features)

    distances, indices = knn.kneighbors([query_vector])
    recommended_indices = indices.flatten()

    original_local_indices = [idx for idx in matched_indices if idx in filtered_data.index]
    recommended_indices = [i for i in recommended_indices if i not in original_local_indices]

    recommendations = filtered_data.iloc[recommended_indices].copy()

    if recommendations.empty:
        return f"No recommendations found with the filter '{filter_by}'. Try removing or changing the filter."

    if 'track_name' in recommendations.columns:
        recommendations['track_name_lower'] = recommendations['track_name'].str.lower().str.strip()
        recommendations = recommendations.drop_duplicates(subset='track_name_lower')
        recommendations.drop(columns='track_name_lower', inplace=True)

    columns = ['track_id', 'track_name', 'artists', 'album_name', 'track_genre', 'popularity']
    recommendations = recommendations[columns]

    if 'popularity' in recommendations.columns:
        if recommend_type == 'song':
            recommendations = recommendations.sort_values(by='popularity', ascending=False)
        elif recommend_type == 'artist':
            recommendations = (
                recommendations.groupby('artists', as_index=False)['popularity']
                .max()
                .sort_values(by='popularity', ascending=False)
                .reset_index(drop=True)
            )
        elif recommend_type == 'album':
            recommendations = (
                recommendations.groupby(['album_name', 'artists'], as_index=False)['popularity']
                .max()
                .sort_values(by='popularity', ascending=False)
                .reset_index(drop=True)
            )

    return recommendations.head(number_of_recommendation).reset_index(drop=True)
