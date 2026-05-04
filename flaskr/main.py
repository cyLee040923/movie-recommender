from flask import (
    Blueprint, render_template, request, jsonify
)
import urllib.parse
from .tools.data_tool import *
from surprise import Reader
from surprise import KNNBasic, KNNWithMeans
from surprise import Dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import random
from collections import Counter

bp = Blueprint('main', __name__, url_prefix='/')

movies, genres, rates = loadData()

# Build a static TF-IDF content matrix for all movies (title + overview)
if 'overview' in movies.columns:
    content_texts = (movies['title'].fillna('') + ' ' + movies['overview'].fillna('')).tolist()
else:
    content_texts = movies['title'].fillna('').tolist()

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
content_tfidf_matrix = tfidf_vectorizer.fit_transform(content_texts)
# Map movieId to row index in the TF-IDF matrix
movie_id_to_index = {int(m): idx for idx, m in enumerate(movies['movieId'].tolist())}


# Helper: build a user content profile from rated movies using TF-IDF vectors and ratings as weights
def build_user_content_profile_from_ratings(user_rates_df, mapping, tfidf_matrix):
    # user_rates_df expected to have columns ['userId','movieId','rating']
    vecs = []
    weights = []
    for _, row in user_rates_df.iterrows():
        mid = int(row['movieId'])
        rating = float(row.get('rating', 0))
        idx = mapping.get(mid)
        if idx is None:
            continue
        vec = tfidf_matrix[idx].toarray().flatten()
        vecs.append(vec)
        weights.append(rating)

    if len(vecs) == 0:
        return None
    weights = np.array(weights, dtype=float)
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    profile = np.average(np.vstack(vecs), axis=0, weights=weights)
    return profile


# Reuse cookie parsing everywhere
def _parse_cookie_list(val):
    if not val:
        return []
    decoded = urllib.parse.unquote(val)
    if decoded == '':
        return []
    return [s for s in decoded.split(',') if s != '']


@bp.route('/api/search_movies', methods=['GET'])
def search_movies():
    query = request.args.get('q', '')
    if not query:
        return jsonify([])
    mask = movies['title'].str.contains(query, case=False, na=False)
    matched = movies[mask].head(20)
    result = []
    for _, row in matched.iterrows():
        result.append({
            'movieId': row['movieId'],
            'title': row['title'],
            'release_date': row.get('year', row.get('release_date', '')),
            'cover_url': row['cover_url'],
            'genres': row['genres']
        })
    return jsonify(result)



@bp.route('/', methods=('GET', 'POST'))
def index():
    default_genres = genres.to_dict('records')

    user_genres = _parse_cookie_list(request.cookies.get('user_genres'))
    user_rates = _parse_cookie_list(request.cookies.get('user_rates'))
    user_likes = _parse_cookie_list(request.cookies.get('user_likes'))
    user_dislikes = _parse_cookie_list(request.cookies.get('user_dislikes'))

    default_genres_movies = getMoviesByGenres(user_genres)[:10]
    recommendations_movies, recommendations_message = getRecommendationBy(user_rates, user_genres, user_dislikes)    
    likes_similar_movies, likes_similar_message = getLikedSimilarBy(
        [int(numeric_string) for numeric_string in user_likes]
    )
    likes_movies = getUserLikesBy(user_likes)

    return render_template(
        'index.html',
        genres=default_genres,
        user_genres=user_genres,
        user_rates=user_rates,
        user_likes=user_likes,
        default_genres_movies=default_genres_movies,
        recommendations=recommendations_movies,
        recommendations_message=recommendations_message,
        likes_similars=likes_similar_movies,
        likes_similar_message=likes_similar_message,
        likes=likes_movies,
        movies=movies.to_dict('records'),
    )


# movie details
@bp.route('/movie/<int:movieId>', methods=('GET',))
def movie_details(movieId):
    movie = movies[movies['movieId'] == movieId]
    if len(movie) > 0:
        # Fetch sequence-based recommendations
        sequence_movies = getMoviesBySequence(movieId, k=5)
        return render_template('details.html', movie=movie.iloc[0].to_dict(), sequence_movies=sequence_movies)
    return "Movie not found", 404


# My rated items page 
@bp.route('/rated', methods=('GET',))
def rated_items():
    """
    Page: My rated items
    - Reads the 'user_rates' cookie
    - Uses ratesFromUser(...) to build a user rating DataFrame
    - Joins with movies to get movie metadata
    - Sends list to rated.html as `rated_movies`
    """
    user_rates_cookie = request.cookies.get('user_rates')
    user_rates_list = _parse_cookie_list(user_rates_cookie)

    rated_movies = []

    if user_rates_list:
        # Convert cookie list to ratings DataFrame
        user_rates_df = ratesFromUser(user_rates_list)  # cols: userId, movieId, rating, (timestamp...)

        # Your system uses synthetic userId=611
        user_id = 611
        user_rated_df = user_rates_df[user_rates_df['userId'] == user_id]

        if not user_rated_df.empty:
            merged = pd.merge(
                user_rated_df,
                movies,
                on='movieId',
                how='left',
                suffixes=('_rating', '_movie')
            )

            # highest rating first, then title
            merged = merged.sort_values(
                by=['rating', 'title'],
                ascending=[False, True]
            )

            for _, row in merged.iterrows():
                rated_movies.append({
                    'movieId': int(row['movieId']),
                    'title': row['title'],
                    'year': row.get('year', row.get('release_date', '')),
                    'rating': int(row['rating']),
                })

    return render_template('rated.html', rated_movies=rated_movies)


# Refresh seed movies based on user-selected genres, excluding already rated/liked movies
@bp.route('/api/refresh_seed_movies', methods=['POST'])
def refresh_seed_movies():
    data = request.get_json()
    user_genres = data.get('user_genres', [])
    exclude_ids = data.get('exclude_ids', [])

    interested_genres = []
    for gid in user_genres:
        gid = int(gid)
        genre_name = genres[genres['id'] == gid]['name'].values
        if len(genre_name) > 0:
            interested_genres.append(genre_name[0])

    if not interested_genres:
        return jsonify([])

    mask = movies['genres'].apply(lambda x: is_genre_match(x, interested_genres))
    candidate = movies[mask]
    candidate = candidate[~candidate['movieId'].isin(exclude_ids)]

    if len(candidate) == 0:
        return jsonify([])

    n = min(10, len(candidate))
    sampled = candidate.sample(n)

    result = []
    for _, row in sampled.iterrows():
        result.append({
            'movieId': row['movieId'],
            'title': row['title'],
            'release_date': row.get('year', row.get('release_date', '')),
            'cover_url': row['cover_url'],
            'genres': row['genres']
        })
    return jsonify(result)


def getUserLikesBy(user_likes):
    results = []

    if len(user_likes) > 0:
        mask = movies['movieId'].isin([int(movieId) for movieId in user_likes])
        results = movies.loc[mask]

        original_orders = pd.DataFrame()
        for _id in user_likes:
            movie = results.loc[results['movieId'] == int(_id)]
            if len(original_orders) == 0:
                original_orders = movie
            else:
                original_orders = pd.concat([movie, original_orders])
        results = original_orders

    if len(results) > 0:
        return results.to_dict('records')
    return results


def is_genre_match(movie_genres, interested_genres):
    if isinstance(movie_genres, str):
        movie_genres = movie_genres.split('|')
    return bool(set(movie_genres).intersection(set(interested_genres)))


def getMoviesByGenres(user_genres):
    results = []
    if len(user_genres) > 0:
        genres_mask = genres['id'].isin([int(id) for id in user_genres])
        user_genres_flags = [1 if has is True else 0 for has in genres_mask]
        user_genres_df = pd.DataFrame(user_genres_flags, columns=['value'])
        user_genres_df = pd.concat([user_genres_df, genres['name']], axis=1)
        interested_genres = user_genres_df[user_genres_df['value'] == 1]['name'].tolist()
        results = movies[movies['genres'].apply(lambda x: is_genre_match(x, interested_genres))]

    if len(results) > 0:
        return results.to_dict('records')
    return results


def getRecommendationBy(user_rates, user_genres=None, user_dislikes=None):
    results = []
    if len(user_rates) > 0:
        reader = Reader(rating_scale=(1, 5))
        algo = KNNWithMeans(sim_options={'name': 'pearson', 'user_based': True})
        user_rates = ratesFromUser(user_rates)
        training_rates = pd.concat([rates, user_rates], ignore_index=True)
        training_data = Dataset.load_from_df(training_rates[['userId', 'movieId', 'rating']], reader=reader)
        trainset = training_data.build_full_trainset()
        algo.fit(trainset)
        all_movie_ids = movies['movieId'].unique()
        user_id = 611
        rated_movie_ids = user_rates[user_rates['userId'] == user_id]['movieId'].tolist()

        # genre-filtered candidates
        candidate_movie_ids = [mid for mid in all_movie_ids if mid not in rated_movie_ids]

        if user_genres:
            # Map genre IDs -> names
            interested_genres = []
            for gid in user_genres:
                gid = int(gid)
                genre_name = genres[genres['id'] == gid]['name'].values
                if len(genre_name) > 0:
                    interested_genres.append(genre_name[0])

            # if interested_genres:
            #     def _match_genres(mid):
            #         row = movies[movies['movieId'] == mid]
            #         if row.empty:
            #             return False
            #         return is_genre_match(row.iloc[0]['genres'], interested_genres)

            if interested_genres:
                genre_mask = movies['genres'].apply(lambda x: is_genre_match(x, interested_genres))
                valid_ids = set(movies[genre_mask]['movieId'].astype(int).tolist())
                candidate_movie_ids = [mid for mid in candidate_movie_ids if int(mid) in valid_ids]

                #candidate_movie_ids = [mid for mid in candidate_movie_ids if _match_genres(mid)]
                #movie_row = movies[movies['movieId'] == mid]

        # If nothing left after filtering, fall back
        if len(candidate_movie_ids) == 0:
            candidate_movie_ids = [mid for mid in all_movie_ids if mid not in rated_movie_ids]

        #dislike filtering
        if user_dislikes:
            disliked_ids = set(int(d) for d in user_dislikes if d)
            candidate_movie_ids = [mid for mid in candidate_movie_ids if int(mid) not in disliked_ids]    

        # predictions + TF-IDF hybrid
        predictions = [algo.predict(user_id, movie_id) for movie_id in candidate_movie_ids]
        coll_dict = {int(pred.iid): float(pred.est) for pred in predictions}

        user_rated_df = user_rates[user_rates['userId'] == user_id]

        # === Dislike influence: treat dislikes as rating=1 in CF training ===
        if user_dislikes:
            dislike_rows = []
            for did in user_dislikes:
                if did:
                    dislike_rows.append({'userId': user_id, 'movieId': int(did), 'rating': 1.0})
            if dislike_rows:
                dislike_df = pd.DataFrame(dislike_rows)
                # Only add if not already rated
                already_rated = set(user_rated_df['movieId'].tolist())
                dislike_df = dislike_df[~dislike_df['movieId'].isin(already_rated)]
                training_rates = pd.concat([training_rates, dislike_df], ignore_index=True)
                # Rebuild model with dislike signals
                training_data = Dataset.load_from_df(training_rates[['userId', 'movieId', 'rating']], reader=reader)
                trainset = training_data.build_full_trainset()
                algo.fit(trainset)

        user_profile = build_user_content_profile_from_ratings(
            user_rated_df, movie_id_to_index, content_tfidf_matrix)

        # === Dislike influence: subtract disliked content from user profile ===
        if user_profile is not None and user_dislikes:
            dislike_vecs = []
            for did in user_dislikes:
                if did:
                    idx = movie_id_to_index.get(int(did))
                    if idx is not None:
                        dislike_vecs.append(content_tfidf_matrix[idx].toarray().flatten())
            if dislike_vecs:
                dislike_centroid = np.mean(np.vstack(dislike_vecs), axis=0)
                # Subtract dislike direction, clip at 0
                user_profile = np.clip(user_profile - 0.3 * dislike_centroid, 0, None)
                norm = np.linalg.norm(user_profile)
                if norm > 0:
                    user_profile = user_profile / norm

        content_scores = {}
        if user_profile is not None:
            sims = cosine_similarity(user_profile.reshape(1, -1), content_tfidf_matrix)[0]
            for mid in candidate_movie_ids:
                idx = movie_id_to_index.get(int(mid))
                if idx is not None:
                    content_scores[int(mid)] = float(sims[idx])
                else:
                    content_scores[int(mid)] = 0.0

        coll_vals = np.array(list(coll_dict.values())) if len(coll_dict) > 0 else np.array([])
        if coll_vals.size > 0 and coll_vals.max() - coll_vals.min() > 0:
            coll_min, coll_max = coll_vals.min(), coll_vals.max()
        else:
            coll_min, coll_max = 0.0, 1.0

        content_vals = np.array(list(content_scores.values())) if len(content_scores) > 0 else np.array([])
        if content_vals.size > 0 and content_vals.max() - content_vals.min() > 0:
            cont_min, cont_max = content_vals.min(), content_vals.max()
        else:
            cont_min, cont_max = 0.0, 1.0

        alpha = 0.65

        # Get user's top rated movie title for reason generation
        top_rated_title = ""
        user_rated_df = user_rates[user_rates['userId'] == user_id]
        if not user_rated_df.empty:
            best_row = user_rated_df.loc[user_rated_df['rating'].idxmax()]
            best_movie = movies[movies['movieId'] == int(best_row['movieId'])]
            if not best_movie.empty:
                top_rated_title = best_movie.iloc[0]['title']

        combined_scores = []
        for mid in candidate_movie_ids:
            mid = int(mid)
            est = coll_dict.get(mid, np.mean(coll_vals) if coll_vals.size > 0 else 0.0)
            cont = content_scores.get(mid, 0.0)
            est_norm = (est - coll_min) / (coll_max - coll_min) if coll_max - coll_min > 0 else 0.0
            cont_norm = (cont - cont_min) / (cont_max - cont_min) if cont_max - cont_min > 0 else cont
            score = alpha * est_norm + (1 - alpha) * cont_norm

            movie_row = movies[movies['movieId'] == mid]
            
            # Generate reason for each recommendation
            if est_norm >= cont_norm:
                reason = "Highly rated by users with similar taste"
            elif top_rated_title:
                reason = f"Similar content to \"{top_rated_title}\""
            else:
                reason = "Matches your content preferences"
            if not movie_row.empty and interested_genres:
                movie_genres = movie_row.iloc[0]['genres']
                matched = [g for g in interested_genres if g in movie_genres]
                if matched:
                    reason += f" · {', '.join(matched[:2])}"
            combined_scores.append((mid, score, reason))

        combined_scores.sort(key=lambda x: x[1], reverse=True)
        top_results = combined_scores[:12]
        top_movie_ids = [mid for mid, _, _ in top_results]
        reason_map = {mid: reason for mid, _, reason in top_results}

        results_df = movies[movies['movieId'].isin(top_movie_ids)]
        results = []
        for record in results_df.to_dict('records'):
            record['reason'] = reason_map.get(int(record['movieId']), "Recommended for you")
            results.append(record)

        if results:
            return results, "These movies are recommended based on your ratings, selected genres, and content similarity."
    return [], "No recommendations."
    


def getLikedSimilarBy(user_likes):
    results = []
    if len(user_likes) > 0:
        item_rep_matrix, item_rep_vector, feature_list = item_representation_based_movie_genres(movies)
        user_profile = build_user_profile(user_likes, item_rep_vector, feature_list)
        results = generate_recommendation_results(user_profile, item_rep_matrix, item_rep_vector, 12)
    if len(results) > 0:
        return results.to_dict('records'), "The movies are similar to your liked movies."
    return results, "No similar movies found."


def item_representation_based_movie_genres(movies_df):
    movies_with_genres = movies_df.copy(deep=True)
    genre_list = []
    for index, row in movies_df.iterrows():
        for genre in row['genres']:
            movies_with_genres.at[index, genre] = 1
            if genre not in genre_list:
                genre_list.append(genre)

    movies_with_genres = movies_with_genres.fillna(0)
    movies_genre_matrix = movies_with_genres[genre_list].to_numpy()

    return movies_genre_matrix, movies_with_genres, genre_list


def build_user_profile(movieIds, item_rep_vector, feature_list, weighted=True, normalized=True):
    user_movie_rating_df = item_rep_vector[item_rep_vector['movieId'].isin(movieIds)]
    user_movie_df = user_movie_rating_df[feature_list].mean()
    user_profile = user_movie_df.T

    if normalized:
        user_profile = user_profile / sum(user_profile.values)

    return user_profile


def generate_recommendation_results(user_profile, item_rep_matrix, movies_data, k=12):
    u_v = user_profile.values
    u_v_matrix = [u_v]
    recommendation_table = cosine_similarity(u_v_matrix, item_rep_matrix)
    recommendation_table_df = movies_data.copy(deep=True)
    recommendation_table_df['similarity'] = recommendation_table[0]
    rec_result = recommendation_table_df.sort_values(by=['similarity'], ascending=False)[:k]
    return rec_result


# Session-based KNN approach leveraging timestamps
def getMoviesBySequence(target_movieId, k=5):
    # Find all users who watched the target movie
    target_rates = rates[rates['movieId'] == target_movieId]
    if target_rates.empty:
        return []

    subsequent_movies = []

    # For each user, find what they watched after the target movie
    for _, row in target_rates.iterrows():
        uid = row['userId']
        ts = row['timestamp']

        # Filter for movies watched by this user AFTER the target movie's timestamp
        user_future_rates = rates[(rates['userId'] == uid) & (rates['timestamp'] > ts)]
        subsequent_movies.extend(user_future_rates['movieId'].tolist())

    if not subsequent_movies:
        return []

    # Count frequencies of subsequent movies to find the most common "next" items
    movie_counts = Counter(subsequent_movies)
    top_movie_ids = [m_id for m_id, count in movie_counts.most_common(k)]

    # Fetch movie details and preserve the frequency-based order
    results = movies[movies['movieId'].isin(top_movie_ids)]
    ordered_results = []
    for m_id in top_movie_ids:
        movie_row = results[results['movieId'] == m_id]
        if not movie_row.empty:
            ordered_results.append(movie_row.iloc[0].to_dict())

    return ordered_results