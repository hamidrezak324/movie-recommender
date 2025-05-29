from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import torch
from transformers import DistilBertTokenizer, DistilBertModel

# Initialize FastAPI
app = FastAPI()

# Load saved files
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
with open('movie_feature_dict.pkl', 'rb') as f:
    movie_feature_dict = pickle.load(f)
with open('movie_genre_embeddings.pkl', 'rb') as f:
    movie_genre_embeddings = pickle.load(f)
with open('genre_columns.pkl', 'rb') as f:
    genre_columns = pickle.load(f)

# Load DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Function to convert genre vectors to text
def genres_to_text(genre_vector, genre_columns):
    genres = [genre_columns[i] for i, val in enumerate(genre_vector) if val == 1]
    return ' '.join(genres) if genres else 'unknown'

# Function to build user profile
def build_user_profile(user_id, ratings, movie_feature_dict):
    user_ratings = ratings[ratings['user_id'] == user_id][['movie_id', 'rating']]
    profile = np.zeros(len(genre_columns))
    total_weight = 0
    for _, row in user_ratings.iterrows():
        movie_id = row['movie_id']
        rating = row['rating']
        if movie_id in movie_feature_dict:
            profile += rating * movie_feature_dict[movie_id]
            total_weight += rating
    if total_weight > 0:
        profile /= total_weight
    return profile

# Basic recommendation function
def recommend_movies(user_id, ratings, movie_feature_dict, movies, top_n=5):
    user_profile = build_user_profile(user_id, ratings, movie_feature_dict)
    similarities = cosine_similarity([user_profile], list(movie_feature_dict.values()))[0]
    seen_movies = set(ratings[ratings['user_id'] == user_id]['movie_id'])
    movie_scores = [(mid, score) for mid, score in zip(movie_feature_dict.keys(), similarities) if mid not in seen_movies]
    movie_scores.sort(key=lambda x: x[1], reverse=True)
    top_movies = movie_scores[:top_n]
    recommendations = movies[movies['movie_id'].isin([mid for mid, _ in top_movies])][['movie_id', 'title']].to_dict('records')
    return recommendations

# RLAIF feedback function
def rlaif_feedback(user_id, ratings, movie_genre_embeddings, learning_rate=0.5):
    user_profile = build_user_profile(user_id, ratings, movie_feature_dict)
    
    # Get movies rated 5 as reward
    feedback_ratings = ratings[(ratings['user_id'] == user_id) & (ratings['rating'] == 5)]
    reward = np.zeros(768)
    total_feedback = 0
    for _, row in feedback_ratings.iterrows():
        movie_id = row['movie_id']
        if movie_id in movie_genre_embeddings:
            reward += movie_genre_embeddings[movie_id]
            total_feedback += 1
    if total_feedback > 0:
        reward /= total_feedback
    
    # Convert user profile to DistilBERT embedding space
    user_profile_embedding = np.zeros(768)
    for movie_id, genre_vector in movie_feature_dict.items():
        genre_text = genres_to_text(genre_vector, genre_columns)
        inputs = tokenizer(genre_text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        user_profile_embedding += user_profile[np.where(genre_vector == 1)].mean() * embedding
    
    # Apply RLAIF update
    updated_profile = user_profile_embedding + learning_rate * (reward - user_profile_embedding)
    return updated_profile

# Recommendation function with updated profile (RLAIF)
def recommend_with_updated_profile(user_id, updated_profile, movie_genre_embeddings, movies, seen_movies, top_n=5):
    similarities = cosine_similarity([updated_profile], list(movie_genre_embeddings.values()))[0]
    movie_scores = [(mid, score) for mid, score in zip(movie_genre_embeddings.keys(), similarities) if mid not in seen_movies]
    movie_scores.sort(key=lambda x: x[1], reverse=True)
    top_movies = movie_scores[:top_n]
    recommendations = movies[movies['movie_id'].isin([mid for mid, _ in top_movies])][['movie_id', 'title']].to_dict('records')
    return recommendations

# Endpoint for basic recommendations
@app.get("/recommend/{user_id}")
def get_recommendations(user_id: int, top_n: int = 5):
    recommendations = recommend_movies(user_id, ratings, movie_feature_dict, movies, top_n)
    return {"user_id": user_id, "recommendations": recommendations}

# Endpoint for RLAIF-based recommendations
@app.get("/recommend_rlaif/{user_id}")
def get_recommendations_rlaif(user_id: int, top_n: int = 5):
    updated_profile = rlaif_feedback(user_id, ratings, movie_genre_embeddings)
    seen_movies = set(ratings[ratings['user_id'] == user_id]['movie_id'])
    recommendations = recommend_with_updated_profile(user_id, updated_profile, movie_genre_embeddings, movies, seen_movies, top_n)
    return {"user_id": user_id, "recommendations": recommendations}

