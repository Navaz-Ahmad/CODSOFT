import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset
# For simplicity, let's assume we have a dataset of movie ratings by users in the form of a DataFrame
# We'll create a small dataset as an example

data = {
    'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    'movie_id': [101, 102, 103, 101, 104, 105, 102, 104, 106],
    'rating': [5, 3, 4, 4, 5, 2, 5, 4, 3]
}

df = pd.DataFrame(data)

# Step 2: Create a utility matrix where rows are users and columns are movies
# Matrix will contain user ratings for movies; NaN indicates no rating
utility_matrix = df.pivot_table(index='user_id', columns='movie_id', values='rating')

# Step 3: Fill NaN values with 0 (could also use mean imputation or other strategies)
utility_matrix = utility_matrix.fillna(0)

# Step 4: Calculate the similarity matrix between users using cosine similarity
user_similarity = cosine_similarity(utility_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=utility_matrix.index, columns=utility_matrix.index)

# Step 5: Predict ratings for unrated movies by calculating weighted sum of similar users' ratings
def predict_ratings(user_id, utility_matrix, user_similarity_df):
    # Get the similarity scores for the given user with all other users
    similarity_scores = user_similarity_df[user_id]
    
    # Get the ratings of all other users
    user_ratings = utility_matrix
    
    # Calculate the weighted sum of ratings
    weighted_ratings = np.dot(similarity_scores, user_ratings) / np.array([np.abs(similarity_scores).sum()])
    
    # Return predicted ratings for the given user
    return pd.Series(weighted_ratings, index=user_ratings.columns)

# Step 6: Recommend movies to the user based on predicted ratings
def recommend_movies(user_id, utility_matrix, user_similarity_df, num_recommendations=3):
    # Predict the ratings for all movies for the user
    predicted_ratings = predict_ratings(user_id, utility_matrix, user_similarity_df)
    
    # Get the movies the user has not yet rated
    user_rated_movies = utility_matrix.loc[user_id][utility_matrix.loc[user_id] > 0].index
    unrated_movies = utility_matrix.columns.difference(user_rated_movies)
    
    # Sort the predicted ratings for unrated movies and get the top N recommendations
    recommendations = predicted_ratings[unrated_movies].sort_values(ascending=False)[:num_recommendations]
    
    return recommendations

# Example usage: Recommend 3 movies to user 1
user_id = 1
recommendations = recommend_movies(user_id, utility_matrix, user_similarity_df, num_recommendations=3)
print("Top movie recommendations for user", user_id, ":\n", recommendations)
