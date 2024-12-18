import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Title and description of the app
st.title("Movie Recommendation System")
st.write("Explore movies and get personalized recommendations based on your preferences!")

# Load dataset
@st.cache_data  # Cache data to optimize performance
def load_data():
    movies = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/movies.csv")
    movies['genres'] = movies['genres'].apply(lambda x: x.split("|"))
    movies = movies[movies['genres'] != '(no genres listed)']
    return movies

movies = load_data()

# Extract release year from titles
def extract_year_from_title(title):
    import re
    t = title.split(' ')
    year = None
    if re.search(r'\(\d+\)', t[-1]):
        year = t[-1].strip('()')
        year = int(year)
    return year

movies['year'] = movies['title'].apply(extract_year_from_title)
movies = movies[~movies['year'].isnull()]  # Remove rows with null years

# Add decade column for analysis
def round_down(year):
    return year - (year % 10)

movies['decade'] = movies['year'].apply(round_down)

# Encode genres and decades for similarity computation
from collections import Counter

genres_counts = Counter(g for genres in movies['genres'] for g in genres)
genres = list(genres_counts.keys())

for g in genres:
    movies[g] = movies['genres'].transform(lambda x: int(g in x))

movie_decades = pd.get_dummies(movies['decade'])
movie_features = pd.concat([movies[genres], movie_decades], axis=1)

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(movie_features, movie_features)

# Movie finder function using fuzzy matching
def movie_finder(title):
    all_titles = movies['title'].tolist()
    closest_match = process.extractOne(title, all_titles)
    return closest_match[0]

# Recommendation function
def get_content_based_recommendations(title_string, n_recommendations=10):
    title = movie_finder(title_string)
    idx = dict(zip(movies['title'], list(movies.index)))[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:(n_recommendations + 1)]
    similar_movies_indices = [i[0] for i in sim_scores]
    return title, movies.iloc[similar_movies_indices]

# Streamlit user interface
st.sidebar.header("Movie Search")
user_input_title = st.sidebar.text_input("Enter a movie title:", "Toy Story (1995)")
n_recommendations = st.sidebar.slider("Number of recommendations:", 1, 20, 10)

if st.sidebar.button("Get Recommendations"):
    try:
        selected_title, recommended_movies = get_content_based_recommendations(user_input_title, n_recommendations)
        st.write(f"Because you watched **{selected_title}**, you might like:")
        st.table(recommended_movies[['title', 'year', 'genres']])
    except Exception as e:
        st.error(f"Error: {e}. Please try another title.")

# Visualization: Genre distribution bar chart
if st.checkbox("Show Genre Distribution"):
    genres_counts_df = pd.DataFrame([genres_counts]).T.reset_index()
    genres_counts_df.columns = ['genres', 'count']
    genres_counts_df.sort_values(by='count', ascending=False, inplace=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='genres', y='count', data=genres_counts_df, palette='viridis', ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)

# Visualization: Movies per decade bar chart
if st.checkbox("Show Movies Per Decade"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x=movies['decade'], palette='Blues', ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)
