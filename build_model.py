import numpy as np
import pandas as pd
import os
import ast
import gzip
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# List CSV files
print("Files found:")
for f in os.listdir('.'):
    if f.endswith('.csv'):
        print(f"  {f}")

# Load data
print("\nLoading CSVs...")
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
print(f"Movies shape: {movies.shape}")
print(f"Credits shape: {credits.shape}")

# Merge
movies = movies.merge(credits, on='title')

# Select columns
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

# Drop nulls
movies.dropna(inplace=True)

# Helper functions
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter += 1
    return L

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

def collapse(L):
    return [i.replace(" ", "") for i in L]

# Parse JSON columns
print("\nProcessing data...")
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetch_director)

# Remove spaces from multi-word names
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)

# Split overview into words
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Build tags
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Final dataframe
new = movies[['movie_id', 'title', 'tags']].copy()
new['tags'] = new['tags'].apply(lambda x: " ".join(x))

print(f"Final dataset shape: {new.shape}")
print("\nSample tags for Avatar:")
print(new[new['title'] == 'Avatar']['tags'].values[0][:200])

# Vectorize
print("\nVectorizing...")
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new['tags']).toarray()
print(f"Vector shape: {vector.shape}")

# Cosine similarity
print("\nComputing cosine similarity (this may take a moment)...")
similarity = cosine_similarity(vector)
print(f"Similarity matrix shape: {similarity.shape}")

# Save model files
os.makedirs('model', exist_ok=True)
pickle.dump(new, open('model/movie_list.pkl', 'wb'))
with gzip.open('model/similarity.pkl.gz', 'wb') as f:
    pickle.dump(similarity.astype('float16'), f, protocol=4)
print("\n✅ Saved model/movie_list.pkl")
print("✅ Saved model/similarity.pkl.gz")
print("\n🎉 Done! You can now run: streamlit run app.py")
