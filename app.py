import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("mal_anime.csv")

# Data preprocessing
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

df['genres'] = df['genres'].apply(clean_data)

# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english')
df['genres'] = df['genres'].fillna('')
tfidf_matrix = tfidf.fit_transform(df['genres'])

# Cosine similarity calculation
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Function to get recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    anime_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[anime_indices]

# Streamlit app
st.title('Anime Recommendation System')
st.write("Welcome! This recommender system suggests anime based on genres to the one you choose, helping you discover new shows you might enjoy.")

selected_anime = st.selectbox('Select an anime:', df['title'])

if st.button('Get Recommendations'):
    recommendations = get_recommendations(selected_anime)
    st.subheader('Recommended Anime')
    if recommendations.size > 0:
        cnt = 1
        for anime in recommendations:
            st.write(str(cnt)+". "+anime)
            cnt+=1
    else:
        st.write("No recommendations found.")







