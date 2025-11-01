#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import warnings
warnings.filterwarnings('ignore')

# Import the MovieRecommendationSystem class
# Note: Save the main recommendation system as movie_recommender.py
# and import it here: from movie_recommender import MovieRecommendationSystem

class MovieRecommendationSystem:
    def __init__(self, dataset_path=None):
        """Initialize the Movie Recommendation System"""
        self.movies_df = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.cosine_sim = None
        
        if dataset_path:
            self.load_dataset(dataset_path)
        else:
            self.create_sample_dataset()
    
    def create_sample_dataset(self):
        """Create a sample movie dataset for demonstration"""
        sample_data = {
            'title': [
                'The Dark Knight', 'Inception', 'The Matrix', 'Avatar', 'Titanic',
                'The Godfather', 'Pulp Fiction', 'Forrest Gump', 'The Shawshank Redemption',
                'Fight Club', 'Interstellar', 'The Lion King', 'Toy Story', 'Finding Nemo',
                'Iron Man', 'The Avengers', 'Spider-Man', 'Batman Begins', 'Joker',
                'Parasite', 'La La Land', 'The Grand Budapest Hotel', 'Moonlight',
                'Get Out', 'Black Panther', 'Wonder Woman', 'Aquaman', 'Shazam',
                'Guardians of the Galaxy', 'Thor', 'Captain America', 'Doctor Strange',
                'Ant-Man', 'The Incredible Hulk', 'Captain Marvel', 'Deadpool',
                'Logan', 'X-Men', 'Fantastic Four', 'The Fantastic Beasts'
            ],
            'genres': [
                'Action Crime Drama', 'Action Sci-Fi Thriller', 'Action Sci-Fi', 'Action Adventure Sci-Fi', 'Drama Romance',
                'Crime Drama', 'Crime Drama', 'Drama Romance', 'Drama',
                'Drama Thriller', 'Adventure Drama Sci-Fi', 'Animation Adventure Family', 'Animation Adventure Family', 'Animation Adventure Family',
                'Action Adventure Sci-Fi', 'Action Adventure Sci-Fi', 'Action Adventure Sci-Fi', 'Action Crime Drama', 'Crime Drama Thriller',
                'Comedy Drama Thriller', 'Comedy Drama Musical', 'Adventure Comedy Drama', 'Drama',
                'Horror Mystery Thriller', 'Action Adventure Sci-Fi', 'Action Adventure Fantasy', 'Action Adventure Sci-Fi', 'Action Adventure Comedy',
                'Action Adventure Comedy Sci-Fi', 'Action Adventure Fantasy', 'Action Adventure Sci-Fi', 'Action Adventure Fantasy Sci-Fi',
                'Action Adventure Comedy Sci-Fi', 'Action Adventure Sci-Fi', 'Action Adventure Sci-Fi', 'Action Adventure Comedy Sci-Fi',
                'Action Adventure Sci-Fi', 'Action Adventure Sci-Fi', 'Action Adventure Sci-Fi', 'Adventure Fantasy Sci-Fi'
            ],
            'description': [
                'A dark knight fights crime in Gotham City with advanced technology and martial arts skills.',
                'A skilled thief enters people\'s dreams to steal secrets but faces his most challenging mission.',
                'A computer hacker discovers the reality he knows is actually a simulated world controlled by machines.',
                'A paraplegic marine becomes part of the Avatar program on an alien planet called Pandora.',
                'A tragic love story aboard the ill-fated RMS Titanic during its maiden voyage.',
                'The aging patriarch of an organized crime dynasty transfers control to his reluctant son.',
                'The lives of two mob hitmen, a boxer, and other criminals intertwine in Los Angeles.',
                'A simple man with low IQ achieves extraordinary things and influences historical events.',
                'A banker convicted of murdering his wife forms friendships and finds redemption in prison.',
                'An insomniac office worker forms an underground fight club with a soap salesman.',
                'A team of explorers travels through a wormhole in space to save humanity.',
                'A young lion prince flees his kingdom after his father\'s death and later returns to reclaim his throne.',
                'A cowboy toy feels threatened by a new space ranger toy in a child\'s room.',
                'A clownfish searches for his son who was captured by divers and taken to a fish tank.',
                'A billionaire industrialist builds a high-tech suit of armor to fight crime and terrorism.',
                'Superheroes assemble to fight an alien invasion threatening Earth.',
                'A teenager gains spider powers and learns to be a superhero in New York City.',
                'A young Bruce Wayne begins his journey to become Batman and fight crime in Gotham.',
                'A failed comedian descends into madness and becomes the criminal mastermind known as Joker.',
                'A poor family schemes to infiltrate a wealthy household with unexpected consequences.',
                'A jazz musician and actress fall in love while pursuing their dreams in Los Angeles.',
                'The adventures of a legendary concierge at a famous European hotel and his protégé.',
                'A young black man struggles with his identity and sexuality in a rough Miami neighborhood.',
                'A young black man visits his white girlfriend\'s family estate and uncovers disturbing secrets.',
                'The king of Wakanda fights to protect his nation from enemies seeking to exploit its resources.',
                'An Amazon princess leaves her island home to fight in World War I and save mankind.',
                'The half-human, half-Atlantean ruler must unite the underwater and surface worlds.',
                'A teenage boy gains superpowers and must learn to use them responsibly.',
                'A group of intergalactic criminals become unlikely heroes to save the galaxy.',
                'The Norse god of thunder must prove himself worthy of his powers and hammer.',
                'A weakly man becomes a super-soldier during World War II to fight against evil forces.',
                'A former neurosurgeon becomes a master of the mystic arts after a car accident.',
                'A thief becomes a superhero with the ability to shrink in size while gaining strength.',
                'A scientist transforms into a giant green monster when he becomes angry.',
                'A pilot gains incredible powers and becomes one of the universe\'s most powerful heroes.',
                'A wisecracking mercenary with accelerated healing powers fights crime in his own chaotic way.',
                'An aging wolverine cares for Professor X while confronting his own mortality.',
                'Mutants with special powers fight for acceptance in a world that fears them.',
                'A team of scientists gains superpowers after exposure to cosmic radiation.',
                'A young wizard discovers magical creatures and adventures in the wizarding world.'
            ]
        }
        
        self.movies_df = pd.DataFrame(sample_data)
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        
        return text
    
    def create_feature_matrix(self):
        """Create TF-IDF feature matrix from movie descriptions and genres"""
        if self.movies_df is None:
            return
        
        combined_features = []
        
        for idx, row in self.movies_df.iterrows():
            genres = self.preprocess_text(str(row.get('genres', '')))
            description = self.preprocess_text(str(row.get('description', '')))
            combined_text = f"{genres} {genres} {description}"
            combined_features.append(combined_text)
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.8,
            min_df=2
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(combined_features)
    
    def calculate_similarity(self):
        """Calculate cosine similarity matrix"""
        if self.tfidf_matrix is None:
            return
        
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
    
    def find_movie_index(self, movie_title):
        """Find the index of a movie by title"""
        exact_match = self.movies_df[self.movies_df['title'].str.lower() == movie_title.lower()]
        if not exact_match.empty:
            return exact_match.index[0]
        
        partial_match = self.movies_df[self.movies_df['title'].str.lower().str.contains(movie_title.lower(), na=False)]
        if not partial_match.empty:
            return partial_match.index[0]
        
        return None
    
    def get_recommendations(self, movie_title, num_recommendations=5):
        """Get movie recommendations based on cosine similarity"""
        if self.cosine_sim is None:
            self.create_feature_matrix()
            self.calculate_similarity()
        
        movie_idx = self.find_movie_index(movie_title)
        
        if movie_idx is None:
            return []
        
        sim_scores = list(enumerate(self.cosine_sim[movie_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        similar_movies = sim_scores[1:num_recommendations+1]
        
        recommendations = []
        for idx, score in similar_movies:
            movie_info = {
                'title': self.movies_df.iloc[idx]['title'],
                'genres': self.movies_df.iloc[idx]['genres'],
                'similarity_score': round(score, 3),
                'description': self.movies_df.iloc[idx]['description']
            }
            recommendations.append(movie_info)
        
        return recommendations

# Initialize the recommendation system
@st.cache_resource
def load_recommender():
    recommender = MovieRecommendationSystem()
    recommender.create_feature_matrix()
    recommender.calculate_similarity()
    return recommender

def main():
    st.set_page_config(
        page_title="Movie Recommender",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 Movie Recommendation System")
    st.markdown("Find your next favorite movie based on content similarity!")
    
    # Load the recommender
    recommender = load_recommender()
    
    # Create columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Search Options")
        
        # Movie selection
        movie_titles = recommender.movies_df['title'].tolist()
        selected_movie = st.selectbox(
            "Select a movie:",
            [""] + movie_titles,
            index=0
        )
        
        # Or type manually
        manual_input = st.text_input("Or type a movie title:")
        
        # Number of recommendations
        num_recommendations = st.slider(
            "Number of recommendations:",
            min_value=1,
            max_value=10,
            value=5
        )
        
        # Get recommendations button
        if st.button("Get Recommendations", type="primary"):
            movie_to_recommend = manual_input.strip() if manual_input.strip() else selected_movie
            
            if movie_to_recommend:
                recommendations = recommender.get_recommendations(movie_to_recommend, num_recommendations)
                
                if recommendations:
                    st.session_state.recommendations = recommendations
                    st.session_state.source_movie = movie_to_recommend
                else:
                    st.error(f"Movie '{movie_to_recommend}' not found!")
            else:
                st.warning("Please select or enter a movie title.")
    
    with col2:
        st.header("Recommendations")
        
        if 'recommendations' in st.session_state:
            st.subheader(f"Movies similar to '{st.session_state.source_movie}'")
            
            for i, movie in enumerate(st.session_state.recommendations, 1):
                with st.expander(f"{i}. {movie['title']} (Score: {movie['similarity_score']})"):
                    st.write(f"**Genres:** {movie['genres']}")
                    st.write(f"**Description:** {movie['description']}")
                    st.write(f"**Similarity Score:** {movie['similarity_score']}")
        else:
            st.info("Select a movie and click 'Get Recommendations' to see similar movies.")
    
    # Display dataset info
    st.sidebar.header("Dataset Information")
    st.sidebar.info(f"Total movies: {len(recommender.movies_df)}")
    
    # Display available movies
    with st.sidebar.expander("Available Movies"):
        for title in movie_titles:
            st.write(f"• {title}")
    
    # About section
    st.sidebar.header("About")
    st.sidebar.markdown("""
    This movie recommendation system uses:
    - **TF-IDF Vectorization** for text processing
    - **Cosine Similarity** for finding similar movies
    - **Content-based filtering** based on genres and descriptions
    """)

if __name__ == "__main__":
    main()


# In[ ]:




