import pandas as pd
import streamlit as st
import numpy as np
import nltk
from nltk.corpus import wordnet

try:
    wordnet.synsets('test')
except LookupError:
    nltk.download('wordnet')
    
# Set page config for app
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the preprocessed data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Demo2.csv')
        # Convert string representation of lists to actual lists
        df['genres_processed'] = df['genres'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        return df
    except:
        st.error("Failed to load data. Please make sure 'movie_sentiment_summary.csv' is in the same directory as this app.")
        return None

# Function to find similar words
def find_similar_words(word, emotion_to_genres, all_emotions, all_genres):
    """Find similar words"""
    word = word.lower().strip()
    
    # Direct matches
    if word in all_emotions:
        return {'type': 'emotion', 'word': word}
    
    if word in [g.lower() for g in all_genres]:
        matching_genre = next(g for g in all_genres if g.lower() == word)
        return {'type': 'genre', 'word': matching_genre}
    
    # Comprehensive emotion mapping
    emotion_mapping = {
        # Joy related
        'funny': 'joy', 'hilarious': 'joy', 'happy': 'joy', 'comedy': 'joy',
        'humorous': 'joy', 'heartwarming': 'joy', 'uplifting': 'joy',
        'romantic': 'joy', 'romcom': 'joy', 'pleasant': 'joy',
        'delightful': 'joy', 'cheerful': 'joy', 'amusing': 'joy',
        'lighthearted': 'joy', 'laughter': 'joy',
        
        # Sadness related
        'sad': 'sadness', 'depressing': 'sadness', 'heartbreaking': 'sadness',
        'melancholy': 'sadness', 'tragic': 'sadness', 'gloomy': 'sadness',
        'tearful': 'sadness', 'sorrowful': 'sadness', 'grief': 'sadness',
        'weeping': 'sadness', 'somber': 'sadness', 'crying': 'sadness',
        
        # Fear related
        'scary': 'fear', 'terrifying': 'fear', 'frightening': 'fear',
        'horror': 'fear', 'creepy': 'fear', 'spooky': 'fear',
        'chilling': 'fear', 'eerie': 'fear', 'horrific': 'fear',
        'tense': 'fear', 'dread': 'fear', 'phobia': 'fear', 'panic': 'fear',
        
        # Anticipation related
        'exciting': 'anticipation', 'suspenseful': 'anticipation',
        'thrilling': 'anticipation', 'adventure': 'anticipation',
        'action-packed': 'anticipation', 'gripping': 'anticipation',
        'intriguing': 'anticipation', 'mysterious': 'anticipation',
        'expectation': 'anticipation', 'awaiting': 'anticipation',
        'suspense': 'anticipation',
        
        # Anger related
        'angry': 'anger', 'rage': 'anger', 'violent': 'anger',
        'furious': 'anger', 'intense': 'anger', 'brutal': 'anger',
        'outraged': 'anger', 'vengeance': 'anger', 'wrath': 'anger',
        'hostility': 'anger',
        
        # Surprise related
        'surprising': 'surprise', 'unexpected': 'surprise',
        'shocking': 'surprise', 'twist': 'surprise', 'unpredictable': 'surprise',
        'astonishing': 'surprise', 'amazing': 'surprise', 'startling': 'surprise',
        
        # Optimism related
        'optimistic': 'optimism', 'hopeful': 'optimism', 'inspiring': 'optimism',
        'positive': 'optimism', 'motivational': 'optimism', 'encouraging': 'optimism',
        
        # Disgust related
        'disgusting': 'disgust', 'gross': 'disgust', 'revolting': 'disgust',
        'offensive': 'disgust', 'repulsive': 'disgust', 'nauseating': 'disgust',
        'distasteful': 'disgust', 'vile': 'disgust'
    }
    
    # Check if word is in emotion mapping
    if word in emotion_mapping and emotion_mapping[word] in all_emotions:
        return {'type': 'emotion', 'word': emotion_mapping[word]}
    
    # Genre-specific mapping
    genre_mapping = {
        # Action related
        'action': 'Action', 'fight': 'Action', 'explosion': 'Action',
        'martial arts': 'Action', 'superhero': 'Action', 'stunt': 'Action',
        
        # Comedy related
        'comedy': 'Comedy', 'funny': 'Comedy', 'laugh': 'Comedy',
        'humorous': 'Comedy', 'sitcom': 'Comedy',
        
        # Drama related
        'drama': 'Drama', 'emotional': 'Drama', 'serious': 'Drama',
        'character-driven': 'Drama',
        
        # Horror related
        'horror': 'Horror', 'scary': 'Horror', 'terror': 'Horror',
        'nightmare': 'Horror', 'monster': 'Horror', 'slasher': 'Horror',
        
        # Romance related
        'romance': 'Romance', 'love': 'Romance', 'romantic': 'Romance',
        'relationship': 'Romance', 'dating': 'Romance',
        
        # Thriller related
        'thriller': 'Thriller', 'suspense': 'Thriller', 'tense': 'Thriller',
        'mystery': 'Thriller', 'crime': 'Thriller',
        
        # Sci-Fi related
        'sci-fi': 'Science Fiction', 'scifi': 'Science Fiction',
        'science fiction': 'Science Fiction', 'futuristic': 'Science Fiction',
        'space': 'Science Fiction', 'alien': 'Science Fiction',
        'dystopian': 'Science Fiction',
        
        # Fantasy related
        'fantasy': 'Fantasy', 'magical': 'Fantasy', 'mythical': 'Fantasy',
        'dragon': 'Fantasy', 'wizard': 'Fantasy', 'fairy tale': 'Fantasy',
        
        # Documentary related
        'documentary': 'Documentary', 'real life': 'Documentary',
        'educational': 'Documentary', 'factual': 'Documentary',
        
        # Animation related
        'animation': 'Animation', 'animated': 'Animation',
        'cartoon': 'Animation', 'anime': 'Animation',
        
        # Adventure related
        'adventure': 'Adventure', 'quest': 'Adventure', 'journey': 'Adventure',
        'exploration': 'Adventure', 'treasure': 'Adventure'
    }
     
    # Check if word is in genre mapping
    if word in genre_mapping and genre_mapping[word] in all_genres:
        return {'type': 'genre', 'word': genre_mapping[word]}
    
    # Multi-word genres (like "Science Fiction")
    for term, genre in genre_mapping.items():
        if ' ' in term and term in word:
            if genre in all_genres:
                return {'type': 'genre', 'word': genre}
                
    # Check for partial matches in genres and emotions
    for genre in all_genres:
        if word in genre.lower() or genre.lower() in word:
            return {'type': 'genre', 'word': genre}
    
    for emotion in all_emotions:
        if word in emotion.lower() or emotion.lower() in emotion:
            return {'type': 'emotion', 'word': emotion}
   
    # Try WordNet for synonyms
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())
    
    # Check if any synonym is in our mappings
    for synonym in synonyms:
        if synonym in emotion_mapping and emotion_mapping[synonym] in all_emotions:
            return {'type': 'emotion', 'word': emotion_mapping[synonym]}
            
        if synonym in genre_mapping and genre_mapping[synonym] in all_genres:
            return {'type': 'genre', 'word': genre_mapping[synonym]}
    
    # Check for synonyms matching emotions directly
    for synonym in synonyms:
        if synonym in all_emotions:
            return {'type': 'emotion', 'word': synonym}
    
    # Check for synonyms matching genres directly
    for synonym in synonyms:
        if synonym in [g.lower() for g in all_genres]:
            matching_genre = next(g for g in all_genres if g.lower() == synonym)
            return {'type': 'genre', 'word': matching_genre}
    
    # Final check for partial matches with synonyms
    for synonym in synonyms:
        for emotion in all_emotions:
            if synonym in emotion.lower() or emotion.lower() in synonym:
                return {'type': 'emotion', 'word': emotion}
        
        for genre in all_genres:
            if synonym in genre.lower() or genre.lower() in synonym:
                return {'type': 'genre', 'word': genre}
    
    # Default fallback - try to intelligently pick between joy and anticipation
    if any(word in term for term in ['excit', 'thrill', 'tense', 'grip', 'adventure']):
        return {'type': 'emotion', 'word': 'anticipation', 'is_suggestion': True}
    else:
        return {'type': 'emotion', 'word': 'joy', 'is_suggestion': True}

# Movie recommendation function
def get_recommendations(input_text, df, emotion_to_genres, all_emotions, all_genres, sort_by='avg_rating', top_n=10):
    """Get movie recommendations based on genre or emotion input with accuracy scores"""
    # Normalize input
    input_text = input_text.lower().strip()
    
    # Find similar words
    similar = find_similar_words(input_text, emotion_to_genres, all_emotions, all_genres)
    
    if similar.get('is_suggestion', False):
        st.info(f"Could not find an exact match for '{input_text}'. Suggesting movies with emotion: {similar['word']}")
        # Lower base accuracy for suggestions
        base_accuracy = 0.6
    else:
        st.success(f"Found match: {similar['word']} (type: {similar['type']})")
        # Higher base accuracy for direct matches
        base_accuracy = 0.9
    
    # Get recommendations based on match type
    if similar['type'] == 'emotion':
        # Filter movies by emotion
        filtered_df = df[df['emotion'].str.lower() == similar['word'].lower()].copy()
        
        # Add perfect accuracy for exact emotion matches
        filtered_df['accuracy'] = base_accuracy + 0.1
        
        # If too few results, also include related genres
        if len(filtered_df) < 5 and similar['word'] in emotion_to_genres:
            top_genres = list(emotion_to_genres[similar['word']].keys())[:3]
            st.info(f"Including movies with related genres: {', '.join(top_genres)}")
            
            # Create a condition to match any of the top genres
            genre_condition = df['genres'].apply(
                lambda x: any(genre in str(x) for genre in top_genres)
            )
            additional_df = df[genre_condition].copy()
            
            # Add lower accuracy for genre-based additions (decreasing by position)
            for i, genre in enumerate(top_genres):
                genre_matches = additional_df['genres'].apply(lambda x: genre in str(x))
                # Calculate decreasing accuracy based on how strongly the genre relates to the emotion
                additional_df.loc[genre_matches, 'accuracy'] = base_accuracy - (0.1 * (i + 1))
            
            filtered_df = pd.concat([filtered_df, additional_df]).drop_duplicates()
    
    elif similar['type'] == 'genre':
        # Filter movies containing the genre
        genre_match = similar['word']
        filtered_df = df[df['genres'].apply(lambda x: genre_match in str(x))].copy()
        
        # Calculate accuracy based on how prominent the genre is in each movie
        def calculate_genre_accuracy(genres_str, target_genre):
            # Get all genres for the movie
            if isinstance(genres_str, list):
                all_movie_genres = genres_str
            else:
                try:
                    all_movie_genres = eval(genres_str)
                except:
                    all_movie_genres = str(genres_str).replace('[', '').replace(']', '').replace("'", "").split(',')
            
            all_movie_genres = [str(g).strip() for g in all_movie_genres]
            
            # If target genre is the only genre, highest accuracy
            if len(all_movie_genres) == 1 and target_genre in str(all_movie_genres):
                return base_accuracy + 0.1
            
            # If target genre is first in the list, higher accuracy
            if str(all_movie_genres[0]).strip() == target_genre:
                return base_accuracy + 0.05
                
            # Otherwise base accuracy
            return base_accuracy
        
        filtered_df['accuracy'] = filtered_df['genres'].apply(
            lambda x: calculate_genre_accuracy(x, genre_match)
        )
    
    # Sort the results
    if sort_by == 'avg_rating':
        filtered_df = filtered_df.sort_values(by=['avg_rating', 'accuracy', 'review_count'], ascending=False)
    elif sort_by == 'avg_sentiment_score':
        filtered_df = filtered_df.sort_values(by=['avg_sentiment_score', 'accuracy', 'review_count'], ascending=False)
    
    # Ensure accuracy values are within reasonable range and formatted as percentages
    filtered_df['accuracy'] = filtered_df['accuracy'].clip(0, 1)
    filtered_df['accuracy'] = (filtered_df['accuracy'] * 100).round(1).astype(str) + '%'
    
    # Return top N results with accuracy column
    return filtered_df.head(top_n)[['movie_name', 'avg_rating', 'avg_sentiment_score', 'genres', 'emotion', 'accuracy']]

# Create emotion to genre mapping based on the dataset
def create_emotion_genre_mapping(df):
    """Create a mapping of emotions to common genres based on the dataset"""
    emotion_to_genres = {}
    
    # Process through each movie
    for idx, row in df.iterrows():
        emotion = row['emotion'].lower()
        if emotion not in emotion_to_genres:
            emotion_to_genres[emotion] = {}
        
        # Get genres
        genres = row['genres_processed']
        
        # Count genre frequency for each emotion
        for genre in genres:
            genre = str(genre).strip("'[]")
            if genre in emotion_to_genres[emotion]:
                emotion_to_genres[emotion][genre] += 1
            else:
                emotion_to_genres[emotion][genre] = 1
    
    # Sort each emotion's genres by frequency
    for emotion in emotion_to_genres:
        emotion_to_genres[emotion] = {k: v for k, v in 
                                     sorted(emotion_to_genres[emotion].items(), 
                                            key=lambda item: item[1], 
                                            reverse=True)}
    
    return emotion_to_genres

# Extract all unique emotions and genres from the dataset
def extract_emotions_and_genres(df):
    """Extract all unique emotions and genres from the dataset"""
    all_emotions = list(set(df['emotion'].str.lower()))
    
    all_genres = set()
    for genres in df['genres_processed']:
        for genre in genres:
            all_genres.add(str(genre).replace("'", "").strip())
    
    return all_emotions, list(all_genres)

# App title and description
st.title("🎬 Movie Recommendation System")
st.markdown("""
    This app helps you find movies based on your current mood or genre preferences.
    Simply enter how you're feeling or what kind of movie you're in the mood for!
""")

# Load data
df = load_data()

if df is not None:
    # Create mappings
    emotion_to_genres = create_emotion_genre_mapping(df)
    all_emotions, all_genres = extract_emotions_and_genres(df)

    # Display available emotions and genres
    with st.expander("Show available emotions and genres"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Available Emotions")
            st.write(", ".join(sorted(all_emotions)))
            
        with col2:
            st.subheader("Top Genres")
            st.write(", ".join(sorted(all_genres)[:20]))
    
    # Input for user's mood or genre preference
    st.subheader("What kind of movie are you looking for today?")
    user_input = st.text_input("Enter your mood or a genre (e.g., 'happy', 'exciting', 'comedy', 'sci-fi')", 
                              placeholder="Type what emotion or what genre you want to watch...")
    
    # Sorting preference
    sort_option = st.radio("Sort recommendations by:", ["Rating", "Sentiment Score"], horizontal=True)
    sort_by = 'avg_rating' if sort_option == "Rating" else 'avg_sentiment_score'
    
    # Number of recommendations
    num_recs = st.slider("Number of recommendations:", 5, 20, 10)
    
    # Get recommendations
    if user_input:
        with st.spinner('Finding the perfect movies for you...'):
            recommendations = get_recommendations(
                user_input, 
                df, 
                emotion_to_genres, 
                all_emotions, 
                all_genres, 
                sort_by=sort_by, 
                top_n=num_recs
            )
        
        if not recommendations.empty:
            st.subheader("Your Personalized Movie Recommendations")
            
            # Apply formatting to the avg_rating and avg_sentiment_score columns
            recommendations['avg_rating'] = recommendations['avg_rating'].round(1)
            recommendations['avg_sentiment_score'] = recommendations['avg_sentiment_score'].round(1)
            
            # Make genres more readable
            recommendations['genres'] = recommendations['genres'].apply(
                lambda x: ", ".join([g.strip("'[]") for g in eval(str(x))] if isinstance(x, str) else [g.strip("'[]") for g in x])
            )
            
            # Custom columns display
            for i, row in recommendations.iterrows():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"### {row['movie_name']}")
                    st.markdown(f"**Genres:** {row['genres']}")
                    st.markdown(f"**Emotion:** {row['emotion']}")
                    st.markdown(f"**Match:** {row['accuracy']}") 
                
                with col2:
                    st.metric("Rating", f"{row['avg_rating']}/5")
                    #st.markdown(f"**Match:** {row['accuracy']}") 
                
                with col3:
                    st.metric("Emotional Impact", f"{row['avg_sentiment_score']}/5")
                
                st.divider()
                
        else:
            st.warning("No movies found matching your criteria. Try a different emotion or genre.")
    
    

# Footer
st.markdown("---")
st.markdown("Movie Recommender App powered by NLP and sentiment analysis")

# Update user input field if mood or genre button is clicked
if 'user_input' in st.session_state:
    st.experimental_rerun()
