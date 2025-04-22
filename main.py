import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from ast import literal_eval

# Set page title and configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Load the dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("movie_sentiment_summary.csv")
        # Convert string representation of lists to actual lists
        if isinstance(df['genres'].iloc[0], str):
            try:
                df['genres_processed'] = df['genres'].apply(literal_eval)
            except:
                # If literal_eval fails, create a simplified version
                df['genres_processed'] = df['genres'].apply(
                    lambda x: [g.strip().replace("'", "") for g in str(x).replace('[', '').replace(']', '').split(',')]
                )
        else:
            df['genres_processed'] = df['genres']
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Create a sample dataframe to avoid errors
        return pd.DataFrame({
            'movie_name': ['Sample Movie'],
            'avg_rating': [4.5],
            'avg_sentiment_score': [4.0],
            'genres': ["['Drama']"],
            'genres_processed': [['Drama']],
            'emotion': ['joy']
        })

# Function to find similar words
def find_similar_words(word, all_emotions, all_genres):
    """Find similar words to the input using pre-defined mappings"""
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
        'funny': 'joy',
        'hilarious': 'joy',
        'happy': 'joy',
        'comedy': 'joy',
        'humorous': 'joy',
        'heartwarming': 'joy',
        'uplifting': 'joy',
        'romantic': 'joy',
        'romcom': 'joy',
        'pleasant': 'joy',
        'delightful': 'joy',
        'cheerful': 'joy',
        'amusing': 'joy',
        'lighthearted': 'joy',
        'laughter': 'joy',
        
        # Sadness related
        'sad': 'sadness',
        'depressing': 'sadness',
        'heartbreaking': 'sadness',
        'melancholy': 'sadness',
        'tragic': 'sadness',
        'gloomy': 'sadness',
        'tearful': 'sadness',
        'sorrowful': 'sadness',
        'grief': 'sadness',
        'weeping': 'sadness',
        'somber': 'sadness',
        'crying': 'sadness',
        
        # Fear related
        'scary': 'fear',
        'terrifying': 'fear',
        'frightening': 'fear',
        'horror': 'fear',
        'creepy': 'fear',
        'spooky': 'fear',
        'chilling': 'fear',
        'eerie': 'fear',
        'horrific': 'fear',
        'tense': 'fear',
        'dread': 'fear',
        'phobia': 'fear',
        'panic': 'fear',
        
        # Anticipation related
        'exciting': 'anticipation',
        'suspenseful': 'anticipation',
        'thrilling': 'anticipation',
        'adventure': 'anticipation',
        'action-packed': 'anticipation',
        'gripping': 'anticipation',
        'intriguing': 'anticipation',
        'mysterious': 'anticipation',
        'expectation': 'anticipation',
        'awaiting': 'anticipation',
        'suspense': 'anticipation',
        
        # Anger related
        'angry': 'anger',
        'rage': 'anger',
        'violent': 'anger',
        'furious': 'anger',
        'intense': 'anger',
        'brutal': 'anger',
        'outraged': 'anger',
        'vengeance': 'anger',
        'wrath': 'anger',
        'hostility': 'anger',
        
        # Surprise related
        'surprising': 'surprise',
        'unexpected': 'surprise',
        'shocking': 'surprise',
        'twist': 'surprise',
        'unpredictable': 'surprise',
        'astonishing': 'surprise',
        'amazing': 'surprise',
        'startling': 'surprise',
        
        # Optimism related
        'optimistic': 'optimism',
        'hopeful': 'optimism',
        'inspiring': 'optimism',
        'positive': 'optimism',
        'motivational': 'optimism',
        'encouraging': 'optimism',
        
        # Disgust related
        'disgusting': 'disgust',
        'gross': 'disgust',
        'revolting': 'disgust',
        'offensive': 'disgust',
        'repulsive': 'disgust',
        'nauseating': 'disgust',
        'distasteful': 'disgust',
        'vile': 'disgust'
    }
    
    # Check if word is in emotion mapping
    if word in emotion_mapping and emotion_mapping[word] in all_emotions:
        return {'type': 'emotion', 'word': emotion_mapping[word]}
    
    # Genre-specific mapping
    genre_mapping = {
        # Action related
        'action': 'Action',
        'fight': 'Action',
        'explosion': 'Action',
        'martial arts': 'Action',
        'superhero': 'Action',
        'stunt': 'Action',
        
        # Comedy related
        'comedy': 'Comedy',
        'funny': 'Comedy',
        'laugh': 'Comedy',
        'humorous': 'Comedy',
        'sitcom': 'Comedy',
        
        # Drama related
        'drama': 'Drama',
        'emotional': 'Drama',
        'serious': 'Drama',
        'character-driven': 'Drama',
        
        # Horror related
        'horror': 'Horror',
        'scary': 'Horror',
        'terror': 'Horror',
        'nightmare': 'Horror',
        'monster': 'Horror',
        'slasher': 'Horror',
        
        # Romance related
        'romance': 'Romance',
        'love': 'Romance',
        'romantic': 'Romance',
        'relationship': 'Romance',
        'dating': 'Romance',
        
        # Thriller related
        'thriller': 'Thriller',
        'suspense': 'Thriller',
        'tense': 'Thriller',
        'mystery': 'Thriller',
        'crime': 'Thriller',
        
        # Sci-Fi related
        'sci-fi': 'Science Fiction',
        'scifi': 'Science Fiction',
        'science fiction': 'Science Fiction',
        'futuristic': 'Science Fiction',
        'space': 'Science Fiction',
        'alien': 'Science Fiction',
        'dystopian': 'Science Fiction',
        
        # Fantasy related
        'fantasy': 'Fantasy',
        'magical': 'Fantasy',
        'mythical': 'Fantasy',
        'dragon': 'Fantasy',
        'wizard': 'Fantasy',
        'fairy tale': 'Fantasy',
        
        # Documentary related
        'documentary': 'Documentary',
        'real life': 'Documentary',
        'educational': 'Documentary',
        'factual': 'Documentary',
        
        # Animation related
        'animation': 'Animation',
        'animated': 'Animation',
        'cartoon': 'Animation',
        'anime': 'Animation',
        
        # Adventure related
        'adventure': 'Adventure',
        'quest': 'Adventure',
        'journey': 'Adventure',
        'exploration': 'Adventure',
        'treasure': 'Adventure'
    }
     
    # Check if word is in genre mapping
    if word in genre_mapping:
        return {'type': 'genre', 'word': genre_mapping[word]}
    
    # Multi-word genres (like "Science Fiction")
    for term, genre in genre_mapping.items():
        if ' ' in term and term in word:
            return {'type': 'genre', 'word': genre}
    
    # Default fallback - try to intelligently pick between joy and anticipation
    if any(word in term for term in ['excit', 'thrill', 'tense', 'grip', 'adventure']):
        return {'type': 'emotion', 'word': 'anticipation', 'is_suggestion': True}
    else:
        return {'type': 'emotion', 'word': 'joy', 'is_suggestion': True}

# Function to get movie recommendations based on genre or emotion
def get_genre_emotion_recommendations(input_text, df, all_emotions, all_genres, sort_by='avg_rating', top_n=10):
    """Get movie recommendations based on genre or emotion input with accuracy scores"""
    # Normalize input
    input_text = input_text.lower().strip()
    
    # Find similar words
    similar = find_similar_words(input_text, all_emotions, all_genres)
    
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
                    all_movie_genres = literal_eval(genres_str)
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

# Main function
def main():
    # Title and description
    st.title("🎬 Movie Recommendation System")
    st.markdown("""
    This app recommends movies based on your preferred genre or mood. Just type what you're looking for, 
    whether it's a specific genre like 'Comedy' or an emotion like 'Happy', 'Sad', or 'Exciting'.
    """)
    
    # Load data
    df = load_data()
    
    # Extract all unique emotions and genres
    all_emotions = list(set(df['emotion'].str.lower()))
    
    all_genres = set()
    for genres in df['genres_processed']:
        for genre in genres:
            all_genres.add(str(genre).replace("'", "").strip())
    all_genres = list(all_genres)
    
    # Create emotion to genre mapping
    emotion_to_genres = {}
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
    
    # Sidebar for input controls
    st.sidebar.header("What kind of movie are you looking for?")
    
    # Input for genre or emotion
    user_input = st.sidebar.text_input("Enter a genre (e.g., 'Comedy') or emotion (e.g., 'Happy', 'Sad', 'Exciting')")
    
    # Radio button for sorting method
    sort_method = st.sidebar.radio(
        "Sort results by:",
        ['Average Rating', 'Sentiment Score'],
        horizontal=True
    )
    
    sort_by = 'avg_rating' if sort_method == 'Average Rating' else 'avg_sentiment_score'
    
    # Add information about available options
    with st.sidebar.expander("Available Emotions"):
        st.write(", ".join(sorted(all_emotions)))
    
    with st.sidebar.expander("Available Genres"):
        st.write(", ".join(sorted(all_genres)))
    
    # Show example emotions and their top genres
    with st.sidebar.expander("Top genres for each emotion"):
        for emotion, genres in emotion_to_genres.items():
            top_genres = list(genres.keys())[:3]
            st.write(f"- {emotion}: {', '.join(top_genres)}")
    
    # Get recommendations when the user has entered input
    if user_input:
        try:
            recommendations = get_genre_emotion_recommendations(
                user_input, df, all_emotions, all_genres, sort_by
            )
            
            # Display recommendations
            st.subheader(f"Top Recommendations for '{user_input}'")
            st.markdown(f"*Sorted by {sort_method}*")
            
            if recommendations.empty:
                st.warning("No movies found matching your criteria. Try a different genre or emotion.")
            else:
                # Format the dataframe
                recommendations = recommendations.reset_index(drop=True)
                
                # Create three columns
                col1, col2, col3 = st.columns([4, 1, 1])
                
                with col1:
                    st.markdown("#### Movie Title")
                    for i, title in enumerate(recommendations['movie_name']):
                        st.markdown(f"**{i+1}. {title}**")
                
                with col2:
                    st.markdown("#### Rating")
                    for rating in recommendations['avg_rating']:
                        st.markdown(f"⭐ {rating:.2f}")
                
                with col3:
                    st.markdown("#### Sentiment")
                    for sentiment in recommendations['avg_sentiment_score']:
                        st.markdown(f"😊 {sentiment:.2f}")
                
                # Show detailed view with genres and emotions
                with st.expander("Show detailed information"):
                    # Convert genres column to a more readable format
                    recommendations['genres_display'] = recommendations['genres'].apply(
                        lambda x: ", ".join([g.strip().replace("'", "") for g in str(x).replace('[', '').replace(']', '').split(',')])
                    )
                    
                    # Select and rename columns for display
                    display_df = recommendations[['movie_name', 'avg_rating', 'avg_sentiment_score', 'genres_display', 'emotion', 'accuracy']]
                    display_df = display_df.rename(columns={
                        'movie_name': 'Movie Title',
                        'avg_rating': 'Rating',
                        'avg_sentiment_score': 'Sentiment Score',
                        'genres_display': 'Genres',
                        'emotion': 'Emotion',
                        'accuracy': 'Match Accuracy'
                    })
                    
                    st.dataframe(display_df, hide_index=True)
                
                # Visualize the ratings distribution
                with st.expander("Rating Distribution"):
                    fig, ax = plt.subplots(figsize=(10, 5))
                    
                    bars = ax.bar(
                        recommendations['movie_name'], 
                        recommendations['avg_rating'], 
                        alpha=0.7, 
                        color='skyblue', 
                        label='Rating'
                    )
                    
                    # Add sentiment score line
                    ax2 = ax.twinx()
                    ax2.plot(
                        recommendations['movie_name'], 
                        recommendations['avg_sentiment_score'], 
                        'o-', 
                        color='orange', 
                        label='Sentiment'
                    )
                    
                    # Add labels and legend
                    ax.set_xlabel('Movie Title')
                    ax.set_ylabel('Rating', color='skyblue')
                    ax2.set_ylabel('Sentiment Score', color='orange')
                    ax.set_title(f'Rating and Sentiment Scores for Top {len(recommendations)} Movies')
                    
                    # Rotate x-axis labels
                    plt.xticks(rotation=45, ha='right')
                    
                    # Add both legends
                    ax.legend(loc='upper left')
                    ax2.legend(loc='upper right')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
    
    else:
        # Show overview when no input is provided
        st.subheader("How it works")
        st.markdown("""
        This recommendation system suggests movies based on:
        
        1. **Genres** - Enter a specific genre like "Comedy", "Action", or "Romance"
        2. **Emotions** - Or enter how you want to feel, like "Happy", "Excited", or "Surprised"
        
        The system analyzes movie ratings and sentiment scores from reviews to find the perfect match for your mood.
        
        **Try it now!** Enter a genre or emotion in the sidebar.
        """)
        
        # Show sample visualization of emotions and genres
        st.subheader("Emotions and their common genres")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for visualization
        emotions = []
        top_genres = []
        
        for emotion, genres in emotion_to_genres.items():
            emotions.append(emotion)
            top_genre = list(genres.keys())[0] if genres else "Unknown"
            top_genres.append(top_genre)
        
        # Create visualization
        bars = ax.bar(emotions, [1] * len(emotions), alpha=0.7)
        
        # Add genre labels to bars
        for i, bar in enumerate(bars):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                0.5,
                top_genres[i],
                ha='center',
                va='center',
                rotation=0,
                color='black',
                fontweight='bold'
            )
        
        # Remove y-axis
        ax.set_yticks([])
        ax.set_ylabel('')
        
        # Set title
        ax.set_title('Top Genre for Each Emotion')
        
        plt.tight_layout()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
