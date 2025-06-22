import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re
import nltk
from nltk.corpus import stopwords
import os # Added for NLTK data path handling

# Configure page with professional color scheme
st.set_page_config(
    page_title="Zomato AI Recommender",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for consistent, accessible colors
st.markdown("""
<style>
    /* Define color palette */
    :root {
        --primary: #FF4B4B;  /* Zomato's brand red */
        --secondary: #E03E3E; /* Slightly darker red for accents and button hovers */
        --background-main: #FDFCEF; /* Main app background - very light beige */
        --background-sidebar: #F5F5DC; /* Sidebar background - light cream */
        --background-input-box: #FAF5E6; /* Input field/dataframe cell background - beige */
        --card: #FF4B4B;     /* Restaurant card background - Zomato red */
        --text-dark: #1A1A1A; /* Very dark grey/almost black for most text */
        --text-light: #FFFFFF; /* White text, for elements on dark backgrounds */
        --light-text: #4A4A4A; /* Medium grey for secondary details (e.g., votes count) */
        --border-light: #CCCCCC;   /* Light grey for borders, visible */
    }
    
    /* Overall App Styling */
    .stApp {
        background-color: var(--background-main); /* Apply main beige background */
        color: var(--text-dark); /* Default text color for elements not explicitly styled */
    }
    
    /* Headers - all levels */
    h1, h2, h3, h4 {
        color: var(--primary) !important; /* Headers remain Zomato red for prominence */
    }
    
    /* Sidebar styling */
    .st-emotion-cache-1wqx_jl { /* Targets the outer sidebar container */
        background-color: var(--background-sidebar); /* Light cream sidebar background */
    }
    .st-emotion-cache-1wqx_jl .st-emotion-cache-16txtg3 { /* Targets the inner sidebar content div */
        color: var(--text-dark); /* Ensure sidebar general text is dark and visible */
    }
    /* FIX: Labels for sliders and selectbox in sidebar to appear white */
    .stSidebar .stSlider > label,
    .stSidebar .stSelectbox > label {
        color: var(--text-light) !important; /* White text for sidebar labels */
    }

    /* Sliders styling */
    .stSlider > div > div > div:nth-child(2) { /* Slider track (unfilled part) */
        background: var(--secondary) !important;
    }
    .stSlider > div > div > div:nth-child(3) { /* Slider fill (filled part) */
        background: var(--primary) !important;
    }
    .stSlider > div > div > div:nth-child(4) { /* Slider thumb (draggable circle) */
        background: var(--primary) !important;
        border-color: var(--primary) !important;
    }
    
    /* Select boxes styling */
    /* Text inside the dropdown when an item is selected */
    .stSelectbox > div > div > div {
        color: var(--text-dark) !important; /* Dark text for selected item */
        background-color: var(--background-input-box); /* Beige background for selected item display */
    }
    /* The dropdown button/border */
    .stSelectbox > div > div {
        border-color: var(--border-light) !important; /* Light grey border, now visible */
        border-radius: 8px; /* Rounded corners */
        background-color: var(--background-input-box); /* Beige background for the selectbox itself */
    }

    /* Buttons styling */
    .stButton > button {
        background-color: var(--primary); /* Primary red background */
        color: var(--text-light); /* White text */
        border-radius: 8px; /* Rounded corners */
        padding: 0.5rem 1rem;
        border: none;
        transition: background-color 0.2s; /* Smooth transition on hover */
    }
    .stButton > button:hover {
        background-color: var(--secondary); /* Slightly darker red on hover */
        color: var(--text-light); /* Ensure text remains white on hover for good contrast */
    }
    
    /* Metrics styling (e.g., Total Recommendations Found) */
    .stMetric > div > div { /* Metric value (the large number) */
        color: var(--primary) !important; /* Primary red color */
        font-weight: bold;
    }
    .stMetric label { /* Metric label text (e.g., "Total Recommendations Found") */
        color: var(--text-dark) !important; /* Dark text for metric labels */
    }

    /* Restaurant cards styling */
    .restaurant-card {
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 12px;
        background: var(--card); /* Zomato red background for cards */
        box-shadow: 0 4px 12px rgba(0,0,0,0.08); /* Soft shadow */
        border-left: 4px solid var(--primary); /* Left border in primary red */
        transition: transform 0.2s, box-shadow 0.2s; /* Smooth transitions */
    }
    .restaurant-card:hover {
        transform: translateY(-3px); /* Lift effect on hover */
        box-shadow: 0 6px 16px rgba(0,0,0,0.12); /* Slightly stronger shadow on hover */
    }
    .restaurant-card h4 {
        color: var(--text-light) !important; /* White text for restaurant name */
        font-size: 1.4rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .restaurant-card p {
        color: var(--text-light) !important; /* White text for paragraph content (cuisines, rating details) */
        margin: 0.3rem 0;
        font-size: 1rem;
    }
    .restaurant-card .rating {
        color: var(--text-light) !important; /* White text for rating stars */
        font-weight: bold;
    }
    .restaurant-card .rating span { /* For the votes part of the rating (e.g., "(1234 votes)") */
        color: var(--text-light) !important; /* White text for votes count */
        font-weight: normal;
    }
    
    /* Dataframes styling */
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    /* Targeting cell content and header labels for dark text */
    .stDataFrame .ag-root-wrapper-body .ag-row-level-0 .ag-cell, 
    .stDataFrame .ag-root-wrapper-body .ag-row-level-0 .ag-header-cell-label {
        color: var(--text-dark) !important; /* Dark text for all dataframe content */
    }
    /* Ensure header text of dataframe is readable */
    .stDataFrame thead th {
        background-color: var(--background-input-box); /* Beige background for header */
        color: var(--primary) !important; /* Make header text primary red */
    }
    /* Zebra striping for table rows */
    .stDataFrame tbody tr {
        background-color: var(--background-input-box); /* Beige background for odd rows */
    }
    .stDataFrame tbody tr:nth-child(even) {
        background-color: var(--background-main); /* Main app background for even rows */
    }

    /* Tabs styling */
    /* Target the text color for the tabs themselves (unselected and selected) */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        color: var(--text-dark) !important; /* Ensure unselected tab text is dark grey and always visible */
        font-weight: bold; /* Make tab text bold for prominence */
    }
    /* Styling for the currently selected tab */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] [data-testid="stMarkdownContainer"] p {
        color: var(--primary) !important; /* Selected tab text in primary red */
    }

    /* FIX: Ensure `st.info` messages and other general instruction text are black */
    div[data-testid="stAlert"] { /* Targets the alert container for st.info */
        background-color: var(--background-input-box) !important; /* Beige background for alerts */
        color: var(--text-dark) !important; /* Dark text for alert messages */
        border-color: var(--border-light) !important;
    }
    div[data-testid="stAlert"] div[data-testid="stMarkdownContainer"] p {
        color: var(--text-dark) !important; /* Ensure text inside the alert is dark */
    }
    
    /* FIX: General Streamlit Markdown text color (for descriptions like "Select a restaurant you like...") */
    div.stMarkdown p {
        color: var(--text-dark) !important; /* Ensure all general markdown text is dark */
    }

</style>
""", unsafe_allow_html=True)

# Cuisine to emoji mapping (expanded for better coverage)
CUISINE_EMOJIS = {
    'american': '🍔', 'bbq': '🍖', 'italian': '🍝', 'indian': '🍛', 'chinese': '�',
    'japanese': '🍣', 'mexican': '🌮', 'thai': '🍜', 'mediterranean': '🥙', 'french': '🥐',
    'desserts': '🍰', 'cafe': '☕', 'bakery': '🥖', 'pizza': '🍕', 'burger': '🍔',
    'seafood': '🍤', 'sushi': '🍣', 'korean': '🍲', 'lebanese': '🧆', 'finger food': '🍟',
    'continental': '🍽️', 'fast food': '🍟', 'street food': '🍢', 'healthy food': '🥗',
    'drinks': '🍹', 'beverages': '🥤', 'north indian': '🥘', 'south indian': ' dosai',
    'biryani': '🍲', 'mumbai': ' vada pav', 'deli': '🥪', 'steak': '🥩', 'pub food': '🍺',
    'cocktails': '🍸', 'ice cream': '🍦', 'waffle': '🧇', 'tea': '🍵', 'coffee': '☕',
    'arabian': '🧆', 'asian': '🍚', 'breakfast': '🍳', 'continental': '🍽️', 'european': '🇫🇷',
    'grill': '🥩', 'kebab': '🍖', 'kerala': '🌶️', 'malaysian': '🍜', 'mughlai': '🍛',
    'pakistani': '🍛', 'rajasthani': '🌶️', 'sandwich': '🥪', 'sichuan': '🌶️', 'sweet': '🍬',
    'turkish': '🥙', 'vietnamese': '🍜', 'wraps': '🌯'
}

# Define a writable NLTK data path for cloud deployments
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

# Download NLTK data silently and only once (check if already downloaded)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)

@st.cache_data
def load_data():
    """
    Loads and preprocesses the Zomato dataset.
    Includes cleaning cuisine text, removing stopwords, and creating
    separate columns for clustering (pure text) and display (with emojis).
    """
    try:
        df = pd.read_csv('zomato.csv', encoding='latin-1')
        df = df[['Restaurant Name', 'Cuisines', 'Aggregate rating', 'Votes']].dropna()
        df.columns = ['name', 'cuisines', 'rating', 'votes']
        
        stop_words = set(stopwords.words('english'))
        
        # Function to clean text for clustering (removes non-alphabetic, lowercases, removes stopwords)
        def clean_for_clustering(text):
            text = re.sub(r'[^a-zA-Z,\s]', '', str(text).lower())
            words = [word.strip() for word in text.split(',') if word.strip() and word.strip() not in stop_words]
            return ', '.join(words)

        # Function to add emojis to original cuisine names for display
        def add_emojis_to_cuisines(cuisine_string):
            original_cuisines = [c.strip() for c in cuisine_string.split(',') if c.strip()]
            emoji_cuisines_list = []
            for oc in original_cuisines:
                best_match_emoji = '🍴' # Default emoji if no specific match
                # Iterate through emoji keys to find a match (case-insensitive substring)
                for key, emoji in CUISINE_EMOJIS.items():
                    if key in oc.lower(): 
                        best_match_emoji = emoji
                        break # Take the first matching emoji
                emoji_cuisines_list.append(f"{best_match_emoji} {oc}")
            return ', '.join(emoji_cuisines_list)
                
        df['cleaned_for_clustering'] = df['cuisines'].apply(clean_for_clustering)
        df['display_cuisines'] = df['cuisines'].apply(add_emojis_to_cuisines)

        return df
    except FileNotFoundError:
        st.error("Error: 'zomato.csv' not found. Please ensure the dataset file is in the same directory as the script.")
        return pd.DataFrame() # Return empty DataFrame on error
    except Exception as e:
        st.error(f"An unexpected error occurred while loading or processing data: {str(e)}")
        return pd.DataFrame() # Return empty DataFrame on any other error

@st.cache_data
def cluster_restaurants(df):
    """
    Performs TF-IDF vectorization and K-Means clustering on cleaned cuisine data.
    """
    if df.empty:
        return df # If DataFrame is empty, return it as is
    
    tfidf = TfidfVectorizer(max_features=1000)
    features = tfidf.fit_transform(df['cleaned_for_clustering']) # Use the text specifically cleaned for clustering
    
    # Use n_init for robust KMeans initialization (avoids warning in newer scikit-learn versions)
    kmeans = KMeans(n_clusters=15, random_state=42, n_init=10) 
    df['cluster'] = kmeans.fit_predict(features)
    return df

def main():
    st.title("🍽️ Zomato AI Restaurant Recommender")
    st.markdown("### Discover restaurants with similar cuisines using machine learning clustering")
    
    # Load and process data
    df = load_data()
    if df.empty:
        st.warning("Application cannot run without data. Please ensure 'zomato.csv' is correctly placed and accessible.")
        return # Stop execution if data loading failed
    
    df = cluster_restaurants(df)
    
    # Sidebar filters
    with st.sidebar:
        st.header("🔍 Filters")
        # FIX: Sidebar general text (like "Adjust your search criteria:") is now dark by default.
        st.markdown("Adjust your search criteria:")
        
        min_rating = st.slider(
            "⭐ Minimum Rating", # FIX: Label in sidebar is now white
            1.0, 5.0, 3.5, 0.1,
            help="Filter by minimum restaurant rating. Only restaurants with this rating or higher will be shown."
        )
        
        min_votes = st.slider(
            "🗳️ Minimum Votes", # FIX: Label in sidebar is now white
            0, 5000, 100, 50,
            help="Filter by minimum number of votes. Only restaurants with this many votes or more will be included."
        )
        
        # Get unique cuisines from the original 'cuisines' column for the selectbox
        all_cuisines = sorted(df['cuisines'].str.split(', ').explode().unique().tolist())
        selected_cuisine = st.selectbox(
            "🍽️ Filter by Cuisine", # FIX: Label in sidebar is now white
            ['All Cuisines'] + all_cuisines, # Add 'All Cuisines' option
            help="Select a specific cuisine type to narrow down your search."
        )
    
    # Apply filters to the DataFrame
    filtered_df = df[(df['rating'] >= min_rating) & (df['votes'] >= min_votes)]
    if selected_cuisine != 'All Cuisines':
        # Ensure 'cuisines' column contains the selected cuisine (case-insensitive)
        filtered_df = filtered_df[filtered_df['cuisines'].str.contains(selected_cuisine, case=False, na=False)]
    
    # Main content tabs
    tab1, tab2 = st.tabs(["🔍 Explore Restaurants", "🤖 Smart Recommendations"])
    
    with tab1:
        st.subheader("🍽️ Restaurant Explorer")
        # FIX: st.info messages now appear in black text on beige background
        if filtered_df.empty:
            st.info("No restaurants found with the selected filters. Please adjust your criteria in the sidebar.")
        else:
            # Display filtered restaurants in a DataFrame
            st.dataframe(
                filtered_df[['name', 'display_cuisines', 'rating', 'votes']].sort_values(['rating', 'votes'], ascending=False),
                height=500, # Set a fixed height for the dataframe
                use_container_width=True, # Make it responsive to container width
                hide_index=True, # Hide default pandas index for cleaner look
                column_config={ # Configure column display for better readability
                    "name": "Restaurant",
                    "display_cuisines": "Cuisines", # Use the emoji-enhanced cuisines for display
                    "rating": st.column_config.NumberColumn("Rating", format="%.1f ⭐"), # Format rating with a star
                    "votes": st.column_config.NumberColumn("Votes", format="🗳️ %d") # Format votes with an emoji
                }
            )
            
            # Separator for visualizations
            st.markdown("---") 
            
            # Display visualizations in two columns
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 Rating Distribution")
                fig1, ax1 = plt.subplots(figsize=(10, 4)) # Set explicit figure size
                # Use hex code for reliability in Matplotlib (maps to var(--primary))
                filtered_df['rating'].hist(bins=20, ax=ax1, color='#FF4B4B', edgecolor='white') 
                ax1.set_xlabel("Rating")
                ax1.set_ylabel("Number of Restaurants")
                plt.tight_layout() # Adjust layout to prevent labels overlapping
                st.pyplot(fig1)
            
            with col2:
                st.subheader("📈 Votes vs Rating")
                fig2, ax2 = plt.subplots(figsize=(10, 4)) # Set explicit figure size
                # Use hex code for reliability (maps to var(--secondary))
                ax2.scatter(filtered_df['rating'], filtered_df['votes'], alpha=0.7, color="#FF9A9A", # Changed color back to secondary for consistency
                            s=filtered_df['votes']/10 + 20) # Size points by votes for visual impact
                ax2.set_xlabel("Rating")
                ax2.set_ylabel("Votes")
                plt.tight_layout() # Adjust layout
                st.pyplot(fig2)
    
    with tab2:
        st.subheader("🔍 Find Similar Restaurants")
        # FIX: st.info messages now appear in black text on beige background
        if filtered_df.empty:
            st.info("No restaurants match your current filters. Please adjust the filters in the sidebar to enable recommendations.")
        else:
            # Selectbox for choosing a restaurant for recommendations
            # FIX: "Select a restaurant you like to find similar ones:" is now black
            restaurant_name = st.selectbox(
                "Select a restaurant you like to find similar ones:",
                filtered_df['name'].unique(), # Populate with unique restaurant names from filtered data
                key="restaurant_select" # Unique key for this widget
            )
            
            if restaurant_name: # Only proceed if a restaurant is selected
                # Check if the selected restaurant is still in the filtered_df (filters might have changed)
                if restaurant_name not in filtered_df['name'].values:
                    # FIX: st.warning messages now appear in black text on beige background
                    st.warning("The selected restaurant is no longer available with current filters. Please choose another.")
                else:
                    # Get the cluster of the selected restaurant
                    cluster = filtered_df[filtered_df['name'] == restaurant_name]['cluster'].values[0]
                    # Find other restaurants in the same cluster, excluding the selected one itself
                    similar_restaurants = filtered_df[
                        (filtered_df['cluster'] == cluster) & 
                        (filtered_df['name'] != restaurant_name)
                    ]
                    
                    if similar_restaurants.empty:
                        # FIX: st.info messages now appear in black text on beige background
                        st.info(f"No other similar restaurants found in the same cluster for '{restaurant_name}' with the current filters.")
                    else:
                        # Display total recommendations found
                        st.metric("🍽️ Total Recommendations Found", len(similar_restaurants))
                        
                        # Display top 10 recommendations in styled cards
                        st.subheader(f"✨ Top Picks Similar to {restaurant_name}:")
                        # Sort by rating and votes for the best recommendations
                        for _, row in similar_restaurants.sort_values(['rating', 'votes'], ascending=False).head(10).iterrows():
                            with st.container():
                                st.markdown(f"""
                                <div class="restaurant-card">
                                    <h4>{row['name']}</h4>
                                    <p><b>🥘 Cuisines:</b> {row['display_cuisines']}</p>
                                    <p class="rating">⭐ {row['rating']:.1f} <span style="font-weight: normal;">(🗳️ {row['votes']} votes)</span></p>
                                </div>
                                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
