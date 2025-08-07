from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContextAwareRecommender:
    def __init__(self):
        # Initialize the TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer()
        self.context_vectorized = None
        self.context_data = None

    def train(self, context_data):
        # Verify the presence of required columns
        if 'context' not in context_data.columns:
            raise ValueError("The dataset must contain a 'context' column.")
        
        # Store the context data for recommendations
        self.context_data = context_data

        # Fit and transform the context column
        self.context_vectorized = self.vectorizer.fit_transform(context_data['context'])
        print("TF-IDF Vectorizer successfully fitted.")  # Debugging log

    def recommend(self, user_context):
        # Ensure the vectorizer is trained
        if self.context_vectorized is None:
            raise ValueError("The TF-IDF vectorizer is not fitted. Please train the model first.")

        # Transform the user context to vectorized form
        user_context_vectorized = self.vectorizer.transform([user_context])
        
        # Compute similarity scores between user context and all dataset contexts
        scores = cosine_similarity(user_context_vectorized, self.context_vectorized).flatten()
        
        # Select the top 10 recommendations
        recommendations = self.context_data.iloc[scores.argsort()[-10:][::-1]]

        # Return relevant columns as dictionary (adjusted column names based on dataset)
        return recommendations[['name', 'artist', 'img', 'preview', 'spotify_id']].to_dict(orient='records')
