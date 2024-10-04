import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Function to load and preprocess the dataset
@st.cache_resource
def load_and_train_model():
    # Load dataset
    file_path = 'training data.csv'  # Update with your correct file path
    df = pd.read_csv(file_path, encoding='ISO-8859-1')

    # Rename columns for clarity
    df.columns = ['ID', 'Topic', 'Sentiment', 'Text']

    # Drop unnecessary columns
    df = df[['Sentiment', 'Text']]

    # Clean the data by dropping rows with missing values
    df.dropna(subset=['Sentiment', 'Text'], inplace=True)

    # Encode sentiment labels: Positive -> 1, Neutral -> 0, Negative -> -1
    df['Sentiment'] = df['Sentiment'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})

    # Check for any remaining NaN values in the Sentiment column after mapping
    df.dropna(subset=['Sentiment'], inplace=True)

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Sentiment'], test_size=0.2, random_state=42)

    # Convert text data into numerical form using TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    return model, vectorizer

# Function to predict sentiment of a single tweet
def predict_sentiment(tweet, model, vectorizer):
    tweet_vec = vectorizer.transform([tweet])  # Transform the input tweet using the trained vectorizer
    prediction = model.predict(tweet_vec)      # Predict the sentiment
    sentiment = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}[prediction[0]]  # Map prediction to sentiment label
    return sentiment

# Streamlit app
def main():
    st.title("Text - Emotional Analysis")
    st.write("Enter a text and the model will predict its sentiment (Positive, Neutral, Negative).")

    # Load the trained model and vectorizer
    model, vectorizer = load_and_train_model()

    # User input for the tweet
    user_input = st.text_area("Enter a text for sentiment prediction:")

    # If a tweet is entered, predict the sentiment
    if st.button("Predict Sentiment"):
        if user_input:
            sentiment = predict_sentiment(user_input, model, vectorizer)
            st.success(f"Predicted Sentiment: {sentiment}")
        else:
            st.warning("Please enter a text..")

# Run the app
if __name__ == '__main__':
    main()
