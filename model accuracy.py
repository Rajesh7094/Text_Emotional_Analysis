import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
file_path = "C:\\Users\\Welcome\\Desktop\\Text-Emotional Analysis\\training data.csv"  # Update with your correct file path
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Rename columns for clarity
df.columns = ['ID', 'Topic', 'Sentiment', 'Text']

# Drop unnecessary columns
df = df[['Sentiment', 'Text']]

# Clean the data by dropping rows with missing values
df.dropna(subset=['Sentiment', 'Text'], inplace=True)  # This removes rows where 'Sentiment' or 'Text' is NaN

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

# Make predictions on the test data
y_pred = model.predict(X_test_vec)

# Print classification report
print(classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))
