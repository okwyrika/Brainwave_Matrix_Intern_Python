import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import nltk # Ensure nltk is imported for download if needed
import string

# --- Configuration ---
# Update to your new dataset file
DATASET_PATH = 'healthcare_tweets.csv'
ENCODING = 'utf-8' # This custom CSV uses standard UTF-8 encoding

# --- Ensure NLTK VADER lexicon is downloaded ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    print("Downloading NLTK VADER lexicon...")
    nltk.download('vader_lexicon')
    print("NLTK VADER lexicon downloaded.")

# --- Step 1: Load the Dataset ---
print("Loading healthcare dataset...")
# This dataset has a header, so we don't need to define names
df = pd.read_csv(DATASET_PATH, encoding=ENCODING)
print(f"Dataset loaded. Shape: {df.shape}")
print(df.head())

# --- Step 2: Data Preprocessing ---
print("\nStarting data preprocessing...")

# We expect 'text' and 'sentiment' columns
# Make sure the target column is consistent, here it's 'sentiment'
# If you have a different column name for sentiment, update it here.
# For simplicity, we'll ensure our 'sentiment' column uses 'Positive', 'Negative', 'Neutral'
# The provided CSV uses 'Positive' and 'Negative'.

# Text Cleaning Function
def preprocess_text(text):
    text = str(text).lower()  # Convert to lowercase and ensure it's a string
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # Remove @mentions
    text = re.sub(r'#', '', text)  # Remove '#' symbol
    text = re.sub(r'RT[\s]+', '', text)  # Remove RT (retweet)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  # Remove punctuation
    text = re.sub(r'\n', '', text) # Remove newlines
    text = re.sub(r'\w*\d\w*', '', text) # Remove words containing numbers
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    return text

# Apply preprocessing to the 'text' column
df['cleaned_text'] = df['text'].apply(preprocess_text)
print("Text cleaning applied. Sample original vs cleaned:")
print(df[['text', 'cleaned_text']].head())

# --- Step 3: Sentiment Analysis using VADER ---
print("\nPerforming sentiment analysis with VADER...")
sid = SentimentIntensityAnalyzer()

# Function to get VADER sentiment scores
def get_vader_sentiment(text):
    vs = sid.polarity_scores(text)
    # Compound score ranges from -1 (most negative) to +1 (most positive)
    if vs['compound'] >= 0.05:
        return 'Positive'
    elif vs['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral' # VADER can classify neutral

# Apply VADER sentiment analysis
df['vader_sentiment'] = df['cleaned_text'].apply(get_vader_sentiment)
print("VADER sentiment analysis complete. Sample:")
print(df[['text', 'sentiment', 'vader_sentiment']].head()) # Note: 'sentiment' is original
print("\nVADER Sentiment Distribution:")
print(df['vader_sentiment'].value_counts())

# --- Step 4: Evaluation and Visualization ---
print("\nEvaluating results and generating visualizations...")

# 1. Overall VADER Sentiment Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='vader_sentiment', data=df, palette='viridis', order=['Positive', 'Neutral', 'Negative'])
plt.title('Distribution of Sentiment (VADER) in Healthcare Tweets')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.savefig('vader_sentiment_distribution.png')
plt.show()

# 2. Comparison: Original Sentiment vs. VADER Predicted Sentiment
plt.figure(figsize=(10, 7))
sns.histplot(data=df, x='sentiment', hue='vader_sentiment', multiple='stack', palette='tab10', shrink=0.8)
plt.title('Original Sentiment vs. VADER Predicted Sentiment')
plt.xlabel('Original Sentiment')
plt.ylabel('Number of Tweets')
plt.legend(title='VADER Prediction', loc='upper right', labels=['Positive', 'Neutral', 'Negative'])
plt.savefig('original_vs_vader_comparison.png')
plt.show()


# 3. Confusion Matrix (Comparing VADER vs Original Labels)
# For the confusion matrix, we'll compare VADER's 'Positive' and 'Negative'
# predictions against the original 'Positive' and 'Negative' labels.
# VADER's 'Neutral' predictions will be treated as false predictions relative to original labels.

# Ensure both original and predicted labels only contain 'Positive' and 'Negative' for direct comparison
# For tweets where VADER predicted 'Neutral', we need to decide how to handle them in CM.
# A common approach for direct comparison: if original is P/N and VADER is N, it's a mismatch.
# For simplicity, let's filter the data to only include original 'Positive'/'Negative' and
# VADER's 'Positive'/'Negative' for the *direct* classification report.
# Tweets classified as 'Neutral' by VADER will not appear in the confusion matrix.
comparison_df = df[df['sentiment'].isin(['Positive', 'Negative'])] # Filter original
comparison_df = comparison_df[comparison_df['vader_sentiment'].isin(['Positive', 'Negative'])] # Filter VADER predictions

if not comparison_df.empty:
    le = LabelEncoder()
    # Fit on all possible labels to ensure consistency
    le.fit(['Positive', 'Negative']) # Ensures 'Positive' is encoded consistently

    true_labels_encoded = le.transform(comparison_df['sentiment'])
    predicted_labels_encoded = le.transform(comparison_df['vader_sentiment'])

    cm = confusion_matrix(true_labels_encoded, predicted_labels_encoded)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix: Original Sentiment vs VADER Prediction')
    plt.xlabel('VADER Predicted Sentiment')
    plt.ylabel('Original Sentiment')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # 4. Classification Report
    print("\nClassification Report (Original vs VADER Sentiment):")
    print(classification_report(true_labels_encoded, predicted_labels_encoded, target_names=le.classes_))
else:
    print("\nNot enough data for confusion matrix after filtering (original vs VADER 'Positive'/'Negative' only).")


print("\nAnalysis Complete!")