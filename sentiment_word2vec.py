import pandas as pd
import numpy as np
import re
import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words('english'))

NEGATION_WORDS = {
    'not', 'no', 'never', 'nor', 'neither', 'nowhere',
    'dont', 'doesnt', 'didnt', 'wont', 'wouldnt', 'couldnt',
    'shouldnt', 'isnt', 'arent', 'wasnt', 'werent', 'havent',
    'hasnt', 'hadnt', 'cant', 'cannot',
}

def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s.!?;:]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def tokenize(text):
    tokens = word_tokenize(text)
    result = []
    negate = False
    for word in tokens:
        if word in '.!?;:':
            negate = False
            continue
        if negate:
            word = 'neg_' + word
        if word in NEGATION_WORDS:
            negate = True
            continue
        if word in STOPWORDS or len(word) <= 2:
            continue
        result.append(word)
    return result

def load_data(filepath):
    df = pd.read_csv(filepath)
    #df = df[:1000]
    print(f"Dataset shape: {df.shape}")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
    df['sentiment_bin'] = (df['sentiment'] == 'positive').astype(int)
    return df

def preprocess(df):
    print("Cleaning text...")
    df['cleaned'] = df['review'].apply(clean_text)
    before = df.shape[0]
    df = df.drop_duplicates(subset=['cleaned']).reset_index(drop=True)
    after = df.shape[0]
    if before != after:
        print(f"Dropped {before - after} duplicate(s) after cleaning ({before} -> {after})")
    print("Tokenizing text...")
    df['tokens'] = df['cleaned'].apply(tokenize)
    df.drop(columns=['cleaned'], inplace=True)
    return df

def train_word2vec(sentences, vector_size=300, window=2, min_count=3, workers=4, epochs=10):
    print(f"Training custom Word2Vec (size={vector_size}, window={window}, min_count={min_count})...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        seed=42,
    )
    return model.wv

def vectorize_reviews(tokens_list, w2v):
    print("Vectorizing reviews...")

    def vectorize(tokens):
        vectors = [w2v[word] for word in tokens if word in w2v]
        if not vectors:
            return np.zeros(w2v.vector_size)
        return np.mean(vectors, axis=0)

    return np.array([vectorize(tokens) for tokens in tokens_list])

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def train_model(X_train, y_train):
    print("Training Logistic Regression classifier...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['negative', 'positive'])}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

def save_models(model, w2v, model_path='sentiment_model.pkl', w2v_path='w2v_model.pkl'):
    joblib.dump(model, model_path)
    joblib.dump(w2v, w2v_path)
    print(f"Models saved: {model_path}, {w2v_path}")

def main():
    df = load_data('IMDB Dataset.csv')
    df = preprocess(df)

    print('preprocess completed');
    print(df['tokens']);

    w2v = train_word2vec(df['tokens'], window=2, min_count=3)

    print(w2v);

    X = vectorize_reviews(df['tokens'], w2v)
    y = df['sentiment_bin'].values

    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_models(model, w2v)

if __name__ == '__main__':
    main()
