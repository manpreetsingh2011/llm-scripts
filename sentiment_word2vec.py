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

# ---------------------------------------------------------------------------
# 1. DOWNLOAD NLTK RESOURCES (run once)
# ---------------------------------------------------------------------------
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# ---------------------------------------------------------------------------
# 2. GLOBAL CONFIG
# ---------------------------------------------------------------------------
STOPWORDS = set(stopwords.words('english'))

# Words that flip the sentiment of subsequent tokens.
# When one of these is encountered, the next tokens get a "neg_" prefix
# until a sentence boundary (. ! ? ; :) or a contrastive conjunction resets it.
NEGATION_WORDS = {
    'not', 'no', 'never', 'nor', 'neither', 'nowhere',
    'dont', 'doesnt', 'didnt', 'wont', 'wouldnt', 'couldnt',
    'shouldnt', 'isnt', 'arent', 'wasnt', 'werent', 'havent',
    'hasnt', 'hadnt', 'cant', 'cannot',
}

# Contrastive conjunctions that end the scope of a negation.
# e.g. "not good but great" → "but" resets so "great" is NOT tagged as neg_.
CONTRASTIVE_CONJUNCTIONS = {
    'but', 'however', 'nevertheless', 'although', 'though',
}

# Word2Vec hyper-parameters (tuned for best accuracy on this dataset).
WORD2VEC_VECTOR_SIZE = 300
WORD2VEC_WINDOW = 3
WORD2VEC_MIN_COUNT = 5
WORD2VEC_WORKERS = 4
WORD2VEC_EPOCHS = 10
WORD2VEC_SEED = 42

# Logistic Regression hyper-parameters.
LR_MAX_ITER = 1000
LR_RANDOM_STATE = 42
LR_TEST_SIZE = 0.2
LR_SPLIT_SEED = 42

# ---------------------------------------------------------------------------
# 3. TEXT CLEANING
# ---------------------------------------------------------------------------
def clean_text(text):
    """Normalise raw review text before tokenization.

    Steps (in order):
        1. Strip HTML tags (<br/>, <p>, …)
        2. Remove URLs (http://…, www.…)
        3. Keep only letters, whitespace, and sentence-boundary punctuation (. ! ? ; :)
        4. Collapse multiple whitespace characters into one
        5. Trim and lowercase
    """
    text = re.sub(r'<[^>]+>', ' ', text)          # strip HTML tags
    text = re.sub(r'http\S+|www\S+', ' ', text)   # strip URLs
    text = re.sub(r'[^a-zA-Z\s.!?;:]', ' ', text) # keep only alpha + boundary punctuation
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

# ---------------------------------------------------------------------------
# 4. TOKENISATION WITH NEGATION HANDLING
# ---------------------------------------------------------------------------
def tokenize(text):
    """Tokenise a single cleaned review into raw NLTK tokens.

    This is a pure tokenisation step — no filtering, no negation handling.
    """
    return word_tokenize(text)


def apply_negation_scope(tokens):
    """Walk through raw tokens and apply negation-scope logic.

    For each token:
        - Sentence-boundary character (. ! ? ; :) → reset negation.
        - Contrastive conjunction (but, however, …)  → reset negation.
        - If negation is active → prepend "neg_".
        - If token is a negation word → activate negation and *skip*.
        - If token is a stopword or ≤ 2 characters → skip.
        - Otherwise → keep.

    Returns a filtered list of (possibly prefixed) tokens.
    """
    result = []
    negate = False

    for word in tokens:
        if word in '.!?;:':
            negate = False
            continue

        if word in CONTRASTIVE_CONJUNCTIONS:
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

# ---------------------------------------------------------------------------
# 5. DATA LOADING
# ---------------------------------------------------------------------------
def load_data(filepath):
    """Load CSV and add a binary sentiment column (1 = positive, 0 = negative)."""
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
    df['sentiment_bin'] = (df['sentiment'] == 'positive').astype(int)
    return df

# ---------------------------------------------------------------------------
# 6. PREPROCESSING PIPELINE (clean → dedup → tokenise)
# ---------------------------------------------------------------------------
def preprocess(df):
    """Apply cleaning, deduplication, and tokenisation to the DataFrame.

    Duplicates are removed *after* cleaning (so slight formatting differences
    like "Great movie!!" vs "great movie" are caught), but *before* tokenisation
    (to avoid wasting CPU cycles on duplicate work).
    """
    print("Cleaning text...")
    df['cleaned'] = df['review'].apply(clean_text)

    # Deduplicate on normalised (cleaned) text.
    before = df.shape[0]
    df = df.drop_duplicates(subset=['cleaned']).reset_index(drop=True)
    after = df.shape[0]
    if before != after:
        print(f"Dropped {before - after} duplicate(s) after cleaning ({before} -> {after})")

    print("Tokenizing text...")
    df['raw_tokens'] = df['cleaned'].apply(tokenize)
    df['tokens'] = df['raw_tokens'].apply(apply_negation_scope)

    # Temporary columns no longer needed.
    df.drop(columns=['cleaned', 'raw_tokens'], inplace=True)
    return df

# ---------------------------------------------------------------------------
# 7. WORD2VEC TRAINING
# ---------------------------------------------------------------------------
def train_word2vec(sentences):
    """Train a Word2Vec skip-gram model on the tokenised reviews.

    Returns a KeyedVectors object (faster, lighter than the full model).
    """
    print(
        f"Training Word2Vec "
        f"(size={WORD2VEC_VECTOR_SIZE}, "
        f"window={WORD2VEC_WINDOW}, "
        f"min_count={WORD2VEC_MIN_COUNT})..."
    )
    model = Word2Vec(
        sentences=sentences,
        vector_size=WORD2VEC_VECTOR_SIZE,
        window=WORD2VEC_WINDOW,
        min_count=WORD2VEC_MIN_COUNT,
        workers=WORD2VEC_WORKERS,
        epochs=WORD2VEC_EPOCHS,
        seed=WORD2VEC_SEED,
    )
    return model.wv

# ---------------------------------------------------------------------------
# 8. REVIEW VECTORISATION (mean pooling of word vectors)
# ---------------------------------------------------------------------------
def vectorize_reviews(tokens_list, w2v):
    """Convert each review (list of tokens) into a single 300-D vector.

    The vector is the element-wise mean of all word vectors in the review.
    Reviews with no recognised words get a zero vector.
    """
    print("Vectorizing reviews...")

    def vectorize(tokens):
        vectors = [w2v[word] for word in tokens if word in w2v]
        if not vectors:
            return np.zeros(w2v.vector_size)
        return np.mean(vectors, axis=0)

    return np.array([vectorize(tokens) for tokens in tokens_list])

# ---------------------------------------------------------------------------
# 9. TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------
def split_data(X, y):
    """Stratified 80/20 train/test split."""
    return train_test_split(
        X, y,
        test_size=LR_TEST_SIZE,
        random_state=LR_SPLIT_SEED,
        stratify=y,
    )

# ---------------------------------------------------------------------------
# 10. CLASSIFIER TRAINING
# ---------------------------------------------------------------------------
def train_model(X_train, y_train):
    """Train a Logistic Regression classifier on the vectorised reviews."""
    print("Training Logistic Regression classifier...")
    model = LogisticRegression(max_iter=LR_MAX_ITER, random_state=LR_RANDOM_STATE)
    model.fit(X_train, y_train)
    return model

# ---------------------------------------------------------------------------
# 11. EVALUATION
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test):
    """Print accuracy, classification report, and confusion matrix."""
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(
        f"\nClassification Report:\n"
        f"{classification_report(y_test, y_pred, target_names=['negative', 'positive'])}"
    )
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# ---------------------------------------------------------------------------
# 12. SAVE MODELS TO DISK
# ---------------------------------------------------------------------------
def save_models(model, w2v, model_path='sentiment_model.pkl', w2v_path='w2v_model.pkl'):
    """Persist the trained classifier and word vectors to disk via joblib."""
    joblib.dump(model, model_path)
    joblib.dump(w2v, w2v_path)
    print(f"Models saved: {model_path}, {w2v_path}")

# ---------------------------------------------------------------------------
# 13. MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    """Orchestrate the full workflow:

    1. Load CSV         →  load_data()
    2. Clean / dedup / tokenise  →  preprocess()
    3. Train Word2Vec   →  train_word2vec()
    4. Mean-pool vectors → vectorize_reviews()
    5. Train/test split  →  split_data()
    6. Train classifier  →  train_model()
    7. Evaluate          →  evaluate_model()
    8. Save to disk      →  save_models()
    """
    df = load_data('IMDB Dataset.csv')
    df = preprocess(df)

    print(f"\nPreprocessing done. {len(df)} reviews ready.")
    print(df['tokens'])

    # ---------------------------------------------------------------
    # Train Word2Vec embeddings on our tokenised reviews.
    # Word2Vec learns dense, 300-D vectors for each word based on
    # its surrounding context (skip-gram with window=3).
    # ---------------------------------------------------------------
    w2v = train_word2vec(df['tokens'])
    print(f"Vocabulary size: {len(w2v)} words\n")

    # ---------------------------------------------------------------
    # Convert each review into a single 300-D vector by averaging
    # the Word2Vec vectors of all its tokens.
    # ---------------------------------------------------------------
    X = vectorize_reviews(df['tokens'], w2v)
    y = df['sentiment_bin'].values

    # Stratified 80/20 split.
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples\n")

    # Train and evaluate.
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Persist.
    save_models(model, w2v)


if __name__ == '__main__':
    main()
