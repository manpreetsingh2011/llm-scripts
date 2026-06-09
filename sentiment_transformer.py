import pandas as pd
import numpy as np
import re
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# 1. GLOBAL CONFIG
# ---------------------------------------------------------------------------

MODEL_NAME = 'distilbert-base-uncased'
MAX_LENGTH = 128
BATCH_SIZE = 64
EPOCHS = 2
LEARNING_RATE = 2e-5

LR_TEST_SIZE = 0.2
LR_SPLIT_SEED = 42

DEVICE = torch.device(
    'mps' if torch.backends.mps.is_available()
    else ('cuda' if torch.cuda.is_available() else 'cpu')
)

# ---------------------------------------------------------------------------
# 2. TEXT CLEANING (lighter — transformer tokenizers handle casing/punctuation)
# ---------------------------------------------------------------------------

def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------------------------------------------------------------
# 3. DATA LOADING
# ---------------------------------------------------------------------------

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
    df['sentiment_bin'] = (df['sentiment'] == 'positive').astype(int)
    return df

# ---------------------------------------------------------------------------
# 4. PREPROCESSING
# ---------------------------------------------------------------------------

def preprocess(df):
    print("Cleaning text...")
    df['cleaned'] = df['review'].apply(clean_text)
    before = df.shape[0]
    df = df.drop_duplicates(subset=['cleaned']).reset_index(drop=True)
    after = df.shape[0]
    if before != after:
        print(f"Dropped {before - after} duplicate(s) after cleaning ({before} -> {after})")
    return df

# ---------------------------------------------------------------------------
# 5. PYTORCH DATASET
# ---------------------------------------------------------------------------

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding=True, max_length=max_length, return_tensors='pt'
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# ---------------------------------------------------------------------------
# 6. METRICS
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, preds)}

# ---------------------------------------------------------------------------
# 7. TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------

def split_data(df):
    return train_test_split(
        df['cleaned'], df['sentiment_bin'],
        test_size=LR_TEST_SIZE,
        random_state=LR_SPLIT_SEED,
        stratify=df['sentiment_bin'],
    )

# ---------------------------------------------------------------------------
# 8. MODEL TRAINING (fine-tune transformer)
# ---------------------------------------------------------------------------

def train_model(train_texts, train_labels, val_texts, val_labels, model_name=MODEL_NAME):
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(DEVICE)

    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    training_args = TrainingArguments(
        output_dir='./checkpoints',
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        learning_rate=LEARNING_RATE,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        logging_dir='./logs',
        logging_steps=50,
        report_to='none',
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Training transformer model...")
    trainer.train()
    return trainer, tokenizer

# ---------------------------------------------------------------------------
# 9. EVALUATION
# ---------------------------------------------------------------------------

def evaluate_model(trainer, tokenizer, val_texts, val_labels):
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    predictions = trainer.predict(val_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(
        f"\nClassification Report:\n"
        f"{classification_report(y_true, y_pred, target_names=['negative', 'positive'])}"
    )
    print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")

# ---------------------------------------------------------------------------
# 10. SAVE / LOAD
# ---------------------------------------------------------------------------

def save_models(trainer, tokenizer, model_path='sentiment_model'):
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model saved to: {model_path}/")

def load_models(model_path='sentiment_model'):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded from: {model_path}/")
    return model, tokenizer

# ---------------------------------------------------------------------------
# 11. SINGLE-REVIEW PREDICTION
# ---------------------------------------------------------------------------

def predict_review(text, model, tokenizer):
    cleaned = clean_text(text)
    if not cleaned.strip():
        return {'label': 'negative', 'confidence': 0.0, 'cleaned': cleaned}

    inputs = tokenizer(cleaned, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors='pt')
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(logits, dim=-1).item()

    return {
        'label': 'positive' if pred == 1 else 'negative',
        'confidence': probs[0, pred].item(),
        'cleaned': cleaned,
    }

# ---------------------------------------------------------------------------
# 12. MAIN PIPELINE
# ---------------------------------------------------------------------------

def main():
    df = load_data('IMDB Dataset.csv')
    df = preprocess(df)
    print(f"\nPreprocessing done. {len(df)} reviews ready.")

    train_texts, val_texts, train_labels, val_labels = split_data(df)
    print(f"Training set: {len(train_texts)} samples")
    print(f"Test set: {len(val_texts)} samples\n")

    trainer, tokenizer = train_model(train_texts, train_labels, val_texts, val_labels)
    evaluate_model(trainer, tokenizer, val_texts, val_labels)
    save_models(trainer, tokenizer)


if __name__ == '__main__':
    main()

    print("\n" + "=" * 50)
    print("DEMO: Single-review predictions")
    print("=" * 50)

    model, tokenizer = load_models()

    examples = [
        "This movie was absolutely fantastic! The acting was brilliant.",
        "Terrible film. Complete waste of time. Worst movie ever.",
        "It was okay, not great but not terrible either. Decent for one watch.",
        "i am not going to watch this movie again.",
        "not my cup of tea.",
        "acting was decent.",
    ]

    for text in examples:
        result = predict_review(text, model, tokenizer)
        print(f"\nRaw:       {text}")
        print(f"Cleaned:   {result['cleaned']}")
        print(f"Sentiment: {result['label']} ({result['confidence']:.1%} confidence)")
