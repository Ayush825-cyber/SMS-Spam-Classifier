import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

from text_utils import preprocess  # shared with app.py so the saved model can be loaded

# 1. Load dataset
df = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8",
)

# 2. Remove any missing messages
df = df.dropna(subset=["message", "label"])

# 3. Separate input and output
X = df["message"]
y = df["label"]

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. NLP pipeline: TF-IDF (unigrams + bigrams) + class-balanced Logistic Regression
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        preprocessor=preprocess,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )),
    ("classifier", LogisticRegression(
        C=10,
        class_weight="balanced",
        max_iter=2000,
    )),
])

# 6. Train
print("Training model...")
model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Save
joblib.dump(model, "spam_model.pkl")
print("\nModel saved successfully as spam_model.pkl!")
