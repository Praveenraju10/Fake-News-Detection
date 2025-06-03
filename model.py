import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load datasets
fake_df = pd.read_csv("fake.csv")
true_df = pd.read_csv("true.csv")

# Balance the dataset
min_len = min(len(fake_df), len(true_df))
fake_df = fake_df.sample(min_len, random_state=42)
true_df = true_df.sample(min_len, random_state=42)

# Label data
fake_df['label'] = 1
true_df['label'] = 0

# Combine and shuffle
data = pd.concat([fake_df, true_df], axis=0)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Combine text
data['content'] = data['title'].fillna('') + ' ' + data['text'].fillna('')
data['content'] = data['content'].str.replace(r'\W', ' ', regex=True).str.lower()

# Features and labels
X = data['content']
y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=2)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Save model and vectorizer
with open("fake_news_model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

print("✅ Balanced model trained and saved.")
