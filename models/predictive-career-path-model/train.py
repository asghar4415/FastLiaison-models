import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from scipy.sparse import hstack
import joblib
import os
import json

# --- 1. Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'data.jsonl')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# --- 2. Helper Functions ---


def load_jsonl(file_path):
    """Loads a .jsonl file into a pandas DataFrame."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


def combine_text_features(row):
    """Combines all text from internships, projects, and extracurriculars."""
    internship_text = ' '.join(
        [f"{i['role']} {i['description']}" for i in row.get('internships', [])])
    project_text = ' '.join(
        [f"{p['title']} {p['description']}" for p in row.get('projects', [])])
    extracurricular = row.get('extra_curricular', '')
    return f"{internship_text} {project_text} {extracurricular}"


# --- 3. Load and Preprocess Data ---
print("Loading and preprocessing data...")
df = load_jsonl(DATA_FILE)

# Combine text features
df['experience_text'] = df.apply(combine_text_features, axis=1)
df['skills_str'] = df['skills_list'].apply(lambda x: ' '.join(x))
df['courses_str'] = df['courses_taken'].apply(lambda x: ' '.join(x))

# --- 4. Feature Engineering ---
print("Performing feature engineering...")

experience_vectorizer = TfidfVectorizer(max_features=200, stop_words='english')
skills_vectorizer = TfidfVectorizer(max_features=100)
courses_vectorizer = TfidfVectorizer(max_features=100)

experience_tfidf = experience_vectorizer.fit_transform(df['experience_text'])
skills_tfidf = skills_vectorizer.fit_transform(df['skills_str'])
courses_tfidf = courses_vectorizer.fit_transform(df['courses_str'])

numerical_features = df[['cgpa', 'work_experience_months']].fillna(0).values

X = hstack([experience_tfidf, skills_tfidf,
           courses_tfidf, numerical_features]).tocsr()

# --- 5. Labels ---
print("Processing labels...")
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['label_roles'])

# --- 5a. Filter out labels that appear in all or too few examples ---
min_occurrence = 2
max_occurrence = len(df) - 1  # cannot appear in all examples
valid_labels = [
    i for i in range(y.shape[1])
    if y[:, i].sum() >= min_occurrence and y[:, i].sum() <= max_occurrence
]

# Keep only the valid labels
y = y[:, valid_labels]
mlb.classes_ = mlb.classes_[valid_labels]
print(f"Number of valid labels after filtering: {len(mlb.classes_)}")

# --- 6. Training ---
print("Training the model...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

base_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
model = OneVsRestClassifier(base_classifier)
model.fit(X_train, y_train)

# --- 7. Evaluation ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='micro')

print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")
print(f"✅ F1-Score (micro): {f1:.2f}")

# --- 8. Save Artifacts ---
print("\n💾 Saving model and transformers...")
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, os.path.join(MODEL_DIR, 'career_predictor.pkl'))
joblib.dump(experience_vectorizer, os.path.join(
    MODEL_DIR, 'experience_vectorizer.pkl'))
joblib.dump(skills_vectorizer, os.path.join(
    MODEL_DIR, 'skills_vectorizer.pkl'))
joblib.dump(courses_vectorizer, os.path.join(
    MODEL_DIR, 'courses_vectorizer.pkl'))
joblib.dump(mlb, os.path.join(MODEL_DIR, 'multilabel_binarizer.pkl'))

print("\n-----------------------------------")
print("🎉 Training complete!")
print("All artifacts saved inside 'models/' folder.")
print("Next step: run 'python app/main.py' to start the API server.")
