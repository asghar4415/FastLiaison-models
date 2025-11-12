from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from scipy.sparse import hstack
import os
from typing import List, Dict

# --- 1. New, Richer Pydantic Models for the Output ---


class SkillAlignment(BaseModel):
    strong_points: List[str]
    skills_to_develop: List[str]


class CareerSuggestion(BaseModel):
    role_name: str
    match_confidence: float = Field(..., ge=0, le=1)
    summary: str
    alignment: SkillAlignment


class RefinedPredictionResponse(BaseModel):
    career_suggestions: List[CareerSuggestion]

# --- (Input Pydantic models remain the same) ---


class Internship(BaseModel):
    role: str
    description: str


class Project(BaseModel):
    title: str
    description: str


class StudentProfile(BaseModel):
    cgpa: float = Field(..., gt=0, lt=4.1)
    work_experience_months: int = Field(..., ge=0)
    internships: List[Internship]
    projects: List[Project]
    skills_list: List[str]
    courses_taken: List[str]
    extra_curricular: str


# --- 2. Load Models (No change here) ---

MODEL_DIR = MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
JOBS_DIR = os.path.join(os.path.dirname(__file__), "data")


try:
    model_path = os.path.join(MODEL_DIR, "career_predictor.pkl")
    model = joblib.load(model_path)
    experience_vectorizer = joblib.load(
        os.path.join(MODEL_DIR, 'experience_vectorizer.pkl'))
    skills_vectorizer = joblib.load(
        os.path.join(MODEL_DIR, 'skills_vectorizer.pkl'))
    courses_vectorizer = joblib.load(
        os.path.join(MODEL_DIR, 'courses_vectorizer.pkl'))
    mlb = joblib.load(os.path.join(MODEL_DIR, 'multilabel_binarizer.pkl'))
    job_skills_df = pd.read_csv(os.path.join(JOBS_DIR, 'job_skills.csv'))
    job_skills_df['required_skills'] = job_skills_df['required_skills'].apply(
        lambda x: x.split(','))
except FileNotFoundError as e:
    raise RuntimeError(
        f"Model file not found. Have you run train.py? Error: {e}")

app = FastAPI(title="FAST Liaison AI Career Predictor (Refined Output)")

# --- 3. Helper Functions ---


def combine_text_features(profile: StudentProfile):
    internship_text = ' '.join(
        [f"{i.role} {i.description}" for i in profile.internships])
    project_text = ' '.join(
        [f"{p.title} {p.description}" for p in profile.projects])
    return f"{internship_text} {project_text} {profile.extra_curricular}"


def get_skill_alignment(student_skills: list, target_role: str) -> SkillAlignment:
    """Compares student skills to required skills and returns both matches and gaps."""
    strong_points = []
    skills_to_develop = []

    try:
        required_skills_series = job_skills_df[job_skills_df['job_role']
                                               == target_role]['required_skills']
        if not required_skills_series.empty:
            required_skills = required_skills_series.iloc[0]
            student_skills_set = {s.lower() for s in student_skills}

            for skill in required_skills:
                if skill.lower() in student_skills_set:
                    strong_points.append(skill)
                else:
                    skills_to_develop.append(skill)
    except Exception:
        # In case of error, return empty lists
        pass

    return SkillAlignment(strong_points=strong_points, skills_to_develop=skills_to_develop)

# --- 4. THE REFINED API ENDPOINT ---


@app.post("/predict", response_model=RefinedPredictionResponse)
async def predict_career(profile: StudentProfile):
    """Predicts potential career roles with a detailed, clear, and actionable analysis."""

    # --- Feature Engineering for Input ---
    experience_text = combine_text_features(profile)
    skills_text = ' '.join(profile.skills_list)
    courses_text = ' '.join(profile.courses_taken)

    experience_tfidf = experience_vectorizer.transform([experience_text])
    skills_tfidf = skills_vectorizer.transform([skills_text])
    courses_tfidf = courses_vectorizer.transform([courses_text])
    numerical_features = [[profile.cgpa, profile.work_experience_months]]

    X_input = hstack([experience_tfidf, skills_tfidf,
                     courses_tfidf, numerical_features]).tocsr()

    # --- Prediction ---
    # Get the probabilities for each possible role (this is key for the confidence score)
    # The result is a list of lists, e.g., [[0.1, 0.85, 0.72, ...]]
    probabilities = model.predict_proba(X_input)[0]

    # Create a dictionary mapping role names to their confidence scores
    confidence_scores = {role: prob for role,
                         prob in zip(mlb.classes_, probabilities)}

    # Get the roles where the probability is above a certain threshold (e.g., 20%)
    prediction_threshold = 0.20
    predicted_roles = [role for role, score in confidence_scores.items(
    ) if score >= prediction_threshold]

    if not predicted_roles:
        # Handle cases where no role meets the threshold
        # Find the single best role and return only that one
        top_role = mlb.classes_[probabilities.argmax()]
        predicted_roles.append(top_role)

    # --- Build the Rich Response ---
    suggestions = []
    for role in predicted_roles:
        alignment = get_skill_alignment(profile.skills_list, role)

        # Create a simple, dynamic summary
        summary = f"Your profile shows a good alignment for a {role} role."
        if alignment.strong_points:
            summary += f" Your skills in {', '.join(alignment.strong_points[:2])} are particularly relevant."

        suggestion = CareerSuggestion(
            role_name=role,
            match_confidence=round(confidence_scores[role], 2),
            summary=summary,
            alignment=alignment
        )
        suggestions.append(suggestion)

    # Sort suggestions by confidence score, from highest to lowest
    suggestions.sort(key=lambda s: s.match_confidence, reverse=True)

    return RefinedPredictionResponse(career_suggestions=suggestions)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Refined Career Prediction API. Go to /docs to test."}
