# AI-Powered Skill Gap Analysis and Learning Pathways

A microservice that analyzes skill gaps between student profiles and target career roles, and generates personalized learning pathways.

## Features

- **Skill Gap Analysis**: Identifies gaps between current skills and required skills for target roles
- **Learning Pathways**: Generates multiple pathway options (Fast Track, Comprehensive, Foundation)
- **Resource Recommendations**: Suggests courses, tutorials, projects, and certifications
- **Time Estimation**: Estimates time to readiness for target role
- **Priority Scoring**: Ranks skill gaps by importance and urgency

## API Endpoints

### POST `/analyze`
Analyze skill gaps and generate comprehensive learning pathways.

**Request Body:**
```json
{
  "student": {
    "student_id": 1,
    "name": "John Doe",
    "batch": 2024,
    "dept_id": 1,
    "department_name": "Computer Science",
    "cgpa": 3.5,
    "skills": [
      {
        "skill_id": 1,
        "name": "Python",
        "proficiency_level": "Intermediate",
        "is_verified": true
      }
    ],
    "career_goals": ["Software Engineer", "Data Scientist"]
  },
  "target_role": {
    "role_id": 1,
    "role_name": "Senior Software Engineer",
    "required_skills": [
      {
        "name": "Python",
        "required_level": "Advanced",
        "is_mandatory": true,
        "weight": 1.0
      },
      {
        "name": "System Design",
        "required_level": "Intermediate",
        "is_mandatory": true,
        "weight": 0.8
      }
    ],
    "priority": "high"
  }
}
```

### POST `/pathways`
Generate multiple learning pathway options.

### GET `/health`
Health check endpoint.

## Running the Service

### FastAPI Service (REST API)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python main.py

# Or with uvicorn
uvicorn main:app --reload --port 8004
```

### Streamlit App (Interactive UI)

```bash
# Install dependencies (includes streamlit)
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit-app.py

# The app will open in your browser at http://localhost:8501
```

The Streamlit app provides an interactive interface where you can:
- Input student profile and skills
- Define target role and required skills
- View skill gap analysis with visualizations
- Explore learning pathways
- See recommendations and resources
- Download results as JSON

## Integration with Gateway

To integrate with the main gateway, add to `gateway/main.py`:

```python
model_configs = [
    ("explainable-ai-recommendations", "/xai"),
    ("predictive-career-path-model", "/career-path"),
    ("ai-skill-gap-analysis", "/skill-gap"),  # Add this line
]
```

## Future Enhancements

- Database integration for learning resources
- Integration with external learning platforms (Coursera, Udemy, etc.)
- Progress tracking and adaptive learning paths
- Skill verification and certification tracking
- Collaborative learning recommendations

