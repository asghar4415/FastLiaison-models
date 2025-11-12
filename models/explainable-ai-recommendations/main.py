from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from xai import ExplainableJobMatcher

app = FastAPI(
    title="Explainable AI Job Matching Service",
    description="AI-powered job matching with detailed explanations",
    version="1.0.0"
)

# Pydantic Models for Request/Response

class Skill(BaseModel):
    skill_id: int
    name: str
    proficiency_level: str = Field(..., description="Beginner, Intermediate, Advanced, Expert")
    is_verified: bool = False

class Project(BaseModel):
    p_id: int
    title: str
    description: Optional[str] = None
    is_verified: bool = False
    skills: List[int] = Field(default_factory=list, description="List of skill_ids used in project")

class Course(BaseModel):
    course_id: int
    course_name: str
    grade: float = Field(..., ge=0, le=4, description="Grade on 4.0 scale")

class StudentProfile(BaseModel):
    student_id: int
    name: str
    batch: int = Field(..., description="Graduation year, e.g., 2024")
    dept_id: int
    department_name: str
    cgpa: float = Field(..., ge=0, le=4)
    skills: List[Skill]
    projects: List[Project]
    courses: List[Course]

class RequiredSkill(BaseModel):
    skill_id: int
    name: str
    required_level: str = Field(..., description="Beginner, Intermediate, Advanced, Expert")
    is_mandatory: bool = True
    weight: float = Field(default=1.0, description="Importance weight of this skill")

class JobDescription(BaseModel):
    job_id: int
    title: str
    description: str
    company: str
    eligible_batches: List[int] = Field(..., description="Eligible graduation years")
    eligible_departments: List[int]
    eligible_cgpa_min: float = Field(..., ge=0, le=4)
    required_skills: List[RequiredSkill]

class MatchRequest(BaseModel):
    student: StudentProfile
    job: JobDescription

class MatchResponse(BaseModel):
    match_score: float
    recommendation_type: str
    explanation: Dict[str, Any]
    scores_breakdown: Dict[str, Any]
    student_name: str
    job_title: str
    company: str

# Mock data adapter for XAI class
class MockDataAdapter:
    """
    Adapter to make XAI class work with in-memory data instead of DB
    """
    def __init__(self, student_data: StudentProfile, job_data: JobDescription):
        self.student_data = student_data.dict()
        self.job_data = job_data.dict()
    
    def get_student_profile(self, student_id):
        return self.student_data
    
    def get_job_details(self, job_id):
        return self.job_data
    
    def get_student_skills(self):
        return self.student_data.get('skills', [])
    
    def get_job_required_skills(self):
        return self.job_data.get('required_skills', [])
    
    def get_student_projects(self):
        return self.student_data.get('projects', [])
    
    def get_project_skills(self, project_id):
        project = next(
            (p for p in self.student_data.get('projects', []) if p['p_id'] == project_id),
            None
        )
        if project:
            # Return skills based on skill_ids in project
            return [s for s in self.student_data.get('skills', []) if s['skill_id'] in project.get('skills', [])]
        return []
    
    def get_student_courses(self):
        return self.student_data.get('courses', [])
    
    def find_relevant_courses(self, courses, job_title, job_description):
        # Simple keyword matching
        keywords = set(job_title.lower().split() + job_description.lower().split())
        relevant = []
        for course in courses:
            course_words = set(course['course_name'].lower().split())
            if keywords.intersection(course_words):
                relevant.append(course)
        return relevant
    
    def calculate_average_grade(self, courses):
        if not courses:
            return 0
        return sum(c['grade'] for c in courses) / len(courses)
    
    def compare_proficiency(self, student_level, required_level):
        levels = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3, 'Expert': 4}
        student_val = levels.get(student_level, 0)
        required_val = levels.get(required_level, 0)
        
        if student_val >= required_val:
            return 1.0
        elif student_val == required_val - 1:
            return 0.7
        else:
            return 0.4
    
    def calculate_weighted_score(self, scores):
        weights = {
            'skills': 0.40,
            'education': 0.20,
            'projects': 0.15,
            'courses': 0.15,
            'cgpa': 0.10
        }
        total = sum(scores[key]['score'] * weights[key] for key in weights)
        return round(total, 2)

# Modify XAI class to accept adapter
class AdaptedExplainableJobMatcher(ExplainableJobMatcher):
    def __init__(self, student_data, job_data):
        self.adapter = MockDataAdapter(student_data, job_data)
        self.student = self.adapter.get_student_profile(student_data.student_id)
        self.job = self.adapter.get_job_details(job_data.job_id)
        self.weights = {
            'skills': 0.40,
            'education': 0.20,
            'projects': 0.15,
            'courses': 0.15,
            'cgpa': 0.10
        }
        self.graduation_year_multiplier = self.calculate_graduation_year_multiplier()
    
    # Override methods to use adapter
    def get_student_skills(self):
        return self.adapter.get_student_skills()
    
    def get_job_required_skills(self):
        return self.adapter.get_job_required_skills()
    
    def get_student_projects(self):
        return self.adapter.get_student_projects()
    
    def get_project_skills(self, project_id):
        return self.adapter.get_project_skills(project_id)
    
    def get_student_courses(self):
        return self.adapter.get_student_courses()
    
    def find_relevant_courses(self, courses, job_title, job_description):
        return self.adapter.find_relevant_courses(courses, job_title, job_description)
    
    def calculate_average_grade(self, courses):
        return self.adapter.calculate_average_grade(courses)
    
    def compare_proficiency(self, student_level, required_level):
        return self.adapter.compare_proficiency(student_level, required_level)
    
    def calculate_weighted_score(self, scores):
        return self.adapter.calculate_weighted_score(scores)

@app.post("/match", response_model=MatchResponse)
async def match_student_to_job(request: MatchRequest):
    """
    Match a student to a job and provide detailed explanation
    
    Returns match score, recommendation type, and comprehensive analysis
    """
    try:
        matcher = AdaptedExplainableJobMatcher(
            student_data=request.student,
            job_data=request.job
        )
        
        result = matcher.generate_match_with_explanation()
        
        return MatchResponse(
            match_score=result['match_score'],
            recommendation_type=result['recommendation_type'],
            explanation=result['explanation'],
            scores_breakdown=result['scores_breakdown'],
            student_name=request.student.name,
            job_title=request.job.title,
            company=request.job.company
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing match: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "XAI Job Matcher"}

# For future DB integration
@app.post("/match-by-ids")
async def match_by_ids(student_id: int, job_id: int):
    """
    Future endpoint: Match using IDs with DB lookup
    Currently not implemented - requires database connection
    """
    raise HTTPException(
        status_code=501, 
        detail="Database integration not yet implemented. Use /match endpoint with full data."
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)