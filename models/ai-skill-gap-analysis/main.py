from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from skill_gap_analyzer import SkillGapAnalyzer

app = FastAPI(
    title="AI-Powered Skill Gap Analysis and Learning Pathways",
    description="Analyze skill gaps and generate personalized learning pathways for students",
    version="1.0.0"
)

# Pydantic Models for Request/Response

class Skill(BaseModel):
    skill_id: int
    name: str
    proficiency_level: str = Field(..., description="Beginner, Intermediate, Advanced, Expert")
    is_verified: bool = False

class StudentProfile(BaseModel):
    student_id: int
    name: str
    batch: int = Field(..., description="Graduation year, e.g., 2024")
    dept_id: int
    department_name: str
    cgpa: float = Field(..., ge=0, le=4)
    skills: List[Skill]
    career_goals: Optional[List[str]] = Field(default_factory=list, description="Target career roles or positions")

class TargetRole(BaseModel):
    role_id: int
    role_name: str
    required_skills: List[Dict[str, Any]] = Field(..., description="List of required skills with proficiency levels")
    priority: Optional[str] = Field(default="medium", description="high, medium, low")

class SkillGapRequest(BaseModel):
    student: StudentProfile
    target_role: TargetRole

class LearningResource(BaseModel):
    resource_id: str
    title: str
    type: str = Field(..., description="course, tutorial, book, project, certification")
    url: Optional[str] = None
    duration: Optional[str] = None
    difficulty: str = Field(..., description="Beginner, Intermediate, Advanced")
    estimated_hours: Optional[int] = None
    cost: Optional[str] = Field(default="Free", description="Free, Paid, or specific amount")

class SkillGapItem(BaseModel):
    skill_name: str
    current_level: Optional[str] = None
    required_level: str
    gap_severity: str = Field(..., description="critical, high, medium, low")
    priority_score: float = Field(..., ge=0, le=1)
    learning_resources: List[LearningResource]

class LearningPathway(BaseModel):
    pathway_id: str
    pathway_name: str
    description: str
    estimated_completion_time: str
    difficulty: str
    skills_covered: List[str]
    resources: List[LearningResource]
    milestones: List[Dict[str, Any]]

class SkillGapResponse(BaseModel):
    student_name: str
    target_role: str
    overall_gap_score: float = Field(..., ge=0, le=1, description="0 = no gap, 1 = maximum gap")
    skill_gaps: List[SkillGapItem]
    learning_pathways: List[LearningPathway]
    recommendations: Dict[str, Any]
    estimated_time_to_readiness: str

@app.post("/analyze", response_model=SkillGapResponse)
async def analyze_skill_gap(request: SkillGapRequest):
    """
    Analyze skill gaps between student profile and target role, 
    and generate personalized learning pathways
    """
    try:
        analyzer = SkillGapAnalyzer(
            student_data=request.student,
            target_role_data=request.target_role
        )
        
        result = analyzer.generate_gap_analysis()
        
        # Convert dictionaries to Pydantic models
        skill_gaps = [
            SkillGapItem(**gap) for gap in result['skill_gaps']
        ]
        
        learning_pathways = [
            LearningPathway(**pathway) for pathway in result['learning_pathways']
        ]
        
        return SkillGapResponse(
            student_name=request.student.name,
            target_role=request.target_role.role_name,
            overall_gap_score=result['overall_gap_score'],
            skill_gaps=skill_gaps,
            learning_pathways=learning_pathways,
            recommendations=result['recommendations'],
            estimated_time_to_readiness=result['estimated_time_to_readiness']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing skill gap: {str(e)}")

@app.post("/pathways")
async def generate_learning_pathways(request: SkillGapRequest):
    """
    Generate multiple learning pathway options for skill development
    """
    try:
        analyzer = SkillGapAnalyzer(
            student_data=request.student,
            target_role_data=request.target_role
        )
        
        pathways = analyzer.generate_learning_pathways()
        
        return {
            "student_name": request.student.name,
            "target_role": request.target_role.role_name,
            "pathways": pathways
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating pathways: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AI Skill Gap Analysis"}

@app.get("/")
async def root():
    return {
        "message": "Welcome to AI-Powered Skill Gap Analysis and Learning Pathways",
        "version": "1.0.0",
        "endpoints": {
            "/analyze": "POST - Analyze skill gaps and generate learning pathways",
            "/pathways": "POST - Generate learning pathway options",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)

