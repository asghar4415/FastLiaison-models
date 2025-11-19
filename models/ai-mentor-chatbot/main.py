from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from career_mentor import get_career_advice

app = FastAPI(title="FAST Liaison AI Microservice")

# Define the Data Structure expecting from NestJS


class StudentProfile(BaseModel):
    name: str
    cgpa: float
    major: str
    skills: List[str]
    experience: Optional[str] = None


class ChatRequest(BaseModel):
    student_profile: StudentProfile
    message: str


@app.post("/chat/mentor")
async def chat_mentor(request: ChatRequest):
    try:
        # Convert Pydantic model to dict
        student_data = request.student_profile.dict()
        user_query = request.message

        # Get response from AI
        ai_response = get_career_advice(student_data, user_query)

        return {"response": ai_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn main:app --reload --port 8000
