from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Gemini via LangChain (no api_base in constructor)
gemini_llm = ChatOpenAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    api_key=os.getenv("GEMINI_API_KEY")
)


def get_career_advice(student_data, user_query, chat_history=None):
    """
    Returns career advice from Gemini LLM.
    student_data: dict
    user_query: str
    chat_history: list of {"role": "user"/"assistant", "message": str}
    """
    try:
        # System prompt
        system_template = """
You are an expert AI Career Mentor for university students.
Guide students in:
1. Resume building
2. Interview preparation
3. Career exploration

Always tailor advice based on student profile (GPA, major, skills, experience).

Student Profile:
- Name: {name}
- CGPA: {gpa}
- Major: {major}
- Skills: {skills}
- Resume/Experience: {experience}

Guidelines:
1. Be encouraging but realistic.
2. If GPA < 2.5, suggest academic improvement strategies.
3. Reference previous conversation history.
4. Use resume/experience content to suggest improvements in skills, projects, or job applications.
"""

        # Build last 10 messages as text for context
        history_text = ""
        if chat_history:
            for m in chat_history[-10:]:
                role = "Student" if m["role"] == "user" else "Mentor"
                history_text += f"{role}: {m['message']}\n"

        # Construct prompt with system and user messages
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", "{query}\nPrevious Conversation:\n" + history_text)
        ])

        # Run the LLM chain
        response = (prompt | gemini_llm | StrOutputParser()).invoke({
            "name": student_data["name"],
            "gpa": student_data["cgpa"],
            "major": student_data["major"],
            "skills": ", ".join(student_data["skills"]),
            "experience": student_data.get("experience", "None listed"),
            "query": user_query
        })

        return response

    except Exception as e:
        # Fallback for testing if API fails
        return f"[MOCK RESPONSE] Could not reach Gemini API: {e}"
