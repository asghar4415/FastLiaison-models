import streamlit as st
import json
import uuid
from pathlib import Path
from career_mentor import get_career_advice

# Config
CONVO_FILE = Path("conversations.json")

# ---------------- HELPERS ---------------- #


def load_conversations():
    if not CONVO_FILE.exists():
        CONVO_FILE.write_text("{}")
    return json.loads(CONVO_FILE.read_text())


def save_conversations(data):
    CONVO_FILE.write_text(json.dumps(data, indent=4))


def append_message(student_id, role, message):
    data = load_conversations()
    if student_id not in data:
        data[student_id] = []
    data[student_id].append({"role": role, "message": message})
    save_conversations(data)


def get_conversation(student_id):
    data = load_conversations()
    return data.get(student_id, [])


# ---------------- STREAMLIT UI ---------------- #
st.set_page_config(page_title="AI Career Mentor", page_icon="🤖")
st.title("🤖 FAST AI Career Mentor Chatbot")
st.markdown(
    "This chatbot uses mock student data and stores conversations locally.")
st.divider()

# Sidebar profile input
with st.sidebar:
    st.header("Mock Student Profile")
    student_id = st.text_input("Student ID", value=str(uuid.uuid4()))
    student_name = st.text_input("Name", value="Ali Khan")
    cgpa = st.number_input("CGPA", min_value=0.0,
                           max_value=4.0, value=3.2, step=0.01)
    major = st.text_input("Major", value="Computer Science")
    skills = st.text_input("Skills (comma separated)",
                           value="Python, React, SQL")

    # Resume / experience input
    st.subheader("Upload Resume or Paste Experience")
    resume_file = st.file_uploader(
        "Upload Resume (txt/pdf)", type=["txt", "pdf"])
    resume_text = ""
    if resume_file is not None:
        if resume_file.type == "application/pdf":
            # Optional: use PyPDF2 or pdfplumber to extract text
            import pdfplumber
            with pdfplumber.open(resume_file) as pdf:
                resume_text = "\n".join(page.extract_text()
                                        for page in pdf.pages if page.extract_text())
        else:
            resume_text = resume_file.read().decode("utf-8")
    else:
        resume_text = st.text_area(
            "Or paste resume / experience here", value="Internship at ABC Tech")

    if st.button("Save Profile"):
        st.session_state["student_profile"] = {
            "id": student_id,
            "name": student_name,
            "cgpa": cgpa,
            "major": major,
            "skills": [s.strip() for s in skills.split(",")],
            "experience": resume_text
        }
        st.success("Student profile saved!")

# Load profile
if "student_profile" not in st.session_state:
    st.info("Please enter mock student data from the sidebar.")
    st.stop()

profile = st.session_state["student_profile"]
st.subheader(f"Chat as: **{profile['name']}** ({profile['major']})")
conversation_id = st.session_state.get("conversation_id", str(uuid.uuid4()))
conversation_history = get_conversation(profile["id"])

# Display chat
for msg in conversation_history:
    role = "user" if msg["role"] == "user" else "assistant"
    st.chat_message(role).markdown(msg["message"])

# New message input
prompt = st.chat_input("Ask your career mentor...")

if prompt:
    st.chat_message("user").markdown(prompt)
    append_message(profile["id"], "user", prompt)

    # Get AI response
    response = get_career_advice(profile, prompt, conversation_history)
    st.chat_message("assistant").markdown(response)
    append_message(profile["id"], "assistant", response)

    st.session_state["conversation_id"] = conversation_id
