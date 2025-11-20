# Quick Start Guide - Skill Gap Analysis Service

## 🎯 What This Service Does

The **AI-Powered Skill Gap Analysis** service helps students understand:
1. **What skills they're missing** for their target career role
2. **How big the gaps are** (critical, high, medium, low)
3. **What to learn first** (prioritized recommendations)
4. **How to learn it** (personalized learning pathways)
5. **How long it will take** (time estimates)

## 📊 How It Works (Simple Explanation)

### Step 1: Input
- **Your Profile**: Your current skills (e.g., "Python - Intermediate", "React - Beginner")
- **Target Role**: What job you want (e.g., "Senior Software Engineer")
- **Required Skills**: What skills that job needs (e.g., "Python - Advanced", "System Design - Intermediate")

### Step 2: Comparison
The system compares:
- ✅ Skills you HAVE vs. skills you NEED
- ✅ Your proficiency LEVEL vs. required LEVEL
- Example: You have Python (Intermediate) but need Python (Advanced) = **Medium Gap**

### Step 3: Analysis
For each skill gap, it calculates:
- **Gap Severity**: How big is the gap?
  - 🔴 Critical: Missing skill or huge gap (3+ levels)
  - 🟠 High: Big gap (2 levels)
  - 🟡 Medium: Small gap (1 level)
  - 🟢 Low: Almost there!
- **Priority Score**: How important is this skill? (0-1 scale)

### Step 4: Learning Pathways
Creates 3 different learning paths:
1. **Fast Track**: Focus on critical gaps, intensive learning
2. **Comprehensive**: Cover all gaps, moderate pace
3. **Foundation**: Build basics first, beginner-friendly

### Step 5: Resources & Recommendations
- Suggests courses, tutorials, projects
- Estimates time needed
- Provides actionable next steps

## 🚀 Running the Service

### Option 1: Streamlit App (Recommended for Testing)

```bash
cd FastLiaison-models/models/ai-skill-gap-analysis
pip install -r requirements.txt
streamlit run streamlit-app.py
```

**Features:**
- Interactive UI in your browser
- Easy input forms
- Visual results with charts
- Download results as JSON

### Option 2: FastAPI Service (For Integration)

```bash
cd FastLiaison-models/models/ai-skill-gap-analysis
pip install -r requirements.txt
python main.py
```

**Access:**
- API: http://localhost:8004
- Docs: http://localhost:8004/docs
- Via Gateway: http://localhost:8000/skill-gap

## 📝 Example Usage

### Scenario:
- **Student**: Has Python (Intermediate), React (Beginner)
- **Target Role**: Senior Software Engineer
- **Required**: Python (Advanced), System Design (Intermediate), Docker (Intermediate)

### Result:
1. **Python Gap**: Medium (Intermediate → Advanced)
2. **System Design Gap**: Critical (Missing → Intermediate)
3. **Docker Gap**: Critical (Missing → Intermediate)

**Recommendation**: Start with System Design and Docker (critical gaps), then improve Python.

**Time Estimate**: 2-3 months to reach required level

## 🔧 Key Components

### Files:
- `main.py` - FastAPI REST API service
- `skill_gap_analyzer.py` - Core analysis logic
- `streamlit-app.py` - Interactive web UI
- `requirements.txt` - Dependencies

### Main Functions:
- `identify_skill_gaps()` - Finds gaps between current and required skills
- `calculate_gap_severity()` - Determines how critical each gap is
- `generate_learning_pathways()` - Creates learning roadmaps
- `get_learning_resources()` - Suggests courses/tutorials
- `estimate_time_to_readiness()` - Calculates timeline

## 💡 Use Cases

1. **Career Planning**: "What should I learn to become a Data Scientist?"
2. **Job Application Prep**: "Am I ready for this Senior Developer role?"
3. **Skill Development**: "What's the fastest way to fill my skill gaps?"
4. **Academic Guidance**: "Which courses should I take next semester?"

## 🎨 Streamlit App Features

- **Sidebar Input**: Easy forms for student profile and target role
- **Visual Results**: Color-coded gap severity indicators
- **Tabbed Interface**: 
  - Skill Gaps (detailed breakdown)
  - Learning Pathways (3 options)
  - Recommendations (actionable steps)
  - Resources (all learning materials)
- **Download Results**: Export analysis as JSON

## 🔗 Integration

Already integrated with the main gateway at `/skill-gap` endpoint!

Test via gateway:
```bash
# Start gateway
cd FastLiaison-models/gateway
python main.py

# Access service
curl http://localhost:8000/skill-gap/health
```

## 📚 Next Steps

1. **Try the Streamlit app** - Best way to understand the service
2. **Test with your own data** - Input your skills and target role
3. **Explore the API** - Check `/docs` endpoint for interactive API testing
4. **Integrate** - Use the API in your applications

## 🐛 Troubleshooting

**Streamlit not opening?**
- Check if port 8501 is available
- Try: `streamlit run streamlit-app.py --server.port 8502`

**Import errors?**
- Make sure you're in the correct directory
- Run: `pip install -r requirements.txt`

**Analysis not working?**
- Ensure you have at least one skill in profile
- Ensure you have at least one required skill for target role

