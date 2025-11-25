import streamlit as st
import json
from pathlib import Path
from skill_gap_analyzer import SkillGapAnalyzer
from typing import Dict, Any, List

# Page config
st.set_page_config(
    page_title="Skill Gap Analysis",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .gap-score {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
    }
    .skill-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    .critical { border-left-color: #d32f2f; background-color: #ffebee; }
    .high { border-left-color: #f57c00; background-color: #fff3e0; }
    .medium { border-left-color: #fbc02d; background-color: #fffde7; }
    .low { border-left-color: #388e3c; background-color: #e8f5e9; }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">📊 AI-Powered Skill Gap Analysis</div>', unsafe_allow_html=True)
st.markdown("Analyze your skill gaps and discover personalized learning pathways for your target career role.")
st.divider()

# Initialize session state
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

# Sidebar for input
with st.sidebar:
    st.header("📝 Student Profile")
    
    student_name = st.text_input("Name", value="John Doe")
    student_id = st.number_input("Student ID", min_value=1, value=1)
    batch = st.number_input("Graduation Year", min_value=2020, max_value=2030, value=2024)
    dept_id = st.number_input("Department ID", min_value=1, value=1)
    department_name = st.text_input("Department", value="Computer Science")
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=4.0, value=3.5, step=0.01)
    
    st.divider()
    st.subheader("Your Skills")
    
    # Skills input
    num_skills = st.number_input("Number of Skills", min_value=0, max_value=20, value=3)
    skills = []
    
    for i in range(num_skills):
        with st.expander(f"Skill {i+1}", expanded=(i < 3)):
            skill_id = st.number_input(f"Skill ID {i+1}", min_value=1, value=i+1, key=f"skill_id_{i}")
            skill_name = st.text_input(f"Skill Name {i+1}", value=["Python", "React", "SQL"][i] if i < 3 else "", key=f"skill_name_{i}")
            proficiency = st.selectbox(
                f"Proficiency Level {i+1}",
                ["Beginner", "Intermediate", "Advanced", "Expert"],
                index=1 if i < 3 else 0,
                key=f"proficiency_{i}"
            )
            is_verified = st.checkbox(f"Verified {i+1}", value=False, key=f"verified_{i}")
            
            if skill_name:
                skills.append({
                    "skill_id": int(skill_id),
                    "name": skill_name,
                    "proficiency_level": proficiency,
                    "is_verified": is_verified
                })
    
    st.divider()
    st.header("🎯 Target Role")
    
    role_id = st.number_input("Role ID", min_value=1, value=1)
    role_name = st.text_input("Role Name", value="Senior Software Engineer")
    role_priority = st.selectbox("Priority", ["high", "medium", "low"], index=1)
    
    st.subheader("Required Skills")
    num_required = st.number_input("Number of Required Skills", min_value=0, max_value=20, value=4)
    required_skills = []
    
    default_required = [
        {"name": "Python", "level": "Advanced", "mandatory": True, "weight": 1.0},
        {"name": "System Design", "level": "Intermediate", "mandatory": True, "weight": 0.9},
        {"name": "Docker", "level": "Intermediate", "mandatory": False, "weight": 0.7},
        {"name": "AWS", "level": "Intermediate", "mandatory": False, "weight": 0.8}
    ]
    
    for i in range(num_required):
        with st.expander(f"Required Skill {i+1}", expanded=(i < 4)):
            req_skill_id = st.number_input(f"Skill ID {i+1}", min_value=1, value=i+10, key=f"req_skill_id_{i}")
            req_skill_name = st.text_input(
                f"Skill Name {i+1}",
                value=default_required[i]["name"] if i < len(default_required) else "",
                key=f"req_skill_name_{i}"
            )
            req_level = st.selectbox(
                f"Required Level {i+1}",
                ["Beginner", "Intermediate", "Advanced", "Expert"],
                index=["Beginner", "Intermediate", "Advanced", "Expert"].index(default_required[i]["level"]) if i < len(default_required) else 1,
                key=f"req_level_{i}"
            )
            is_mandatory = st.checkbox(f"Mandatory {i+1}", value=default_required[i]["mandatory"] if i < len(default_required) else True, key=f"mandatory_{i}")
            weight = st.slider(f"Weight {i+1}", 0.0, 1.0, value=default_required[i]["weight"] if i < len(default_required) else 1.0, key=f"weight_{i}")
            
            if req_skill_name:
                required_skills.append({
                    "skill_id": int(req_skill_id),
                    "name": req_skill_name,
                    "required_level": req_level,
                    "is_mandatory": is_mandatory,
                    "weight": weight
                })
    
    # Analyze button
    if st.button("🔍 Analyze Skill Gaps", type="primary", use_container_width=True):
        if not skills:
            st.error("Please add at least one skill to your profile.")
        elif not required_skills:
            st.error("Please add at least one required skill for the target role.")
        else:
            with st.spinner("Analyzing skill gaps..."):
                try:
                    # Create student profile
                    student_profile = {
                        "student_id": int(student_id),
                        "name": student_name,
                        "batch": int(batch),
                        "dept_id": int(dept_id),
                        "department_name": department_name,
                        "cgpa": float(cgpa),
                        "skills": skills
                    }
                    
                    # Create target role
                    target_role = {
                        "role_id": int(role_id),
                        "role_name": role_name,
                        "required_skills": required_skills,
                        "priority": role_priority
                    }
                    
                    # Run analysis
                    analyzer = SkillGapAnalyzer(student_profile, target_role)
                    result = analyzer.generate_gap_analysis()
                    
                    st.session_state.analysis_result = result
                    st.session_state.student_name = student_name
                    st.session_state.target_role = role_name
                    st.success("Analysis complete!")
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

# Main content area
if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    
    # Overall Gap Score
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        gap_score = result['overall_gap_score']
        score_color = "🔴" if gap_score > 0.7 else "🟠" if gap_score > 0.4 else "🟡" if gap_score > 0.2 else "🟢"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                     border-radius: 1rem; color: white; margin: 1rem 0;">
            <h2 style="margin: 0; color: white;">Overall Gap Score</h2>
            <div class="gap-score" style="color: white;">{score_color} {gap_score:.1%}</div>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                {result['estimated_time_to_readiness']} to reach target role
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Skill Gaps", "🛤️ Learning Pathways", "💡 Recommendations", "📚 Resources"])
    
    # Tab 1: Skill Gaps
    with tab1:
        st.header("Skill Gap Analysis")
        
        if result['skill_gaps']:
            for gap in result['skill_gaps']:
                severity = gap['gap_severity']
                severity_colors = {
                    'critical': ('#d32f2f', '🔴'),
                    'high': ('#f57c00', '🟠'),
                    'medium': ('#fbc02d', '🟡'),
                    'low': ('#388e3c', '🟢')
                }
                color, icon = severity_colors.get(severity, ('#757575', '⚪'))
                
                with st.container():
                    st.markdown(f"""
                    <div class="skill-card {severity}" style="border-left-color: {color};">
                        <h3>{icon} {gap['skill_name']}</h3>
                        <p><strong>Current Level:</strong> {gap['current_level'] or 'Not Acquired'}</p>
                        <p><strong>Required Level:</strong> {gap['required_level']}</p>
                        <p><strong>Gap Severity:</strong> {severity.upper()}</p>
                        <p><strong>Priority Score:</strong> {gap['priority_score']:.2f}/1.0</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Learning resources for this skill
                    with st.expander(f"📚 Learning Resources for {gap['skill_name']}"):
                        for resource in gap['learning_resources']:
                            st.markdown(f"""
                            **{resource['title']}** ({resource['type']})
                            - Duration: {resource['duration']}
                            - Difficulty: {resource['difficulty']}
                            - Estimated Hours: {resource['estimated_hours']}
                            - Cost: {resource['cost']}
                            """)
        else:
            st.success("🎉 No skill gaps detected! You're ready for this role!")
    
    # Tab 2: Learning Pathways
    with tab2:
        st.header("Learning Pathways")
        
        pathways = result['learning_pathways']
        
        for pathway in pathways:
            with st.expander(f"🛤️ {pathway['pathway_name']} - {pathway['estimated_completion_time']}", expanded=True):
                st.markdown(f"**Description:** {pathway['description']}")
                st.markdown(f"**Difficulty:** {pathway['difficulty']}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Skills Covered")
                    for skill in pathway['skills_covered']:
                        st.markdown(f"- {skill}")
                
                with col2:
                    st.subheader("Milestones")
                    for milestone in pathway['milestones']:
                        st.markdown(f"**Week {milestone['milestone_number']}:** {milestone['skill']} → {milestone['target_level']}")
                
                st.subheader("Resources")
                for resource in pathway['resources'][:5]:  # Show top 5
                    st.markdown(f"- **{resource['title']}** ({resource['type']}) - {resource['duration']}")
    
    # Tab 3: Recommendations
    with tab3:
        st.header("Actionable Recommendations")
        
        recs = result['recommendations']
        
        if recs.get('immediate_actions'):
            st.subheader("🚨 Immediate Actions")
            for action in recs['immediate_actions']:
                st.markdown(f"""
                <div style="padding: 1rem; background-color: #fff3cd; border-radius: 0.5rem; margin: 0.5rem 0;">
                    <strong>{action['action']}</strong><br>
                    <em>{action['reason']}</em><br>
                    Priority: {action['priority'].upper()}
                </div>
                """, unsafe_allow_html=True)
        
        if recs.get('short_term_goals'):
            st.subheader("📅 Short-term Goals (2-3 months)")
            for goal in recs['short_term_goals']:
                st.markdown(f"- **{goal['goal']}** - Timeline: {goal['timeline']}")
        
        if recs.get('long_term_goals'):
            st.subheader("🎯 Long-term Goals")
            for goal in recs['long_term_goals']:
                st.markdown(f"- {goal}")
        
        if recs.get('focus_areas'):
            st.subheader("🎯 Focus Areas")
            st.markdown("Top skills to prioritize:")
            for area in recs['focus_areas']:
                st.markdown(f"- **{area}**")
    
    # Tab 4: All Resources
    with tab4:
        st.header("All Learning Resources")
        
        all_resources = []
        for gap in result['skill_gaps']:
            for resource in gap['learning_resources']:
                resource['skill'] = gap['skill_name']
                all_resources.append(resource)
        
        if all_resources:
            # Group by type
            resource_types = {}
            for resource in all_resources:
                rtype = resource['type']
                if rtype not in resource_types:
                    resource_types[rtype] = []
                resource_types[rtype].append(resource)
            
            for rtype, resources in resource_types.items():
                with st.expander(f"📚 {rtype.title()}s ({len(resources)})", expanded=True):
                    for resource in resources:
                        st.markdown(f"""
                        **{resource['title']}** (for {resource['skill']})
                        - Duration: {resource['duration']}
                        - Difficulty: {resource['difficulty']}
                        - Hours: {resource['estimated_hours']}
                        - Cost: {resource['cost']}
                        """)
                        st.divider()
        else:
            st.info("No learning resources available.")
    
    # Download results
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Run New Analysis"):
            st.session_state.analysis_result = None
            st.rerun()
    with col2:
        results_json = json.dumps({
            "student_name": st.session_state.student_name,
            "target_role": st.session_state.target_role,
            **result
        }, indent=2)
        st.download_button(
            "💾 Download Results (JSON)",
            results_json,
            file_name="skill_gap_analysis.json",
            mime="application/json"
        )

else:
    # Welcome screen
    st.info("👈 Please fill in your student profile and target role in the sidebar, then click 'Analyze Skill Gaps' to get started.")
    
    # Show example
    with st.expander("📖 How it works"):
        st.markdown("""
        ### How Skill Gap Analysis Works:
        
        1. **Input Your Profile**: Enter your current skills with proficiency levels
        2. **Define Target Role**: Specify the role you want and its required skills
        3. **Analysis**: The system compares your skills vs. requirements
        4. **Gap Detection**: Identifies missing skills and proficiency gaps
        5. **Learning Pathways**: Generates 3 personalized learning paths
        6. **Resources**: Suggests courses, tutorials, and projects
        7. **Recommendations**: Provides actionable next steps
        
        ### Gap Severity Levels:
        - 🔴 **Critical**: Skill missing or 3+ level gap
        - 🟠 **High**: 2 level gap
        - 🟡 **Medium**: 1 level gap  
        - 🟢 **Low**: Meets or exceeds requirement
        """)

