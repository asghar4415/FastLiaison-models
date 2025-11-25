import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import joblib
from datetime import datetime
import random



class SyntheticTrainingDataGenerator:
    """
    Generate training data by running your rule-based system
    on all student-job combinations
    """
    
    def __init__(self, rule_based_matcher_class):
        """
        Args:
            rule_based_matcher_class: Your ComponentBasedJobMatchingSystem class
        """
        self.rule_based_matcher = rule_based_matcher_class
    
    def generate_training_data_from_demo(self, demo_dataset):
        """
        Generate training data by matching all students with all jobs
        
        Args:
            demo_dataset: Output from DemoDataGenerator
        
        Returns:
            DataFrame with features and labels
        """
        
        print("🎯 Generating synthetic training data...")
        print(f"Students: {len(demo_dataset['students'])}")
        print(f"Jobs: {len(demo_dataset['jobs'])}")
        print(f"Total combinations: {len(demo_dataset['students']) * len(demo_dataset['jobs'])}\n")
        
        training_samples = []
        
        # Match every student with every job
        for student in demo_dataset['students']:
            for job in demo_dataset['jobs']:
                
                # Use your rule-based system to calculate match
                match_result = self.calculate_match_with_rules(
                    student, 
                    job,
                    demo_dataset
                )
                
                # Add realistic variations/noise
                match_result = self.add_realistic_variations(match_result)
                
                # Create training sample
                sample = {
                    # Identifiers
                    'student_id': student['s_id'],
                    'job_id': job['j_id'],
                    
                    # Features (what ML model will learn from)
                    'skills_score': match_result['component_scores']['skills']['score'],
                    'education_score': match_result['component_scores']['education']['score'],
                    'projects_score': match_result['component_scores']['projects']['score'],
                    'courses_score': match_result['component_scores']['courses']['score'],
                    'cgpa_score': match_result['component_scores']['cgpa']['score'],
                    
                    # Derived features
                    'matched_skills_count': len(match_result['component_scores']['skills']['matched_skills']),
                    'missing_skills_count': len(match_result['component_scores']['skills']['missing_skills']),
                    'mandatory_missing': sum(1 for s in match_result['component_scores']['skills']['missing_skills'] if s.get('is_mandatory', False)),
                    'cgpa_excess': match_result['component_scores']['cgpa']['exceeds_by'],
                    'graduation_year_multiplier': self.calculate_grad_multiplier(student),
                    
                    # Interaction features
                    'skills_x_education': match_result['component_scores']['skills']['score'] * match_result['component_scores']['education']['score'] / 100,
                    'projects_x_courses': match_result['component_scores']['projects']['score'] * match_result['component_scores']['courses']['score'] / 100,
                    
                    # Labels (what ML model will predict)
                    'match_score': match_result['match_score'],  # Target 1: Score
                    'recommendation_type': match_result['recommendation_type'],  # Target 2: Category
                    
                    # Simulated outcomes (realistic labels)
                    'would_apply': self.simulate_application_probability(match_result),
                    'would_get_interview': self.simulate_interview_probability(match_result),
                    'match_quality': self.simulate_match_quality(match_result),
                    
                    # Feedback engagement simulation
                    'needs_skills_feedback': len(match_result['component_scores']['skills']['missing_skills']) > 0,
                    'needs_education_feedback': match_result['component_scores']['education']['score'] < 100,
                    'has_projects_strength': match_result['component_scores']['projects']['score'] >= 60,
                }
                
                training_samples.append(sample)
        
        df = pd.DataFrame(training_samples)
        print(f"✅ Generated {len(df)} training samples")
        print(f"\nMatch Score Distribution:")
        print(df['match_score'].describe())
        print(f"\nRecommendation Types:")
        print(df['recommendation_type'].value_counts())
        
        return df
    
    def calculate_match_with_rules(self, student, job, dataset):
        """
        Use your rule-based system to calculate match
        This is your "ground truth generator"
        """
        
        # Get student skills
        student_skills = [s for s in dataset['student_skills'] if s['s_id'] == student['s_id']]
        
        # Get student projects
        student_projects = [p for p in dataset['projects'] if p['std_id'] == student['s_id']]
        
        # Get student courses
        student_courses = [c for c in dataset['courses'] if c['s_id'] == student['s_id']]
        
        # Calculate component scores using your logic
        skills_score = self.score_skills(student_skills, job['required_skills'])
        education_score = self.score_education(student, job)
        projects_score = self.score_projects(student_projects, job['required_skills'])
        courses_score = self.score_courses(student_courses, job)
        cgpa_score = self.score_cgpa(student, job)
        
        # Calculate weighted score (your rule-based weights)
        weights = {
            'skills': 0.40,
            'education': 0.20,
            'projects': 0.15,
            'courses': 0.15,
            'cgpa': 0.10
        }
        
        match_score = (
            skills_score['score'] * weights['skills'] +
            education_score['score'] * weights['education'] +
            projects_score['score'] * weights['projects'] +
            courses_score['score'] * weights['courses'] +
            cgpa_score['score'] * weights['cgpa']
        )
        
        # Apply graduation year multiplier
        grad_multiplier = self.calculate_grad_multiplier(student)
        match_score *= grad_multiplier
        match_score = min(match_score, 100)  # Cap at 100
        
        # Classify recommendation
        if match_score >= 85:
            rec_type = 'Perfect_Match'
        elif match_score >= 70:
            rec_type = 'Good_Match'
        elif match_score >= 50:
            rec_type = 'Potential_Match'
        else:
            rec_type = 'Upskill_Opportunity'
        
        return {
            'match_score': round(match_score, 2),
            'recommendation_type': rec_type,
            'component_scores': {
                'skills': skills_score,
                'education': education_score,
                'projects': projects_score,
                'courses': courses_score,
                'cgpa': cgpa_score
            }
        }
    
    def score_skills(self, student_skills, required_skills):
        """Score skills match"""
        matched_skills = []
        missing_skills = []
        
        for req_skill in required_skills:
            student_skill = next(
                (s for s in student_skills if s['skill_master_id'] == req_skill['skill_master_id']),
                None
            )
            
            if student_skill:
                # Calculate proficiency match
                proficiency_match = self.compare_proficiency(
                    student_skill['proficiency_level'],
                    req_skill['required_level']
                )
                matched_skills.append({
                    'name': req_skill['skill_name'],
                    'proficiency_match': proficiency_match,
                    'weight': req_skill['weight']
                })
            else:
                missing_skills.append({
                    'name': req_skill['skill_name'],
                    'is_mandatory': req_skill['is_mandatory']
                })
        
        # Calculate score
        total_weight = sum(s['weight'] for s in required_skills)
        matched_weight = sum(
            req['weight'] * next((m['proficiency_match'] for m in matched_skills if m['name'] == req['skill_name']), 0)
            for req in required_skills
        )
        score = (matched_weight / total_weight) * 100 if total_weight > 0 else 0
        
        return {
            'score': round(score, 2),
            'matched_skills': matched_skills,
            'missing_skills': missing_skills
        }
    
    def compare_proficiency(self, student_level, required_level):
        """Compare proficiency levels"""
        levels = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3, 'Expert': 4}
        student_num = levels.get(student_level, 0)
        required_num = levels.get(required_level, 0)
        
        if student_num >= required_num:
            return 1.0
        elif student_num == required_num - 1:
            return 0.7
        else:
            return 0.3
    
    def score_education(self, student, job):
        """Score education eligibility"""
        batch_eligible = student['batch'] in job['eligible_batches']
        dept_eligible = any(dept in student.get('dept_name', '') for dept in job['eligible_departments'])
        cgpa_eligible = student['cgpa'] >= job['eligible_cgpa_min']
        
        score = 0
        if batch_eligible: score += 40
        if dept_eligible: score += 40
        if cgpa_eligible: score += 20
        
        return {
            'score': score,
            'batch_eligible': batch_eligible,
            'department_eligible': dept_eligible,
            'cgpa_eligible': cgpa_eligible
        }
    
    def score_projects(self, student_projects, required_skills):
        """Score relevant projects"""
        relevant_projects = []
        
        for project in student_projects:
            # Check if project skills overlap with job skills
            project_skills = set(project.get('skills', []))
            required_skill_names = set(s['skill_name'] for s in required_skills)
            overlap = project_skills.intersection(required_skill_names)
            
            if overlap:
                relevant_projects.append({
                    'title': project['title'],
                    'matched_skills_count': len(overlap)
                })
        
        score = min(len(relevant_projects) * 25, 100)
        
        return {
            'score': score,
            'relevant_projects_count': len(relevant_projects),
            'projects': relevant_projects
        }
    
    def score_courses(self, student_courses, job):
        """Score relevant coursework"""
        # Simple keyword matching
        job_keywords = set(job['title'].lower().split() + job['description'].lower().split())
        relevant_courses = []
        
        for course in student_courses:
            course_keywords = set(course['course_name'].lower().split())
            if job_keywords.intersection(course_keywords):
                relevant_courses.append(course)
        
        avg_grade = np.mean([c['gpa'] for c in relevant_courses]) if relevant_courses else 0
        score = (len(relevant_courses) * 20 + avg_grade * 20)
        
        return {
            'score': min(score, 100),
            'relevant_courses_count': len(relevant_courses),
            'courses': [c['course_name'] for c in relevant_courses]
        }
    
    def score_cgpa(self, student, job):
        """Score CGPA performance"""
        min_cgpa = job['eligible_cgpa_min']
        student_cgpa = student['cgpa']
        
        if student_cgpa < min_cgpa:
            score = 0
        else:
            score = min((student_cgpa / 4.0) * 100, 100)
        
        return {
            'score': round(score, 2),
            'student_cgpa': student_cgpa,
            'required_cgpa': min_cgpa,
            'exceeds_by': student_cgpa - min_cgpa
        }
    
    def calculate_grad_multiplier(self, student):
        """Calculate graduation year multiplier"""
        current_year = datetime.now().year
        batch_year = student['batch']
        years_in_program = current_year - batch_year
        
        if years_in_program >= 4:
            return 1.15
        elif years_in_program == 3:
            return 1.10
        elif years_in_program == 2:
            return 1.05
        else:
            return 1.00
    
    def add_realistic_variations(self, match_result):
        """
        Add noise to make synthetic data more realistic
        Real-world data is never perfect!
        """
        
        # Add small random noise to scores (±5%)
        noise_factor = random.uniform(0.95, 1.05)
        match_result['match_score'] = min(
            match_result['match_score'] * noise_factor,
            100
        )
        
        return match_result
    
    def simulate_application_probability(self, match_result):
        """
        Simulate: Would student actually apply?
        Higher match score = higher probability
        """
        score = match_result['match_score']
        
        if score >= 80:
            prob = 0.9
        elif score >= 60:
            prob = 0.7
        elif score >= 40:
            prob = 0.4
        else:
            prob = 0.1
        
        # Add randomness
        return 1 if random.random() < prob else 0
    
    def simulate_interview_probability(self, match_result):
        """
        Simulate: Would student get interview?
        Based on match score and education eligibility
        """
        score = match_result['match_score']
        education_score = match_result['component_scores']['education']['score']
        
        if score >= 85 and education_score == 100:
            prob = 0.8
        elif score >= 70:
            prob = 0.5
        elif score >= 50:
            prob = 0.2
        else:
            prob = 0.05
        
        return 1 if random.random() < prob else 0
    
    def simulate_match_quality(self, match_result):
        """
        Simulate overall match quality (0-1 scale)
        This combines score with other factors
        """
        score = match_result['match_score'] / 100
        
        # Adjust based on component scores
        skills_score = match_result['component_scores']['skills']['score'] / 100
        education_score = match_result['component_scores']['education']['score'] / 100
        
        # Weighted combination with noise
        quality = (score * 0.6 + skills_score * 0.3 + education_score * 0.1)
        quality = quality * random.uniform(0.9, 1.1)  # Add noise
        
        return min(max(quality, 0), 1)  # Clamp to [0, 1]

