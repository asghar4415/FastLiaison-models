from datetime import datetime

class ExplainableJobMatcher:

    def __init__(self, student_id, job_id):
        self.student = self.get_student_profile(student_id)
        self.job = self.get_job_details(job_id)
        self.weights = {
            'skills': 0.40,
            'education': 0.20,
            'projects': 0.15,
            'courses': 0.15,
            'cgpa': 0.10
        }
        self.graduation_year_multiplier = self.calculate_graduation_year_multiplier()

    def generate_match_with_explanation(self):
        """
        Main function to generate match score and detailed explanation
        """
        scores = self.calculate_component_scores()
        overall_score = self.calculate_weighted_score(scores)
        explanation = self.build_explanation(scores)
        recommendation_type = self.classify_recommendation(overall_score, scores)

        return {
            'match_score': overall_score,
            'recommendation_type': recommendation_type,
            'explanation': explanation,
            'scores_breakdown': scores
        }

    def calculate_component_scores(self):
        """
        Calculate individual component scores
        """
        return {
            'skills': self.score_skills(),
            'education': self.score_education(),
            'projects': self.score_projects(),
            'courses': self.score_courses(),
            'cgpa': self.score_cgpa()
        }
    
    def calculate_graduation_year_multiplier(self):
        """
        Calculate a multiplier based on graduation year/batch to prioritize seniors.
        Earlier batch years (seniors) get higher multipliers.
        
        Batch year ranges (assuming current year is 2025):
        - 2021 batch (Senior/4th year+): 1.15x multiplier
        - 2022 batch (3rd year): 1.10x multiplier
        - 2023 batch (2nd year): 1.05x multiplier
        - 2024 batch (Freshman/1st year): 1.00x multiplier (base)
        """
        current_year = datetime.now().year  # You can use datetime.now().year for dynamic year
        batch_year = self.student.get('batch', current_year)
        
        # Calculate years in program (higher = more senior)
        years_in_program = current_year - batch_year
        
        # Define multipliers based on seniority
        if years_in_program >= 4:  # Senior (4th year or higher)
            return 1.15
        elif years_in_program == 3:  # 3rd year
            return 1.10
        elif years_in_program == 2:  # 2nd year
            return 1.05
        else:  # Freshman or 1st year
            return 1.00

    def score_skills(self):
        """
        Score skills match with detailed breakdown
        """
        required_skills = self.get_job_required_skills()
        student_skills = self.get_student_skills()

        matched_skills = []
        missing_skills = []
        proficiency_matches = []

        for req_skill in required_skills:
            student_skill = next(
                (s for s in student_skills if s['skill_id'] == req_skill['skill_id']),
                None
            )

            if student_skill:
                proficiency_match = self.compare_proficiency(
                    student_skill['proficiency_level'],
                    req_skill['required_level']
                )
                matched_skills.append({
                    'name': req_skill['name'],
                    'required_level': req_skill['required_level'],
                    'student_level': student_skill['proficiency_level'],
                    'proficiency_match': proficiency_match,
                    'is_verified': student_skill['is_verified'],
                    'weight': req_skill['weight']
                })
                proficiency_matches.append(proficiency_match)
            else:
                missing_skills.append({
                    'name': req_skill['name'],
                    'required_level': req_skill['required_level'],
                    'is_mandatory': req_skill['is_mandatory']
                })

        # Calculate score
        total_weight = sum(s['weight'] for s in required_skills)
        matched_weight = sum(
            skill['weight'] * skill['proficiency_match']
            for skill in matched_skills
        )
        score = (matched_weight / total_weight) * 100 if total_weight > 0 else 0

        return {
            'score': round(score, 2),
            'matched_skills': matched_skills,
            'missing_skills': missing_skills,
            'match_percentage': len(matched_skills) / len(required_skills) * 100,
            'avg_proficiency_match': sum(proficiency_matches) / len(proficiency_matches) if proficiency_matches else 0
        }

    def score_education(self):
        """
        Score education eligibility
        """
        batch_eligible = self.student['batch'] in self.job['eligible_batches']
        dept_eligible = self.student['dept_id'] in self.job['eligible_departments']
        cgpa_eligible = self.student['cgpa'] >= self.job['eligible_cgpa_min']

        score = 0
        if batch_eligible: score += 40
        if dept_eligible: score += 40
        if cgpa_eligible: score += 20

        return {
            'score': score,
            'batch_eligible': batch_eligible,
            'department_eligible': dept_eligible,
            'cgpa_eligible': cgpa_eligible,
            'cgpa_difference': self.student['cgpa'] - self.job['eligible_cgpa_min']
        }

    def score_projects(self):
        """
        Score relevant projects
        """
        student_projects = self.get_student_projects()
        job_skills = self.get_job_required_skills()

        relevant_projects = []
        for project in student_projects:
            project_skills = self.get_project_skills(project['p_id'])
            overlap = set(ps['skill_id'] for ps in project_skills).intersection(
                set(js['skill_id'] for js in job_skills)
            )

            if overlap:
                relevant_projects.append({
                    'title': project['title'],
                    'matched_skills_count': len(overlap),
                    'is_verified': project['is_verified']
                })

        score = min(len(relevant_projects) * 25, 100)

        return {
            'score': score,
            'relevant_projects_count': len(relevant_projects),
            'projects': relevant_projects,
            'total_projects': len(student_projects)
        }

    def score_courses(self):
        """
        Score relevant coursework
        """
        student_courses = self.get_student_courses()
        # Simple keyword matching or more sophisticated NLP
        relevant_courses = self.find_relevant_courses(
            student_courses,
            self.job['title'],
            self.job['description']
        )

        avg_grade = self.calculate_average_grade(relevant_courses)
        score = (len(relevant_courses) * 20 + avg_grade * 20)

        return {
            'score': min(score, 100),
            'relevant_courses_count': len(relevant_courses),
            'courses': [c['course_name'] for c in relevant_courses],
            'average_grade': avg_grade
        }

    def score_cgpa(self):
        """
        Score CGPA performance
        """
        min_cgpa = self.job['eligible_cgpa_min'] or 0
        student_cgpa = self.student['cgpa']

        if student_cgpa < min_cgpa:
            score = 0
        else:
            # Normalize: exceeding minimum gives higher scores
            score = min((student_cgpa / 4.0) * 100, 100)

        return {
            'score': round(score, 2),
            'student_cgpa': student_cgpa,
            'required_cgpa': min_cgpa,
            'exceeds_by': student_cgpa - min_cgpa
        }

    def build_explanation(self, scores):
        """
        Generate human-readable explanation
        """
        primary_reasons = []
        supporting_reasons = []
        concerns = []

        # Skills explanation
        if scores['skills']['score'] >= 70:
            matched = [s['name'] for s in scores['skills']['matched_skills'][:5]]
            primary_reasons.append({
                'icon': '🎯',
                'title': 'Strong Skills Match',
                'description': f"You have {len(scores['skills']['matched_skills'])} out of {len(scores['skills']['matched_skills']) + len(scores['skills']['missing_skills'])} required skills",
                'details': f"Matched skills: {', '.join(matched)}",
                'score_contribution': scores['skills']['score'] * self.weights['skills']
            })
        elif scores['skills']['missing_skills']:
            missing_mandatory = [s['name'] for s in scores['skills']['missing_skills'] if s['is_mandatory']]
            if missing_mandatory:
                concerns.append({
                    'icon': '⚠️',
                    'title': 'Missing Required Skills',
                    'description': f"You're missing {len(missing_mandatory)} mandatory skills",
                    'details': f"Missing: {', '.join(missing_mandatory[:3])}",
                    'action': 'Consider upskilling in these areas'
                })

        # Education explanation
        if scores['education']['score'] == 100:
            primary_reasons.append({
                'icon': '🎓',
                'title': 'Eligible Candidate',
                'description': 'You meet all education requirements',
                'details': f"Batch: ✓, Department: ✓, CGPA: {self.student['cgpa']} (required: {self.job['eligible_cgpa_min']})",
                'score_contribution': scores['education']['score'] * self.weights['education']
            })

        # Projects explanation
        if scores['projects']['relevant_projects_count'] > 0:
            projects = [p['title'] for p in scores['projects']['projects'][:3]]
            supporting_reasons.append({
                'icon': '💼',
                'title': 'Relevant Project Experience',
                'description': f"You have {scores['projects']['relevant_projects_count']} relevant projects",
                'details': f"Projects: {', '.join(projects)}",
                'score_contribution': scores['projects']['score'] * self.weights['projects']
            })

        # Courses explanation
        if scores['courses']['relevant_courses_count'] > 0:
            supporting_reasons.append({
                'icon': '📚',
                'title': 'Relevant Coursework',
                'description': f"Completed {scores['courses']['relevant_courses_count']} relevant courses",
                'details': f"Courses: {', '.join(scores['courses']['courses'][:3])}",
                'score_contribution': scores['courses']['score'] * self.weights['courses']
            })

        # CGPA explanation
        if scores['cgpa']['exceeds_by'] > 0.5:
            supporting_reasons.append({
                'icon': '⭐',
                'title': 'Strong Academic Performance',
                'description': f"Your CGPA ({self.student['cgpa']}) exceeds requirement by {scores['cgpa']['exceeds_by']:.2f}",
                'score_contribution': scores['cgpa']['score'] * self.weights['cgpa']
            })

        return {
            'primary_reasons': sorted(primary_reasons, key=lambda x: x['score_contribution'], reverse=True),
            'supporting_reasons': sorted(supporting_reasons, key=lambda x: x['score_contribution'], reverse=True),
            'concerns': concerns,
            'summary': self.generate_summary(primary_reasons, supporting_reasons, concerns)
        }

    def generate_summary(self, primary, supporting, concerns):
        """
        Generate a concise summary statement
        """
        if len(primary) >= 2:
            return f"Excellent match! {primary[0]['title']} and {primary[1]['title']}."
        elif len(primary) == 1 and len(supporting) >= 1:
            return f"Good match! {primary[0]['title']}, plus {len(supporting)} supporting factors."
        elif len(concerns) > 0:
            return f"Potential match with upskilling. Focus on: {concerns[0]['title']}."
        else:
            return "Moderate match. Review details for improvement areas."

    def classify_recommendation(self, overall_score, scores):
        """
        Classify recommendation type based on score and factors
        """
        if overall_score >= 85 and scores['skills']['score'] >= 80:
            return 'Perfect_Match'
        elif overall_score >= 70:
            return 'Good_Match'
        elif overall_score >= 50:
            return 'Potential_Match'
        else:
            return 'Upskill_Opportunity'