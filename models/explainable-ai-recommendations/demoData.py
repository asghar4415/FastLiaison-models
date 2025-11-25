"""
Enhanced Variant Data Generator (Option B)
- Single-file generator that:
  * Uses FAST - NUCES Karachi as the single university
  * Integrates expanded skills, companies, project templates, and courses
  * Produces students, skills, projects, courses, companies, jobs
  * Produces synthetic matching scores for student-job pairs (0-100)
  * Exports JSON and SQL (INSERT statements)
Run directly: python enhanced_variant_generator.py
"""

import json
import random
from datetime import datetime, timedelta
from uuid import uuid4
from math import floor

random.seed(42)  # stable randomness; change/remove for varied runs

# -----------------------------
# CONFIG
# -----------------------------
NUM_STUDENTS = 120        # total students to generate
JOBS_PER_COMPANY = 6      # jobs per company
NOISE_LEVEL = 6.0         # noise in matching score (0-15 recommended)


# -----------------------------
# GENERATOR CLASS
# -----------------------------
class EnhancedVariantDataGenerator:
    def __init__(self):
        self.skills_db = self._init_skills()
        self.company_templates = self._init_companies()
        self.project_templates = self._init_project_templates()
        self.course_templates = self._init_course_templates()

    # -------------------------
    # MASTER SKILLS
    # -------------------------
    def _init_skills(self):
        """Initialize comprehensive skill master data"""
        skills = {
            'Programming': [
                'Python', 'JavaScript', 'Java', 'C++', 'C#', 'Go', 'Rust',
                'TypeScript', 'PHP', 'Ruby', 'Kotlin', 'Swift', 'Dart'
            ],
            'Web Development': [
                'React', 'Angular', 'Vue.js', 'Node.js', 'Django', 'Flask',
                'Express.js', 'FastAPI', 'Spring Boot', 'ASP.NET', 'Next.js',
                'HTML5', 'CSS3', 'SASS', 'Tailwind CSS', 'Bootstrap'
            ],
            'Database': [
                'SQL', 'PostgreSQL', 'MySQL', 'MongoDB', 'Redis', 'Cassandra',
                'Firebase', 'DynamoDB', 'Oracle', 'SQL Server', 'NoSQL'
            ],
            'Data Science & ML': [
                'Machine Learning', 'Deep Learning', 'NLP', 'Computer Vision',
                'Data Visualization', 'Statistical Analysis', 'Pandas', 'NumPy',
                'TensorFlow', 'PyTorch', 'Scikit-learn', 'Keras', 'XGBoost',
                'Tableau', 'Power BI', 'Excel', 'R', 'Jupyter', 'Matplotlib', 'Seaborn'
            ],
            'Cloud & DevOps': [
                'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'CI/CD',
                'Jenkins', 'GitLab CI', 'GitHub Actions', 'Terraform',
                'Ansible', 'Linux', 'Shell Scripting', 'Nginx', 'Apache'
            ],
            'Mobile Development': [
                'React Native', 'Flutter', 'Android', 'iOS', 'SwiftUI',
                'Jetpack Compose', 'Firebase Mobile', 'Mobile UI/UX'
            ],
            'Tools & Platforms': [
                'Git', 'GitHub', 'GitLab', 'Jira', 'Postman', 'VS Code',
                'IntelliJ', 'Figma', 'Adobe XD', 'Slack', 'Trello'
            ],
            'Soft Skills': [
                'Communication', 'Team Leadership', 'Problem Solving',
                'Project Management', 'Critical Thinking', 'Agile Methodology',
                'Time Management', 'Collaboration', 'Presentation Skills',
                'Technical Writing', 'Client Management', 'Mentoring'
            ],
            'Security': [
                'Cybersecurity', 'Network Security', 'Penetration Testing',
                'OWASP', 'Authentication', 'Encryption', 'Security Auditing'
            ],
            'Specialized': [
                'Blockchain', 'IoT', 'AR/VR', 'Game Development',
                'UI/UX Design', 'API Development', 'Microservices',
                'System Design', 'Algorithm Design', 'Data Structures'
            ]
        }

        skill_master = []
        for category, skill_list in skills.items():
            for skill in skill_list:
                skill_master.append({
                    'skill_master_id': str(uuid4()),
                    'name': skill,
                    'category': category
                })

        return skill_master

    # -------------------------
    # COMPANIES
    # -------------------------
    def _init_companies(self):
        """Initialize diverse company data"""
        return [
            {'name': 'Systems Limited', 'industry': 'Software Development', 'size': 'Large'},
            {'name': 'NetSol Technologies', 'industry': 'Enterprise Software', 'size': 'Large'},
            {'name': 'TPS Worldwide', 'industry': 'Payment Solutions', 'size': 'Enterprise'},
            {'name': 'Folio3 Software', 'industry': 'Custom Software', 'size': 'Medium'},
            {'name': 'Arbisoft', 'industry': 'AI & ML Solutions', 'size': 'Medium'},
            {'name': 'Careem', 'industry': 'Technology/Ride-hailing', 'size': 'Large'},
            {'name': 'Bykea', 'industry': 'Logistics Tech', 'size': 'Startup'},
            {'name': 'Zameen.com', 'industry': 'PropTech', 'size': 'Medium'},
            {'name': 'Daraz', 'industry': 'E-commerce', 'size': 'Large'},
            {'name': 'Jazz xlr8', 'industry': 'Telecom/Innovation', 'size': 'Enterprise'},
            {'name': 'Inbox Business Technologies', 'industry': 'FinTech', 'size': 'Medium'},
            {'name': 'Tech Valley', 'industry': 'IT Services', 'size': 'Startup'},
            {'name': 'CloudPak Solutions', 'industry': 'Cloud Computing', 'size': 'Medium'},
            {'name': 'DataCorp Analytics', 'industry': 'Data Science', 'size': 'Startup'},
            {'name': 'CyberSec Pakistan', 'industry': 'Cybersecurity', 'size': 'Small'}
        ]

    # -------------------------
    # PROJECT TEMPLATES
    # -------------------------
    def _init_project_templates(self):
        """Initialize diverse project templates"""
        return [
            {
                'title': 'E-commerce Platform with Payment Integration',
                'skills': ['Python', 'Django', 'React', 'PostgreSQL', 'AWS', 'API Development'],
                'complexity': 'high'
            },
            {
                'title': 'Real-time Chat Application',
                'skills': ['Node.js', 'Socket.io', 'React', 'MongoDB', 'Redis'],
                'complexity': 'medium'
            },
            {
                'title': 'Machine Learning Price Prediction Model',
                'skills': ['Python', 'Scikit-learn', 'Pandas', 'Data Visualization', 'Jupyter'],
                'complexity': 'high'
            },
            {
                'title': 'Mobile Food Delivery App',
                'skills': ['React Native', 'Node.js', 'MongoDB', 'Firebase', 'Mobile UI/UX'],
                'complexity': 'high'
            },
            {
                'title': 'Automated Testing Framework',
                'skills': ['Python', 'Selenium', 'CI/CD', 'Jenkins', 'Docker'],
                'complexity': 'medium'
            },
            {
                'title': 'Data Analytics Dashboard',
                'skills': ['Python', 'Tableau', 'SQL', 'Data Visualization', 'Statistical Analysis'],
                'complexity': 'medium'
            },
            {
                'title': 'IoT Smart Home System',
                'skills': ['Python', 'IoT', 'Raspberry Pi', 'MQTT', 'Mobile App'],
                'complexity': 'high'
            },
            {
                'title': 'Social Media Analytics Tool',
                'skills': ['Python', 'NLP', 'Data Science', 'Flask', 'MongoDB'],
                'complexity': 'medium'
            },
            {
                'title': 'Cloud Infrastructure Automation',
                'skills': ['AWS', 'Terraform', 'Python', 'Docker', 'Linux'],
                'complexity': 'high'
            },
            {
                'title': 'Personal Finance Tracker',
                'skills': ['React', 'Node.js', 'PostgreSQL', 'Chart.js'],
                'complexity': 'low'
            },
            {
                'title': 'Student Management System',
                'skills': ['Java', 'Spring Boot', 'MySQL', 'Angular'],
                'complexity': 'medium'
            },
            {
                'title': 'Image Recognition System',
                'skills': ['Python', 'TensorFlow', 'Computer Vision', 'Flask', 'OpenCV'],
                'complexity': 'high'
            },
            {
                'title': 'RESTful API for Booking System',
                'skills': ['Node.js', 'Express.js', 'MongoDB', 'API Development', 'Postman'],
                'complexity': 'medium'
            },
            {
                'title': 'Blockchain-based Voting System',
                'skills': ['Blockchain', 'Solidity', 'Web3', 'React', 'Node.js'],
                'complexity': 'high'
            },
            {
                'title': 'Weather Prediction Dashboard',
                'skills': ['Python', 'Machine Learning', 'Flask', 'Data Visualization'],
                'complexity': 'low'
            }
        ]

    # -------------------------
    # COURSE TEMPLATES
    # -------------------------
    def _init_course_templates(self):
        """Initialize comprehensive course templates"""
        return [
            {'name': 'Data Structures', 'code': 'CS201', 'credits': 4, 'difficulty': 'medium'},
            {'name': 'Algorithms', 'code': 'CS301', 'credits': 4, 'difficulty': 'hard'},
            {'name': 'Database Systems', 'code': 'CS401', 'credits': 3, 'difficulty': 'medium'},
            {'name': 'Machine Learning', 'code': 'CS512', 'credits': 3, 'difficulty': 'hard'},
            {'name': 'Web Development', 'code': 'CS305', 'credits': 3, 'difficulty': 'easy'},
            {'name': 'Software Engineering', 'code': 'CS402', 'credits': 3, 'difficulty': 'medium'},
            {'name': 'Cloud Computing', 'code': 'CS515', 'credits': 3, 'difficulty': 'medium'},
            {'name': 'Data Science', 'code': 'CS601', 'credits': 3, 'difficulty': 'hard'},
            {'name': 'Mobile App Development', 'code': 'CS403', 'credits': 3, 'difficulty': 'medium'},
            {'name': 'Computer Networks', 'code': 'CS302', 'credits': 3, 'difficulty': 'medium'},
            {'name': 'Operating Systems', 'code': 'CS303', 'credits': 4, 'difficulty': 'hard'},
            {'name': 'Artificial Intelligence', 'code': 'CS511', 'credits': 3, 'difficulty': 'hard'},
            {'name': 'Cybersecurity', 'code': 'CS505', 'credits': 3, 'difficulty': 'hard'},
            {'name': 'Object Oriented Programming', 'code': 'CS202', 'credits': 4, 'difficulty': 'medium'},
            {'name': 'Theory of Computation', 'code': 'CS304', 'credits': 3, 'difficulty': 'hard'},
            {'name': 'Computer Architecture', 'code': 'CS203', 'credits': 3, 'difficulty': 'medium'},
            {'name': 'Human Computer Interaction', 'code': 'CS404', 'credits': 3, 'difficulty': 'easy'},
            {'name': 'Distributed Systems', 'code': 'CS602', 'credits': 3, 'difficulty': 'hard'},
            {'name': 'Natural Language Processing', 'code': 'CS513', 'credits': 3, 'difficulty': 'hard'},
            {'name': 'Digital Image Processing', 'code': 'CS506', 'credits': 3, 'difficulty': 'medium'},
            {'name': 'Big Data Analytics', 'code': 'CS603', 'credits': 3, 'difficulty': 'hard'},
            {'name': 'DevOps Practices', 'code': 'CS507', 'credits': 3, 'difficulty': 'medium'},
            {'name': 'Parallel Computing', 'code': 'CS604', 'credits': 3, 'difficulty': 'hard'},
            {'name': 'Information Security', 'code': 'CS405', 'credits': 3, 'difficulty': 'medium'},
            {'name': 'Compiler Construction', 'code': 'CS406', 'credits': 3, 'difficulty': 'hard'}
        ]

    # -------------------------
    # UNIVERSITY & DEPARTMENTS
    # -------------------------
    def generate_university(self):
        """Generate FAST NUCES Karachi"""
        return {
            'university_id': str(uuid4()),
            'name': 'FAST - National University of Computer and Emerging Sciences',
            'code': 'FAST-KHI',
            'location': 'Karachi',
            'campus': 'Karachi Campus'
        }

    def generate_departments(self, university_id):
        """Generate CS departments at FAST"""
        departments = [
            {'name': 'Computer Science', 'code': 'CS'},
            {'name': 'Software Engineering', 'code': 'SE'},
            {'name': 'Data Science', 'code': 'DS'},
            {'name': 'Artificial Intelligence', 'code': 'AI'},
            {'name': 'Cyber Security', 'code': 'CYS'}
        ]

        dept_list = []
        for dept in departments:
            dept_list.append({
                'dept_id': str(uuid4()),
                'university_id': university_id,
                'name': dept['name'],
                'code': dept['code']
            })
        return dept_list

    # -------------------------
    # STUDENTS
    # -------------------------
    def generate_students(self, university_id, departments, count=NUM_STUDENTS):
        """Generate diverse student profiles with realistic distributions"""
        students = []

        # Archetype distribution: probability-based
        archetypes = [
            # Excellent students (20%)
            {
                'label': 'Excellent',
                'prob': 0.20,
                'cgpa_range': (3.7, 4.0),
                'skills_range': (10, 15),
                'projects_range': (4, 8),
                'courses_range': (20, 28),
                'skill_level_weights': (0.6, 0.3, 0.1)  # expert, advanced, intermediate
            },
            # Strong students (30%)
            {
                'label': 'Strong',
                'prob': 0.30,
                'cgpa_range': (3.3, 3.69),
                'skills_range': (8, 12),
                'projects_range': (3, 6),
                'courses_range': (18, 24),
                'skill_level_weights': (0.3, 0.5, 0.2)
            },
            # Good students (30%)
            {
                'label': 'Good',
                'prob': 0.30,
                'cgpa_range': (3.0, 3.29),
                'skills_range': (6, 10),
                'projects_range': (2, 5),
                'courses_range': (16, 22),
                'skill_level_weights': (0.1, 0.5, 0.4)
            },
            # Average students (15%)
            {
                'label': 'Average',
                'prob': 0.15,
                'cgpa_range': (2.7, 2.99),
                'skills_range': (4, 8),
                'projects_range': (1, 3),
                'courses_range': (14, 20),
                'skill_level_weights': (0.0, 0.4, 0.6)
            },
            # Developing students (5%)
            {
                'label': 'Developing',
                'prob': 0.05,
                'cgpa_range': (2.5, 2.69),
                'skills_range': (3, 6),
                'projects_range': (0, 2),
                'courses_range': (12, 18),
                'skill_level_weights': (0.0, 0.2, 0.8)
            }
        ]

        # Precompute cumulative probabilities
        probs = [a['prob'] for a in archetypes]
        names = [a['label'] for a in archetypes]

        first_names = ['Ahmed', 'Ali', 'Hassan', 'Huma', 'Sara', 'Zainab', 'Fatima',
                       'Muhammad', 'Usman', 'Bilal', 'Ayesha', 'Mariam', 'Hamza',
                       'Umer', 'Abdullah', 'Zara', 'Amna', 'Sana', 'Fahad', 'Irfan']
        last_names = ['Khan', 'Ahmed', 'Ali', 'Hussain', 'Sheikh', 'Malik', 'Siddiqui',
                      'Rizvi', 'Qureshi', 'Haider', 'Raza', 'Hassan', 'Jamil']

        for i in range(count):
            # choose archetype
            archetype = random.choices(archetypes, weights=probs, k=1)[0]
            cgpa = round(random.uniform(*archetype['cgpa_range']), 2)
            skills_count = random.randint(*archetype['skills_range'])
            projects_count = random.randint(*archetype['projects_range'])
            courses_count = random.randint(*archetype['courses_range'])
            dept = random.choice(departments)

            fname = random.choice(first_names)
            lname = random.choice(last_names)
            name = f"{fname} {lname}"

            student = {
                's_id': str(uuid4()),
                'university_id': university_id,
                'dept_id': dept['dept_id'],
                'name': name,
                'email': f"{fname.lower()}.{lname.lower()}{i}@fastn.edu.pk",
                'batch': random.choice([2021, 2022, 2023, 2024]),
                'semester': random.randint(5, 8),
                'cgpa': cgpa,
                'graduation_year': 2025,
                'is_verified': random.choice([True, False, True]),  # slightly skews True
                'is_profile_complete': random.choice([True, True, False]),
                'account_status': random.choice(['Active', 'Pending', 'Suspended']),
                # meta for training
                'archetype': archetype['label'],
                'desired_roles': random.sample(
                    ['Software Engineer', 'Data Scientist', 'Full Stack Developer',
                     'ML Engineer', 'Cloud Engineer', 'Mobile Developer', 'DevOps Engineer',
                     'Security Analyst', 'Frontend Developer'], k=2)
            }

            students.append((student, skills_count, projects_count, courses_count, archetype))

        # return list of tuples with counts and archetype for further processing
        return students

    # -------------------------
    # STUDENT SKILLS
    # -------------------------
    def generate_student_skills(self, student_tuple):
        """Generate student skills with per-student counts & archetype weights"""
        student, skills_count, _, _, archetype = student_tuple
        skill_master = self.skills_db

        # choose skills across categories by weighting toward programming/data/cloud for stronger archetypes
        chosen = set()
        attempts = 0
        while len(chosen) < skills_count and attempts < skills_count * 6:
            attempts += 1
            candidate = random.choice(skill_master)
            chosen.add(candidate['name'])

        proficiency_map = {
            'expert': ['Expert', 'Advanced'],
            'advanced': ['Advanced', 'Intermediate'],
            'intermediate': ['Intermediate', 'Beginner'],
            'beginner-intermediate': ['Intermediate', 'Beginner'],
            'beginner': ['Beginner']
        }

        # attempt to map archetype label to our proficiency options
        label = archetype['label'].lower()
        if 'excellent' in label:
            prof_choices = ['Expert', 'Advanced', 'Intermediate']
        elif 'strong' in label:
            prof_choices = ['Advanced', 'Intermediate']
        elif 'good' in label:
            prof_choices = ['Intermediate', 'Beginner']
        elif 'average' in label:
            prof_choices = ['Intermediate', 'Beginner']
        else:
            prof_choices = ['Beginner', 'Intermediate']

        student_skills = []
        for skill_name in chosen:
            master = next((s for s in skill_master if s['name'] == skill_name), None)
            if not master:
                continue

            # proficiency distribution influenced by archetype cgpa a little
            if student['cgpa'] >= 3.7:
                prof = random.choices(['Expert', 'Advanced', 'Intermediate'], weights=[0.4, 0.4, 0.2], k=1)[0]
            elif student['cgpa'] >= 3.3:
                prof = random.choices(['Advanced', 'Intermediate', 'Beginner'], weights=[0.4, 0.4, 0.2], k=1)[0]
            elif student['cgpa'] >= 3.0:
                prof = random.choices(['Intermediate', 'Beginner'], weights=[0.6, 0.4], k=1)[0]
            else:
                prof = random.choices(['Beginner', 'Intermediate'], weights=[0.75, 0.25], k=1)[0]

            s_skill = {
                'skill_id': str(uuid4()),
                's_id': student['s_id'],
                'skill_master_id': master['skill_master_id'],
                'skill_name': skill_name,
                'proficiency_level': prof,
                'test_score': round(random.uniform(55, 98) if prof in ['Advanced', 'Expert'] else random.uniform(30, 75), 2),
                'is_verified': random.choice([True, False, False]) if prof == 'Beginner' else random.choice([True, True, False]),
                'years_of_experience': round(random.uniform(0.0, 4.0) if prof == 'Beginner' else random.uniform(0.5, 5.0), 2)
            }

            student_skills.append(s_skill)

        return student_skills

    # -------------------------
    # PROJECTS
    # -------------------------
    def generate_projects_for_student(self, student_tuple):
        student, _, projects_count, _, archetype = student_tuple
        projects = []

        # map complexity preference by archetype
        if archetype['label'] == 'Excellent':
            weights = [0.6, 0.3, 0.1]  # high, medium, low
        elif archetype['label'] == 'Strong':
            weights = [0.4, 0.4, 0.2]
        elif archetype['label'] == 'Good':
            weights = [0.25, 0.5, 0.25]
        else:
            weights = [0.1, 0.5, 0.4]

        for i in range(projects_count):
            template = random.choices(self.project_templates, k=1)[0]
            # sometimes pick higher complexity than template to boost variance
            complexity = template.get('complexity', 'medium')
            p_id = str(uuid4())
            title = template['title']
            skills = template['skills']

            start_date = (datetime.now() - timedelta(days=random.randint(60, 900))).date().isoformat()
            end_date = (datetime.now() - timedelta(days=random.randint(0, 59))).date().isoformat()

            project = {
                'p_id': p_id,
                'std_id': student['s_id'],
                'title': title,
                'description': f"Implemented {title} using {', '.join(skills[:3])} and best practices",
                'start_date': start_date,
                'end_date': end_date,
                'github_link': f"https://github.com/{student['name'].replace(' ', '')}/{p_id[:8]}",
                'is_verified': random.choice([True, False]),
                'skills': skills,
                'complexity': complexity
            }
            projects.append(project)

        return projects

    # -------------------------
    # COURSES
    # -------------------------
    def generate_courses_for_student(self, student_tuple):
        student, _, _, courses_count, archetype = student_tuple
        courses = []
        # prefer courses aligned with department (rough heuristic)
        dept_code = student['dept_id']  # we don't have name easily here, so random selection
        for i in range(courses_count):
            ct = random.choice(self.course_templates)
            gpa = round(random.uniform(max(2.0, student['cgpa'] - 0.5), min(4.0, student['cgpa'] + 0.3)), 2)
            course = {
                'c_id': str(uuid4()),
                's_id': student['s_id'],
                'course_name': ct['name'],
                'course_code': ct['code'],
                'credit_hours': ct['credits'],
                'gpa': gpa,
                'grade': self._gpa_to_grade(gpa),
                'semester_taken': random.randint(1, student['semester']),
                'year_taken': random.randint(student['batch'] - 3, student['batch']),
                'is_verified': True
            }
            courses.append(course)

        return courses

    def _gpa_to_grade(self, gpa):
        """Convert GPA to letter grade"""
        if gpa >= 3.7: return 'A' if gpa < 4.0 else 'A+'
        elif gpa >= 3.3: return 'A-'
        elif gpa >= 3.0: return 'B+'
        elif gpa >= 2.7: return 'B'
        else: return 'B-'

    # -------------------------
    # COMPANIES & JOBS
    # -------------------------
    def generate_companies_and_jobs(self, companies_to_use=None, jobs_per_company=JOBS_PER_COMPANY):
        """Create companies and jobs. Jobs pick required skills from skill_master,
           set weight and mandatory flags; also set eligible departments and cgpa."""
        if companies_to_use is None:
            companies_to_use = self.company_templates

        companies = []
        jobs = []

        for comp in companies_to_use:
            cid = str(uuid4())
            company = {
                'company_id': cid,
                'name': comp['name'],
                'industry': comp['industry'],
                'size': comp['size'],
                'is_verified': True
            }
            companies.append(company)

            # create jobs for each company
            for _ in range(jobs_per_company):
                job_template_skills = self._sample_skills_for_job()
                j_id = str(uuid4())

                title = random.choice([
                    'Junior Software Engineer', 'Software Engineer', 'Full Stack Developer',
                    'Data Scientist', 'Machine Learning Engineer', 'Cloud Engineer',
                    'DevOps Engineer', 'Mobile Developer', 'Security Analyst', 'Frontend Developer'
                ])

                cgpa_min = round(random.choice([2.7, 3.0, 3.2, 3.3]), 2)
                salary_min = random.randint(30000, 70000)
                salary_max = salary_min + random.randint(10000, 70000)

                job = {
                    'j_id': j_id,
                    'company_id': cid,
                    'title': title,
                    'description': f"{title} at {comp['name']} in {comp['industry']}",
                    'job_type': random.choice(['Full-Time', 'Internship', 'Part-Time', 'Contract']),
                    'work_mode': random.choice(['Remote', 'Onsite', 'Hybrid']),
                    'salary_range_min': salary_min,
                    'salary_range_max': salary_max,
                    'experience_required': random.choice(['0-1 years', '1-2 years', '2-4 years']),
                    'eligible_batches': [2021, 2022, 2023, 2024],
                    'eligible_departments': ['Computer Science', 'Software Engineering', 'Data Science'],
                    'eligible_cgpa_min': cgpa_min,
                    'status': random.choice(['Open', 'Open', 'Closed']),
                    'application_deadline': (datetime.now() + timedelta(days=random.randint(10, 60))).isoformat(),
                    'required_skills': []
                }

                # populate required_skills with weights and mandatory flags
                for i, skill_name in enumerate(job_template_skills):
                    master = next((s for s in self.skills_db if s['name'] == skill_name), None)
                    if not master:
                        continue
                    job_skill = {
                        'job_skill_id': str(uuid4()),
                        'j_id': j_id,
                        'skill_master_id': master['skill_master_id'],
                        'skill_name': skill_name,
                        'required_level': random.choice(['Beginner', 'Intermediate', 'Advanced']),
                        'is_mandatory': i < max(1, int(len(job_template_skills) * 0.5)),  # first half mandatory
                        'weight': round(random.uniform(1.0, 2.0), 2)
                    }
                    job['required_skills'].append(job_skill)

                jobs.append(job)

        return companies, jobs

    def _sample_skills_for_job(self):
        """Sample 4-8 skills for a job, biased towards popular categories"""
        popular_cats = ['Programming', 'Web Development', 'Data Science & ML', 'Cloud & DevOps', 'Database']
        # Flatten skill names then pick
        names = [s['name'] for s in self.skills_db]
        count = random.randint(4, 8)
        chosen = random.sample(names, k=min(count, len(names)))
        return chosen

    # -------------------------
    # MATCHING SCORE GENERATION
    # -------------------------
    def generate_matching_scores(self, students, student_skills_map, student_projects_map, student_courses_map, jobs):
        """Produce a matching score (0-100) for each student-job pair."""
        matches = []
        for student in students:
            s_id = student['s_id']
            s_skills = student_skills_map.get(s_id, [])
            s_skill_names = [sk['skill_name'] for sk in s_skills]
            s_skill_dict = {sk['skill_name']: sk for sk in s_skills}
            s_projects = student_projects_map.get(s_id, [])
            s_courses = student_courses_map.get(s_id, [])

            for job in jobs:
                # base score from skill overlap
                overlap = 0.0
                mandatory_penalty = 0.0
                total_weight = 0.0

                for req in job['required_skills']:
                    weight = req.get('weight', 1.0)
                    total_weight += weight
                    skill = req['skill_name']
                    if skill in s_skill_names:
                        sk = s_skill_dict[skill]
                        # proficiency mapping to numeric
                        prof = sk.get('proficiency_level', 'Beginner')
                        prof_score = {'Beginner': 0.6, 'Intermediate': 0.8, 'Advanced': 1.0, 'Expert': 1.1}.get(prof, 0.7)
                        verify_bonus = 1.05 if sk.get('is_verified') else 1.0
                        exp_bonus = min(1.2, 1.0 + (sk.get('years_of_experience', 0.0) / 5.0))
                        overlap += weight * prof_score * verify_bonus * exp_bonus
                    else:
                        # missing mandatory skill reduces score more
                        if req.get('is_mandatory'):
                            mandatory_penalty += weight * 0.6
                        else:
                            mandatory_penalty += weight * 0.2

                # normalize skill score to 0-60 range
                if total_weight <= 0:
                    skill_score = 0.0
                else:
                    raw_skill_score = max(0.0, overlap - mandatory_penalty)
                    skill_score = (raw_skill_score / (total_weight * 1.2)) * 60.0
                    skill_score = max(0.0, min(60.0, skill_score))

                # project relevance (0-15)
                proj_score = 0.0
                if s_projects:
                    job_skill_set = set([r['skill_name'] for r in job['required_skills']])
                    for proj in s_projects:
                        proj_skills = set(proj['skills'])
                        common = len(job_skill_set.intersection(proj_skills))
                        if common > 0:
                            # more complex projects contribute more
                            complexity = proj.get('complexity', 'medium')
                            complexity_mult = {'low': 1.0, 'medium': 1.4, 'high': 1.8}.get(complexity, 1.2)
                            proj_score += min(3.0 * complexity_mult, common * 1.5 * complexity_mult)
                    proj_score = min(15.0, proj_score)

                # course relevance (0-10)
                course_score = 0.0
                job_sk_names = set([r['skill_name'] for r in job['required_skills']])
                for c in s_courses:
                    # if course name or code has words matching skills, give credit - simple heuristic
                    cname = c['course_name'].lower()
                    for skill in job_sk_names:
                        if skill.lower() in cname or any(part.lower() in cname for part in skill.split()):
                            course_score += 0.6
                course_score = min(10.0, course_score)

                # cgpa contribution (0-10)
                cgpa_contrib = 0.0
                if student.get('cgpa') >= job.get('eligible_cgpa_min', 0):
                    cgpa_contrib = min(10.0, (student.get('cgpa') / 4.0) * 10.0)

                # minor random noise and soft handling of application_deadline and job status
                status_mult = 1.0 if job.get('status') == 'Open' else 0.8
                noise = random.gauss(0, NOISE_LEVEL)  # gaussian noise
                raw_score = skill_score + proj_score + course_score + cgpa_contrib
                raw_score = raw_score * status_mult + noise

                # clamp
                final_score = round(max(0.0, min(100.0, raw_score)), 2)

                match = {
                    'match_id': str(uuid4()),
                    's_id': s_id,
                    'j_id': job['j_id'],
                    'company_id': job['company_id'],
                    'score': final_score,
                    'generated_at': datetime.now().isoformat()
                }
                matches.append(match)

        return matches

    # -------------------------
    # FULL DATASET GENERATION
    # -------------------------
    def generate_complete_dataset(self, num_students=NUM_STUDENTS, jobs_per_company=JOBS_PER_COMPANY):
        print("🚀 Generating enhanced demo data (Option B) ...\n")
        # University
        uni = self.generate_university()
        print("Generated university:", uni['name'])

        # Departments
        depts = self.generate_departments(uni['university_id'])
        print(f"Generated {len(depts)} departments")

        # Students (returns list of tuples with meta)
        students_with_meta = self.generate_students(uni['university_id'], depts, count=num_students)
        students = [t[0] for t in students_with_meta]
        print(f"Generated {len(students)} students (archetype distribution applied)")

        # Student skills / projects / courses
        all_student_skills = []
        all_projects = []
        all_courses = []
        student_skills_map = {}
        student_projects_map = {}
        student_courses_map = {}

        for entry in students_with_meta:
            student = entry[0]
            # skills
            sks = self.generate_student_skills(entry)
            all_student_skills.extend(sks)
            student_skills_map[student['s_id']] = sks
            # projects
            prs = self.generate_projects_for_student(entry)
            all_projects.extend(prs)
            student_projects_map[student['s_id']] = prs
            # courses
            crs = self.generate_courses_for_student(entry)
            all_courses.extend(crs)
            student_courses_map[student['s_id']] = crs

        print(f"Generated {len(all_student_skills)} student skill entries")
        print(f"Generated {len(all_projects)} projects")
        print(f"Generated {len(all_courses)} course records")

        # Companies and Jobs
        companies, jobs = self.generate_companies_and_jobs(self.company_templates, jobs_per_company)
        print(f"Generated {len(companies)} companies")
        print(f"Generated {len(jobs)} job postings")

        # Matching scores
        print("Generating synthetic matching scores (labels) ...")
        matches = self.generate_matching_scores(students, student_skills_map, student_projects_map, student_courses_map, jobs)
        print(f"Generated {len(matches)} student-job match records")

        dataset = {
            'universities': [uni],
            'departments': depts,
            'students': students,
            'student_skills': all_student_skills,
            'projects': all_projects,
            'courses': all_courses,
            'skill_master': self.skills_db,
            'companies': companies,
            'jobs': jobs,
            'matches': matches
        }

        return dataset

    # -------------------------
    # EXPORT UTILITIES
    # -------------------------
    def export_to_json(self, dataset, filename='enhanced_demo_data.json'):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, default=str, indent=2)
        print(f"💾 Exported JSON -> {filename}")

    def export_to_sql(self, dataset, filename='enhanced_demo_data.sql'):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("-- Enhanced Demo Data (Option B)\n\n")
            # Write utility to format values
            def fmt(v):
                if v is None:
                    return 'NULL'
                if isinstance(v, bool):
                    return 'TRUE' if v else 'FALSE'
                if isinstance(v, (int, float)):
                    return str(v)
                # escape single quotes
                return "'" + str(v).replace("'", "''") + "'"

            # university
            for u in dataset['universities']:
                cols = ', '.join(u.keys())
                vals = ', '.join(fmt(v) for v in u.values())
                f.write(f"INSERT INTO University ({cols}) VALUES ({vals});\n")

            # departments
            for d in dataset['departments']:
                cols = ', '.join(d.keys())
                vals = ', '.join(fmt(v) for v in d.values())
                f.write(f"INSERT INTO Department ({cols}) VALUES ({vals});\n")

            # skill_master
            for s in dataset['skill_master']:
                cols = ', '.join(s.keys())
                vals = ', '.join(fmt(v) for v in s.values())
                f.write(f"INSERT INTO Skill_Master ({cols}) VALUES ({vals});\n")

            # students
            for s in dataset['students']:
                cols = ', '.join(s.keys())
                vals = ', '.join(fmt(v) for v in s.values())
                f.write(f"INSERT INTO Student ({cols}) VALUES ({vals});\n")

            # student_skills
            for sk in dataset['student_skills']:
                cols = ', '.join(sk.keys())
                vals = ', '.join(fmt(v) for v in sk.values())
                f.write(f"INSERT INTO Student_Skills ({cols}) VALUES ({vals});\n")

            # projects
            for p in dataset['projects']:
                # store skills as json string
                rec = dict(p)
                rec['skills'] = json.dumps(rec.get('skills', []))
                cols = ', '.join(rec.keys())
                vals = ', '.join(fmt(v) for v in rec.values())
                f.write(f"INSERT INTO Projects ({cols}) VALUES ({vals});\n")

            # courses
            for c in dataset['courses']:
                cols = ', '.join(c.keys())
                vals = ', '.join(fmt(v) for v in c.values())
                f.write(f"INSERT INTO Courses ({cols}) VALUES ({vals});\n")

            # companies
            for c in dataset['companies']:
                cols = ', '.join(c.keys())
                vals = ', '.join(fmt(v) for v in c.values())
                f.write(f"INSERT INTO Company ({cols}) VALUES ({vals});\n")

            # jobs and job skills
            for j in dataset['jobs']:
                jrec = dict(j)
                # extract required_skills array to separate table
                reqs = jrec.pop('required_skills', [])
                cols = ', '.join(jrec.keys())
                vals = ', '.join(fmt(v) for v in jrec.values())
                f.write(f"INSERT INTO Jobs ({cols}) VALUES ({vals});\n")
                for r in reqs:
                    cols_r = ', '.join(r.keys())
                    vals_r = ', '.join(fmt(v) for v in r.values())
                    f.write(f"INSERT INTO Job_Skills ({cols_r}) VALUES ({vals_r});\n")

            # matches
            for m in dataset.get('matches', []):
                cols = ', '.join(m.keys())
                vals = ', '.join(fmt(v) for v in m.values())
                f.write(f"INSERT INTO Matches ({cols}) VALUES ({vals});\n")

        print(f"💾 Exported SQL -> {filename}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    generator = EnhancedVariantDataGenerator()
    dataset = generator.generate_complete_dataset(num_students=NUM_STUDENTS, jobs_per_company=JOBS_PER_COMPANY)

    # Export to disk
    generator.export_to_json(dataset, filename='enhanced_demo_data.json')
    generator.export_to_sql(dataset, filename='enhanced_demo_data.sql')

    # Summary
    print("\n" + "="*50)
    print("📊 DATASET SUMMARY")
    print("="*50)
    print(f"Students: {len(dataset['students'])}")
    print(f"Student Skill Entries: {len(dataset['student_skills'])}")
    print(f"Projects: {len(dataset['projects'])}")
    print(f"Courses: {len(dataset['courses'])}")
    print(f"Companies: {len(dataset['companies'])}")
    print(f"Jobs: {len(dataset['jobs'])}")
    print(f"Matches (student-job pairs): {len(dataset.get('matches', []))}")
    print("✨ Data generation complete!")
