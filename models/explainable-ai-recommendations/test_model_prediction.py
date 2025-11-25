"""
Test the trained model with new student and job data
"""

import joblib
from model_usage import JobMatcherWithFeedback
from model import (EnhancedMLModelTrainer, EnhancedFeatureEngineering,
                   EnhancedMLModelTrainerWithFeedback, HumanizedFeedbackGenerator)


# ============================================================================
# DEMO DATA: 5 Students and 4 Jobs
# ============================================================================

DEMO_STUDENTS = [
    {
        's_id': 'STU001',
        'name': 'Alice Johnson',
        'cgpa': 3.9,
        'batch': 2021,
        'dept_name': 'Computer Science',
        'graduation_year': 2025,
        'skills': [
            {'skill_name': 'Python', 'proficiency_level': 'Expert'},
            {'skill_name': 'Machine Learning', 'proficiency_level': 'Advanced'},
            {'skill_name': 'TensorFlow', 'proficiency_level': 'Advanced'},
            {'skill_name': 'Deep Learning', 'proficiency_level': 'Advanced'},
            {'skill_name': 'SQL', 'proficiency_level': 'Intermediate'},
            {'skill_name': 'Docker', 'proficiency_level': 'Intermediate'},
        ],
        'projects': [
            {'title': 'AI-Powered Recommendation System', 'skills': ['Python', 'Machine Learning', 'TensorFlow', 'SQL']},
            {'title': 'Deep Learning Image Classifier', 'skills': ['Python', 'Deep Learning', 'TensorFlow']},
            {'title': 'ML Pipeline Automation', 'skills': ['Python', 'Docker', 'Machine Learning']},
        ],
        'courses': [
            {'course_name': 'Machine Learning', 'gpa': 4.0},
            {'course_name': 'Artificial Intelligence', 'gpa': 3.9},
            {'course_name': 'Deep Learning', 'gpa': 3.95},
            {'course_name': 'Data Mining', 'gpa': 3.8},
        ]
    },
    {
        's_id': 'STU002',
        'name': 'Bob Smith',
        'cgpa': 3.7,
        'batch': 2022,
        'dept_name': 'Software Engineering',
        'graduation_year': 2026,
        'skills': [
            {'skill_name': 'JavaScript', 'proficiency_level': 'Advanced'},
            {'skill_name': 'React', 'proficiency_level': 'Advanced'},
            {'skill_name': 'Node.js', 'proficiency_level': 'Advanced'},
            {'skill_name': 'Python', 'proficiency_level': 'Intermediate'},
            {'skill_name': 'SQL', 'proficiency_level': 'Intermediate'},
            {'skill_name': 'Git', 'proficiency_level': 'Advanced'},
        ],
        'projects': [
            {'title': 'E-commerce Platform', 'skills': ['React', 'Node.js', 'JavaScript', 'SQL']},
            {'title': 'Real-time Chat Application', 'skills': ['React', 'Node.js', 'WebSocket']},
            {'title': 'Task Management Dashboard', 'skills': ['React', 'Python', 'SQL']},
        ],
        'courses': [
            {'course_name': 'Web Development', 'gpa': 3.8},
            {'course_name': 'Database Systems', 'gpa': 3.7},
            {'course_name': 'Software Engineering', 'gpa': 3.6},
        ]
    },
    {
        's_id': 'STU003',
        'name': 'Charlie Brown',
        'cgpa': 3.5,
        'batch': 2023,
        'dept_name': 'Data Science',
        'graduation_year': 2027,
        'skills': [
            {'skill_name': 'Python', 'proficiency_level': 'Intermediate'},
            {'skill_name': 'R', 'proficiency_level': 'Intermediate'},
            {'skill_name': 'SQL', 'proficiency_level': 'Intermediate'},
            {'skill_name': 'Tableau', 'proficiency_level': 'Beginner'},
            {'skill_name': 'Pandas', 'proficiency_level': 'Intermediate'},
        ],
        'projects': [
            {'title': 'Sales Data Analysis', 'skills': ['Python', 'Pandas', 'SQL']},
            {'title': 'Customer Segmentation', 'skills': ['Python', 'R', 'SQL']},
        ],
        'courses': [
            {'course_name': 'Data Science Fundamentals', 'gpa': 3.5},
            {'course_name': 'Statistical Analysis', 'gpa': 3.4},
            {'course_name': 'Data Visualization', 'gpa': 3.6},
        ]
    },
    {
        's_id': 'STU004',
        'name': 'Diana Prince',
        'cgpa': 3.2,
        'batch': 2024,
        'dept_name': 'Computer Science',
        'graduation_year': 2028,
        'skills': [
            {'skill_name': 'Python', 'proficiency_level': 'Beginner'},
            {'skill_name': 'Java', 'proficiency_level': 'Beginner'},
            {'skill_name': 'Git', 'proficiency_level': 'Beginner'},
        ],
        'projects': [
            {'title': 'Simple Calculator App', 'skills': ['Python']},
        ],
        'courses': [
            {'course_name': 'Introduction to Programming', 'gpa': 3.2},
            {'course_name': 'Data Structures', 'gpa': 3.0},
        ]
    },
    {
        's_id': 'STU005',
        'name': 'Ethan Hunt',
        'cgpa': 3.8,
        'batch': 2022,
        'dept_name': 'Computer Science',
        'graduation_year': 2026,
        'skills': [
            {'skill_name': 'Python', 'proficiency_level': 'Advanced'},
            {'skill_name': 'AWS', 'proficiency_level': 'Intermediate'},
            {'skill_name': 'Docker', 'proficiency_level': 'Advanced'},
            {'skill_name': 'Kubernetes', 'proficiency_level': 'Intermediate'},
            {'skill_name': 'Linux', 'proficiency_level': 'Advanced'},
            {'skill_name': 'Terraform', 'proficiency_level': 'Intermediate'},
        ],
        'projects': [
            {'title': 'Cloud Infrastructure Setup', 'skills': ['AWS', 'Docker', 'Terraform']},
            {'title': 'CI/CD Pipeline', 'skills': ['Docker', 'Kubernetes', 'Linux']},
            {'title': 'Microservices Architecture', 'skills': ['Docker', 'Kubernetes', 'Python']},
        ],
        'courses': [
            {'course_name': 'Cloud Computing', 'gpa': 3.9},
            {'course_name': 'DevOps Practices', 'gpa': 3.8},
            {'course_name': 'System Administration', 'gpa': 3.7},
        ]
    }
]

DEMO_JOBS = [
    {
        'j_id': 'JOB001',
        'title': 'Machine Learning Engineer',
        'description': 'Seeking an experienced ML Engineer to develop and deploy machine learning models. Must have strong Python skills and experience with deep learning frameworks.',
        'required_skills': [
            {'skill_name': 'Python', 'required_level': 'Advanced', 'is_mandatory': True, 'weight': 2.0},
            {'skill_name': 'Machine Learning', 'required_level': 'Intermediate', 'is_mandatory': True, 'weight': 2.0},
            {'skill_name': 'TensorFlow', 'required_level': 'Intermediate', 'is_mandatory': True, 'weight': 1.5},
        ],
        'eligible_batches': [2020, 2021, 2022, 2023],
        'eligible_departments': ['Computer Science', 'Data Science'],
        'eligible_cgpa_min': 3.5
    },
    {
        'j_id': 'JOB002',
        'title': 'Full Stack Developer',
        'description': 'Looking for a Full Stack Developer with expertise in modern web technologies. React and Node.js experience required.',
        'required_skills': [
            {'skill_name': 'JavaScript', 'required_level': 'Advanced', 'is_mandatory': True, 'weight': 2.0},
            {'skill_name': 'React', 'required_level': 'Advanced', 'is_mandatory': True, 'weight': 2.0},
            {'skill_name': 'Node.js', 'required_level': 'Intermediate', 'is_mandatory': True, 'weight': 1.5},
            {'skill_name': 'SQL', 'required_level': 'Intermediate', 'is_mandatory': True, 'weight': 1.0},
        ],
        'eligible_batches': [2021, 2022, 2023, 2024],
        'eligible_departments': ['Computer Science', 'Software Engineering'],
        'eligible_cgpa_min': 3.0
    },
    {
        'j_id': 'JOB003',
        'title': 'Data Scientist',
        'description': 'Data Scientist position requiring strong analytical skills and experience with data analysis tools. Python and SQL proficiency essential.',
        'required_skills': [
            {'skill_name': 'Python', 'required_level': 'Intermediate', 'is_mandatory': True, 'weight': 2.0},
            {'skill_name': 'SQL', 'required_level': 'Intermediate', 'is_mandatory': True, 'weight': 1.5},
            {'skill_name': 'Pandas', 'required_level': 'Intermediate', 'is_mandatory': False, 'weight': 1.0},
            {'skill_name': 'Data Visualization', 'required_level': 'Beginner', 'is_mandatory': False, 'weight': 1.0},
        ],
        'eligible_batches': [2021, 2022, 2023, 2024],
        'eligible_departments': ['Computer Science', 'Data Science', 'Software Engineering'],
        'eligible_cgpa_min': 3.2
    },
    {
        'j_id': 'JOB004',
        'title': 'Cloud DevOps Engineer',
        'description': 'Cloud DevOps Engineer needed to manage cloud infrastructure and CI/CD pipelines. AWS and containerization experience required.',
        'required_skills': [
            {'skill_name': 'AWS', 'required_level': 'Intermediate', 'is_mandatory': True, 'weight': 2.0},
            {'skill_name': 'Docker', 'required_level': 'Advanced', 'is_mandatory': True, 'weight': 2.0},
            {'skill_name': 'Kubernetes', 'required_level': 'Intermediate', 'is_mandatory': True, 'weight': 1.5},
            {'skill_name': 'Linux', 'required_level': 'Intermediate', 'is_mandatory': True, 'weight': 1.0},
        ],
        'eligible_batches': [2020, 2021, 2022, 2023],
        'eligible_departments': ['Computer Science', 'Software Engineering'],
        'eligible_cgpa_min': 3.3
    }
]


def test_new_prediction():
    """
    Test the trained model with demo student and job data
    """
    print("="*60)
    print("🧪 TESTING MODEL PREDICTION WITH DEMO DATA")
    print("="*60)

    # Initialize the matcher
    try:
        matcher = JobMatcherWithFeedback('xai_gradient_boosting.pkl')
    except FileNotFoundError:
        print("❌ Model file not found. Please run model.py first to train and save the model.")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Use first student and first job from demo data
    student = DEMO_STUDENTS[0]
    job = DEMO_JOBS[0]

    # Run prediction
    print(f"\n👤 Student: {student['name']} (ID: {student['s_id']})")
    print(f"💼 Job: {job['title']} (ID: {job['j_id']})")

    result = matcher.match_student_with_job(student, job)

    print("\n✅ Prediction Complete!")
    print(f"\nDetailed Results:")
    print(f"  Match Score: {result['match_score']}")
    print(f"  Category: {result['recommendation_type']}")
    print(f"  Component Scores: {result['component_scores']}")

    return result


def test_multiple_scenarios():
    """
    Test all demo students against all demo jobs
    """
    print("\n" + "="*60)
    print("🧪 TESTING ALL DEMO SCENARIOS")
    print("="*60)
    print(f"\n📊 Testing {len(DEMO_STUDENTS)} students against {len(DEMO_JOBS)} jobs")
    print(f"   Total combinations: {len(DEMO_STUDENTS) * len(DEMO_JOBS)}")

    try:
        matcher = JobMatcherWithFeedback('xai_gradient_boosting.pkl')
    except FileNotFoundError:
        print("❌ Model file not found. Please run model.py first to train and save the model.")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    all_results = []

    # Test each student against each job
    for student_idx, student in enumerate(DEMO_STUDENTS, 1):
        print(f"\n\n{'='*60}")
        print(f"👤 STUDENT {student_idx}: {student['name']} ({student['s_id']})")
        print(f"   CGPA: {student['cgpa']} | Batch: {student['batch']} | Dept: {student['dept_name']}")
        print(f"{'='*60}")

        student_results = []

        for job_idx, job in enumerate(DEMO_JOBS, 1):
            print(f"\n📌 JOB {job_idx}: {job['title']} ({job['j_id']})")
            print("-" * 60)

            result = matcher.match_student_with_job(student, job, feedback_mode='quick')

            student_results.append({
                'job_title': job['title'],
                'job_id': job['j_id'],
                'match_score': result['match_score'],
                'category': result['recommendation_type']
            })

            all_results.append({
                'student_name': student['name'],
                'student_id': student['s_id'],
                'job_title': job['title'],
                'job_id': job['j_id'],
                'match_score': result['match_score'],
                'category': result['recommendation_type']
            })

        # Summary for this student
        print(f"\n📊 Summary for {student['name']}:")
        for sr in student_results:
            print(f"   {sr['job_title']:30s} | Score: {sr['match_score']:5.1f} | {sr['category']}")

    # Overall summary
    print("\n\n" + "="*60)
    print("📊 OVERALL SUMMARY - ALL MATCHES")
    print("="*60)

    # Sort by match score
    all_results.sort(key=lambda x: x['match_score'], reverse=True)

    print(f"\n🏆 Top 10 Matches:")
    print("-" * 60)
    for i, result in enumerate(all_results[:10], 1):
        print(f"{i:2d}. {result['student_name']:20s} → {result['job_title']:30s} | "
              f"Score: {result['match_score']:5.1f} | {result['category']}")

    # Statistics by job
    print(f"\n📈 Statistics by Job:")
    print("-" * 60)
    for job in DEMO_JOBS:
        job_results = [r for r in all_results if r['job_id'] == job['j_id']]
        if job_results:
            avg_score = sum(r['match_score'] for r in job_results) / len(job_results)
            max_score = max(r['match_score'] for r in job_results)
            best_student = max(job_results, key=lambda x: x['match_score'])['student_name']
            print(f"\n{job['title']}:")
            print(f"   Average Score: {avg_score:.2f}")
            print(f"   Best Match: {best_student} ({max_score:.2f})")
            print(f"   Total Candidates: {len(job_results)}")

    # Statistics by student
    print(f"\n📈 Statistics by Student:")
    print("-" * 60)
    for student in DEMO_STUDENTS:
        student_results = [r for r in all_results if r['student_id'] == student['s_id']]
        if student_results:
            avg_score = sum(r['match_score'] for r in student_results) / len(student_results)
            max_score = max(r['match_score'] for r in student_results)
            best_job = max(student_results, key=lambda x: x['match_score'])['job_title']
            print(f"\n{student['name']}:")
            print(f"   Average Score: {avg_score:.2f}")
            print(f"   Best Match: {best_job} ({max_score:.2f})")
            print(f"   Jobs Applied: {len(student_results)}")

    return all_results


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 JOB MATCHING MODEL TEST SUITE")
    print("="*60)
    print(f"\n📋 Demo Data:")
    print(f"   • {len(DEMO_STUDENTS)} Students")
    print(f"   • {len(DEMO_JOBS)} Jobs")
    print(f"   • {len(DEMO_STUDENTS) * len(DEMO_JOBS)} Total Combinations")

    print("\n" + "="*60)
    print("Select test mode:")
    print("1. Single prediction test (one student, one job)")
    print("2. Comprehensive test (all students vs all jobs)")
    print("="*60)

    # Uncomment the test you want to run:

    # Option 1: Single test
    # test_new_prediction()

    # Option 2: Comprehensive test (all students vs all jobs)
    test_multiple_scenarios()