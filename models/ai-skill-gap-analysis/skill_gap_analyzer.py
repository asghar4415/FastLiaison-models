from typing import List, Dict, Any, Optional
from datetime import datetime

class SkillGapAnalyzer:
    """
    Analyzes skill gaps between student profile and target role,
    and generates personalized learning pathways
    """
    
    def __init__(self, student_data, target_role_data):
        self.student = student_data.dict() if hasattr(student_data, 'dict') else student_data
        self.target_role = target_role_data.dict() if hasattr(target_role_data, 'dict') else target_role_data
        self.proficiency_levels = {
            'Beginner': 1,
            'Intermediate': 2,
            'Advanced': 3,
            'Expert': 4
        }
    
    def generate_gap_analysis(self) -> Dict[str, Any]:
        """
        Main function to generate comprehensive skill gap analysis
        """
        skill_gaps = self.identify_skill_gaps()
        overall_gap_score = self.calculate_overall_gap_score(skill_gaps)
        learning_pathways = self.generate_learning_pathways()
        recommendations = self.generate_recommendations(skill_gaps, overall_gap_score)
        estimated_time = self.estimate_time_to_readiness(skill_gaps)
        
        return {
            'overall_gap_score': overall_gap_score,
            'skill_gaps': skill_gaps,
            'learning_pathways': learning_pathways,
            'recommendations': recommendations,
            'estimated_time_to_readiness': estimated_time
        }
    
    def identify_skill_gaps(self) -> List[Dict[str, Any]]:
        """
        Identify gaps between student skills and required skills for target role
        """
        student_skills = {s['name'].lower(): s for s in self.student.get('skills', [])}
        required_skills = self.target_role.get('required_skills', [])
        
        skill_gaps = []
        
        for req_skill in required_skills:
            skill_name = req_skill.get('name', '')
            required_level = req_skill.get('required_level', 'Beginner')
            is_mandatory = req_skill.get('is_mandatory', True)
            
            student_skill = student_skills.get(skill_name.lower())
            
            if student_skill:
                current_level = student_skill.get('proficiency_level', 'Beginner')
                gap_severity = self.calculate_gap_severity(current_level, required_level)
            else:
                current_level = None
                gap_severity = 'critical' if is_mandatory else 'high'
            
            priority_score = self.calculate_priority_score(
                gap_severity, 
                is_mandatory,
                req_skill.get('weight', 1.0)
            )
            
            learning_resources = self.get_learning_resources(skill_name, required_level, current_level)
            
            skill_gaps.append({
                'skill_name': skill_name,
                'current_level': current_level,
                'required_level': required_level,
                'gap_severity': gap_severity,
                'priority_score': round(priority_score, 2),
                'learning_resources': learning_resources
            })
        
        # Sort by priority score (highest first)
        skill_gaps.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return skill_gaps
    
    def calculate_gap_severity(self, current_level: Optional[str], required_level: str) -> str:
        """
        Calculate severity of skill gap
        """
        if current_level is None:
            return 'critical'
        
        current_val = self.proficiency_levels.get(current_level, 0)
        required_val = self.proficiency_levels.get(required_level, 1)
        
        gap = required_val - current_val
        
        if gap >= 3:
            return 'critical'
        elif gap == 2:
            return 'high'
        elif gap == 1:
            return 'medium'
        else:
            return 'low'
    
    def calculate_priority_score(self, gap_severity: str, is_mandatory: bool, weight: float) -> float:
        """
        Calculate priority score for skill gap (0-1 scale)
        """
        severity_scores = {
            'critical': 1.0,
            'high': 0.75,
            'medium': 0.5,
            'low': 0.25
        }
        
        base_score = severity_scores.get(gap_severity, 0.5)
        mandatory_multiplier = 1.2 if is_mandatory else 1.0
        
        return min(base_score * mandatory_multiplier * weight, 1.0)
    
    def get_learning_resources(self, skill_name: str, target_level: str, current_level: Optional[str]) -> List[Dict[str, Any]]:
        """
        Get learning resources for skill development
        This is a placeholder - can be enhanced with actual resource database
        """
        resources = []
        
        # Determine starting difficulty
        if current_level is None:
            difficulty = 'Beginner'
        else:
            current_val = self.proficiency_levels.get(current_level, 1)
            target_val = self.proficiency_levels.get(target_level, 2)
            if target_val - current_val >= 2:
                difficulty = 'Intermediate'
            else:
                difficulty = current_level
        
        # Generate sample resources (can be replaced with database lookup)
        resource_types = ['course', 'tutorial', 'project']
        
        for i, resource_type in enumerate(resource_types):
            resources.append({
                'resource_id': f"{skill_name.lower().replace(' ', '_')}_{resource_type}_{i}",
                'title': f"{skill_name} - {resource_type.title()}",
                'type': resource_type,
                'url': f"https://example.com/{skill_name.lower().replace(' ', '-')}/{resource_type}",
                'duration': self._estimate_duration(target_level, current_level),
                'difficulty': difficulty,
                'estimated_hours': self._estimate_hours(target_level, current_level),
                'cost': 'Free' if i < 2 else 'Paid'
            })
        
        return resources
    
    def _estimate_duration(self, target_level: str, current_level: Optional[str]) -> str:
        """Estimate learning duration"""
        if current_level is None:
            weeks_map = {'Beginner': '4-6', 'Intermediate': '6-8', 'Advanced': '8-12', 'Expert': '12-16'}
            weeks = weeks_map.get(target_level, '4-6')
        else:
            current_val = self.proficiency_levels.get(current_level, 1)
            target_val = self.proficiency_levels.get(target_level, 2)
            gap = target_val - current_val
            weeks_map = {1: '2-4', 2: '4-6', 3: '6-10'}
            weeks = weeks_map.get(gap, '4-6')
        
        return f"{weeks} weeks"
    
    def _estimate_hours(self, target_level: str, current_level: Optional[str]) -> int:
        """Estimate learning hours"""
        if current_level is None:
            hours_map = {'Beginner': 40, 'Intermediate': 60, 'Advanced': 80, 'Expert': 120}
            hours = hours_map.get(target_level, 40)
        else:
            current_val = self.proficiency_levels.get(current_level, 1)
            target_val = self.proficiency_levels.get(target_level, 2)
            gap = target_val - current_val
            hours_map = {1: 20, 2: 40, 3: 60}
            hours = hours_map.get(gap, 30)
        
        return hours
    
    def calculate_overall_gap_score(self, skill_gaps: List[Dict[str, Any]]) -> float:
        """
        Calculate overall gap score (0-1, where 0 = no gap, 1 = maximum gap)
        """
        if not skill_gaps:
            return 0.0
        
        total_priority = sum(gap['priority_score'] for gap in skill_gaps)
        max_possible = len(skill_gaps) * 1.0  # Assuming max priority score is 1.0
        
        return min(total_priority / max_possible, 1.0) if max_possible > 0 else 0.0
    
    def generate_learning_pathways(self) -> List[Dict[str, Any]]:
        """
        Generate multiple learning pathway options
        """
        skill_gaps = self.identify_skill_gaps()
        
        # Pathway 1: Fast Track (focus on critical gaps)
        fast_track = self._create_pathway(
            pathway_id="fast_track",
            pathway_name="Fast Track Pathway",
            description="Intensive pathway focusing on critical skill gaps",
            skill_gaps=[g for g in skill_gaps if g['gap_severity'] in ['critical', 'high']],
            difficulty="Intensive"
        )
        
        # Pathway 2: Comprehensive (all skills)
        comprehensive = self._create_pathway(
            pathway_id="comprehensive",
            pathway_name="Comprehensive Pathway",
            description="Complete skill development covering all gaps",
            skill_gaps=skill_gaps,
            difficulty="Moderate"
        )
        
        # Pathway 3: Foundation First (beginner-friendly)
        foundation = self._create_pathway(
            pathway_id="foundation",
            pathway_name="Foundation Pathway",
            description="Build strong foundations before advanced topics",
            skill_gaps=[g for g in skill_gaps if g['current_level'] in [None, 'Beginner']],
            difficulty="Beginner-Friendly"
        )
        
        return [fast_track, comprehensive, foundation]
    
    def _create_pathway(self, pathway_id: str, pathway_name: str, description: str, 
                        skill_gaps: List[Dict[str, Any]], difficulty: str) -> Dict[str, Any]:
        """Helper to create a learning pathway"""
        all_resources = []
        skills_covered = []
        milestones = []
        
        for i, gap in enumerate(skill_gaps[:5]):  # Limit to top 5 gaps per pathway
            skills_covered.append(gap['skill_name'])
            all_resources.extend(gap['learning_resources'][:2])  # Top 2 resources per skill
            
            milestones.append({
                'milestone_number': i + 1,
                'skill': gap['skill_name'],
                'target_level': gap['required_level'],
                'estimated_completion': f"Week {i + 1}"
            })
        
        total_hours = sum(r.get('estimated_hours', 20) for r in all_resources)
        estimated_weeks = max(1, total_hours // 10)  # Assuming 10 hours per week
        
        return {
            'pathway_id': pathway_id,
            'pathway_name': pathway_name,
            'description': description,
            'estimated_completion_time': f"{estimated_weeks} weeks",
            'difficulty': difficulty,
            'skills_covered': skills_covered,
            'resources': all_resources,
            'milestones': milestones
        }
    
    def generate_recommendations(self, skill_gaps: List[Dict[str, Any]], overall_gap_score: float) -> Dict[str, Any]:
        """
        Generate actionable recommendations
        """
        critical_gaps = [g for g in skill_gaps if g['gap_severity'] == 'critical']
        high_gaps = [g for g in skill_gaps if g['gap_severity'] == 'high']
        
        recommendations = {
            'immediate_actions': [],
            'short_term_goals': [],
            'long_term_goals': [],
            'focus_areas': []
        }
        
        if critical_gaps:
            recommendations['immediate_actions'].append({
                'action': f"Prioritize learning {critical_gaps[0]['skill_name']}",
                'reason': "This is a critical skill gap for your target role",
                'priority': 'high'
            })
        
        if overall_gap_score > 0.7:
            recommendations['immediate_actions'].append({
                'action': "Consider a structured learning program or bootcamp",
                'reason': "Significant skill gaps detected - structured learning recommended",
                'priority': 'high'
            })
        elif overall_gap_score > 0.4:
            recommendations['short_term_goals'].append({
                'goal': "Complete foundational courses for missing skills",
                'timeline': "2-3 months"
            })
        
        recommendations['focus_areas'] = [g['skill_name'] for g in skill_gaps[:3]]
        
        return recommendations
    
    def estimate_time_to_readiness(self, skill_gaps: List[Dict[str, Any]]) -> str:
        """
        Estimate time needed to be ready for target role
        """
        if not skill_gaps:
            return "Ready now"
        
        total_hours = 0
        for gap in skill_gaps:
            for resource in gap['learning_resources'][:1]:  # Top resource per skill
                total_hours += resource.get('estimated_hours', 20)
        
        # Assuming 10-15 hours per week of learning
        weeks = total_hours / 12  # Average 12 hours per week
        months = weeks / 4
        
        if months < 1:
            return f"{int(weeks)} weeks"
        elif months < 6:
            return f"{int(months)} months"
        else:
            return f"{int(months / 6)}-{int(months / 6) + 1} months"

