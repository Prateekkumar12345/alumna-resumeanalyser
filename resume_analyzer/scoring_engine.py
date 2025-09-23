"""
ATS scoring engine for resume analysis
Calculates comprehensive scores without social media dependencies
"""

import logging
from .config import SCORING_CONFIG, ATS_KEYWORDS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ATSScoringEngine:
    """Enhanced ATS scoring system focused on professional content quality"""
    
    def __init__(self):
        self.scoring_config = SCORING_CONFIG
        self.ats_keywords = ATS_KEYWORDS
    
    def calculate_comprehensive_ats_score(self, text, sections, target_role=None):
        """
        Calculate comprehensive ATS score with detailed breakdown
        
        Args:
            text (str): Resume text content
            sections (dict): Parsed resume sections
            target_role (str): Target job role for customized scoring
            
        Returns:
            tuple: (total_score, max_score, detailed_breakdown)
        """
        score_breakdown = {
            'contact_info': {'score': 0, 'max': 15, 'details': [], 'weight': 0.15},
            'technical_skills': {'score': 0, 'max': 30, 'details': [], 'weight': 0.30},
            'experience_quality': {'score': 0, 'max': 25, 'details': [], 'weight': 0.25},
            'quantified_achievements': {'score': 0, 'max': 20, 'details': [], 'weight': 0.20},
            'content_optimization': {'score': 0, 'max': 10, 'details': [], 'weight': 0.10}
        }
        
        # Analyze each scoring category
        self._score_contact_information(sections, score_breakdown['contact_info'])
        self._score_technical_skills(text, sections, target_role, score_breakdown['technical_skills'])
        self._score_experience_quality(sections, score_breakdown['experience_quality'])
        self._score_quantified_achievements(sections, score_breakdown['quantified_achievements'])
        self._score_content_optimization(text, sections, score_breakdown['content_optimization'])
        
        # Calculate totals
        total_score = sum(category['score'] for category in score_breakdown.values())
        max_possible = sum(category['max'] for category in score_breakdown.values())
        
        # Add overall assessment
        overall_percentage = (total_score / max_possible) * 100
        score_breakdown['overall_assessment'] = self._generate_overall_assessment(overall_percentage)
        
        logger.info(f"Calculated ATS score: {total_score}/{max_possible} ({overall_percentage:.1f}%)")
        
        return total_score, max_possible, score_breakdown
    
    def _score_contact_information(self, sections, contact_score):
        """Score contact information completeness and quality"""
        # Email assessment (8 points max)
        if sections.get('email'):
            contact_score['score'] += 8
            contact_score['details'].append("✅ Professional email address provided")
            
            # Bonus for professional email format
            email = sections.get('email', '')
            if any(domain in email.lower() for domain in ['gmail', 'yahoo', 'outlook', 'hotmail']):
                contact_score['details'].append("✅ Uses common email provider (professional)")
        else:
            contact_score['details'].append("❌ CRITICAL: Missing email address - will result in automatic rejection")
        
        # Phone number assessment (7 points max)
        if sections.get('phone'):
            contact_score['score'] += 7
            contact_score['details'].append("✅ Phone number provided for direct contact")
            
            # Check for multiple contact numbers
            if sections.get('phone_count', 0) > 1:
                contact_score['details'].append("✅ Multiple contact methods available")
        else:
            contact_score['details'].append("❌ Missing phone number - reduces recruiter contact options")
        
        # Contact completeness assessment
        if sections.get('email') and sections.get('phone'):
            contact_score['details'].append("🎯 Complete contact information enables easy recruiter outreach")
    
    def _score_technical_skills(self, text, sections, target_role, skills_score):
        """Score technical skills relevance and depth"""
        skills_text = sections.get('skills_text', '').lower()
        text_lower = text.lower()
        individual_skills = sections.get('individual_skills', [])
        
        # Base technical skills assessment
        base_score = 0
        
        # Skills section presence and quality
        if skills_text:
            base_score += 5
            skills_score['details'].append("✅ Dedicated skills section found")
            
            # Skills count assessment
            skills_count = sections.get('skills_count', 0)
            if skills_count >= 12:
                base_score += 5
                skills_score['details'].append(f"✅ Comprehensive skill set ({skills_count} skills listed)")
            elif skills_count >= 6:
                base_score += 3
                skills_score['details'].append(f"✅ Good skill variety ({skills_count} skills listed)")
            elif skills_count > 0:
                base_score += 1
                skills_score['details'].append(f"⚠️ Limited skills listed ({skills_count} skills) - expand to show full expertise")
        else:
            skills_score['details'].append("❌ No dedicated skills section - critical for ATS parsing")
        
        # Role-specific skills analysis
        role_bonus = 0
        if target_role and target_role.lower().replace(' ', '_') in self.ats_keywords:
            role_data = self.ats_keywords[target_role.lower().replace(' ', '_')]
            role_bonus = self._calculate_role_specific_score(text_lower, role_data, skills_score, target_role)
        else:
            # General technical skills assessment
            role_bonus = self._calculate_general_tech_score(text_lower, skills_score)
        
        # Technical skill categories analysis
        skill_categories = sections.get('skill_categories', {})
        category_bonus = self._assess_skill_diversity(skill_categories, skills_score)
        
        # Calculate final technical skills score
        total_technical_score = base_score + role_bonus + category_bonus
        skills_score['score'] = min(total_technical_score, 30)
    
    def _score_experience_quality(self, sections, experience_score):
        """Score work experience quality and presentation"""
        experience_years = sections.get('experience_years', 0)
        position_count = sections.get('position_count', 0)
        experience_quality = sections.get('experience_quality', 0)
        
        # Years of experience scoring
        if experience_years >= 5:
            experience_score['score'] += 10
            experience_score['details'].append(f"✅ Extensive experience ({experience_years} years) - senior-level candidate")
        elif experience_years >= 3:
            experience_score['score'] += 8
            experience_score['details'].append(f"✅ Solid experience ({experience_years} years) - mid-level candidate")
        elif experience_years >= 1:
            experience_score['score'] += 5
            experience_score['details'].append(f"✅ Professional experience ({experience_years} years) - junior-level candidate")
        elif experience_years > 0:
            experience_score['score'] += 2
            experience_score['details'].append(f"⚠️ Limited experience ({experience_years} years) - entry-level positions")
        else:
            experience_score['details'].append("❌ No clear work experience timeline - add internships, projects, or part-time work")
        
        # Position diversity and career progression
        if position_count >= 3:
            experience_score['score'] += 4
            experience_score['details'].append(f"✅ Diverse work history ({position_count} positions) shows adaptability")
        elif position_count >= 2:
            experience_score['score'] += 2
            experience_score['details'].append(f"✅ Multiple positions ({position_count}) show career progression")
        elif position_count == 1:
            experience_score['score'] += 1
            experience_score['details'].append("⚠️ Single position listed - consider adding internships or projects")
        
        # Experience description quality
        if experience_quality >= 70:
            experience_score['score'] += 6
            experience_score['details'].append("✅ High-quality experience descriptions with quantified achievements")
        elif experience_quality >= 50:
            experience_score['score'] += 4
            experience_score['details'].append("✅ Good experience descriptions - consider adding more metrics")
        elif experience_quality >= 30:
            experience_score['score'] += 2
            experience_score['details'].append("⚠️ Basic experience descriptions - add technical details and achievements")
        else:
            experience_score['details'].append("❌ Poor experience descriptions - rewrite with action verbs and quantified results")
        
        # Leadership and collaboration indicators
        if sections.get('has_leadership'):
            experience_score['score'] += 3
            experience_score['details'].append("✅ Leadership experience demonstrated - valuable for senior roles")
        
        # Internship experience for entry-level candidates
        if experience_years <= 2 and sections.get('has_internship'):
            experience_score['score'] += 2
            experience_score['details'].append("✅ Internship experience shows industry exposure")
        
        # Cap the score at maximum
        experience_score['score'] = min(experience_score['score'], 25)
    
    def _score_quantified_achievements(self, sections, achievements_score):
        """Score quantified achievements and impact metrics"""
        achievement_count = sections.get('quantified_achievements', 0)
        achievement_examples = sections.get('achievement_examples', [])
        achievement_diversity = sections.get('achievement_diversity', 0)
        
        # Base achievement scoring
        if achievement_count >= 5:
            achievements_score['score'] += 15
            achievements_score['details'].append(f"✅ Excellent quantified achievements ({achievement_count} metrics) - shows strong business impact")
        elif achievement_count >= 3:
            achievements_score['score'] += 10
            achievements_score['details'].append(f"✅ Good quantified achievements ({achievement_count} metrics) - demonstrates measurable value")
        elif achievement_count >= 1:
            achievements_score['score'] += 5
            achievements_score['details'].append(f"⚠️ Some quantified achievements ({achievement_count} metrics) - add more specific numbers")
        else:
            achievements_score['details'].append("❌ No quantified achievements - critical weakness, add percentages, dollar amounts, time savings")
        
        # Achievement diversity bonus
        if achievement_diversity >= 3:
            achievements_score['score'] += 3
            achievements_score['details'].append("✅ Diverse achievement types (performance, financial, scale) show well-rounded impact")
        elif achievement_diversity >= 2:
            achievements_score['score'] += 2
            achievements_score['details'].append("✅ Multiple achievement categories demonstrate varied contributions")
        
        # Quality of achievement examples
        if achievement_examples:
            high_quality_achievements = [ex for ex in achievement_examples[:3] 
                                       if any(indicator in ex.lower() for indicator in ['%', '$', 'increased', 'reduced', 'improved'])]
            if len(high_quality_achievements) >= 2:
                achievements_score['score'] += 2
                achievements_score['details'].append("✅ High-quality achievement examples with specific metrics")
        
        # Cap at maximum score
        achievements_score['score'] = min(achievements_score['score'], 20)
    
    def _score_content_optimization(self, text, sections, content_score):
        """Score content structure, formatting, and ATS optimization"""
        word_count = sections.get('word_count', 0)
        action_verb_count = sections.get('action_verb_count', 0)
        section_headers = sections.get('section_headers', 0)
        
        # Optimal length assessment
        if 400 <= word_count <= 800:
            content_score['score'] += 4
            content_score['details'].append(f"✅ Optimal resume length ({word_count} words) for ATS processing")
        elif 300 <= word_count <= 1000:
            content_score['score'] += 3
            content_score['details'].append(f"✅ Good resume length ({word_count} words)")
        elif word_count < 300:
            content_score['details'].append(f"❌ Resume too short ({word_count} words) - expand project and experience descriptions")
        else:
            content_score['details'].append(f"⚠️ Resume quite long ({word_count} words) - consider condensing for better ATS performance")
        
        # Action verb usage
        if action_verb_count >= 10:
            content_score['score'] += 3
            content_score['details'].append(f"✅ Excellent use of action verbs ({action_verb_count}) - shows proactive approach")
        elif action_verb_count >= 6:
            content_score['score'] += 2
            content_score['details'].append(f"✅ Good action verb usage ({action_verb_count}) - demonstrates initiative")
        elif action_verb_count >= 3:
            content_score['score'] += 1
            content_score['details'].append(f"⚠️ Limited action verbs ({action_verb_count}) - use more dynamic language")
        else:
            content_score['details'].append("❌ Few action verbs - rewrite with words like 'developed', 'led', 'implemented'")
        
        # Structure and organization
        if section_headers >= 4:
            content_score['score'] += 2
            content_score['details'].append("✅ Well-structured with clear section headers")
        elif section_headers >= 2:
            content_score['score'] += 1
            content_score['details'].append("✅ Basic structure with section headers")
        
        # Education presence (especially important for entry-level)
        if sections.get('has_education'):
            content_score['score'] += 1
            content_score['details'].append("✅ Education section present")
        
        # Cap at maximum score
        content_score['score'] = min(content_score['score'], 10)
    
    def _calculate_role_specific_score(self, text_lower, role_data, skills_score, target_role):
        """Calculate role-specific technical skills score"""
        role_score = 0
        
        # Core skills matching
        core_skills = role_data.get('core_skills', [])
        matched_core = [skill for skill in core_skills if skill.lower() in text_lower]
        core_score = min(15, len(matched_core) * 2)
        role_score += core_score
        
        if matched_core:
            skills_score['details'].append(f"✅ {target_role} core skills: {', '.join(matched_core[:6])}")
        
        # Framework matching
        frameworks = role_data.get('frameworks', [])
        matched_frameworks = [fw for fw in frameworks if fw.lower() in text_lower]
        framework_score = min(8, len(matched_frameworks) * 1.5)
        role_score += framework_score
        
        if matched_frameworks:
            skills_score['details'].append(f"✅ Relevant frameworks: {', '.join(matched_frameworks[:4])}")
        
        # Methodology matching
        methodologies = role_data.get('methodologies', [])
        matched_methodologies = [method for method in methodologies if method.lower() in text_lower]
        method_score = min(5, len(matched_methodologies) * 1)
        role_score += method_score
        
        if matched_methodologies:
            skills_score['details'].append(f"✅ Industry methodologies: {', '.join(matched_methodologies[:3])}")
        
        # Identify missing critical skills
        missing_core = [skill for skill in core_skills[:6] if skill.lower() not in text_lower]
        if missing_core:
            skills_score['details'].append(f"❌ Missing {target_role} skills: {', '.join(missing_core[:4])}")
        
        return role_score
    
    def _calculate_general_tech_score(self, text_lower, skills_score):
        """Calculate general technical skills score when no specific role is targeted"""
        general_score = 0
        
        # General programming skills
        programming_skills = ['python', 'java', 'javascript', 'c++', 'sql', 'html', 'css']
        matched_programming = [skill for skill in programming_skills if skill in text_lower]
        
        if len(matched_programming) >= 4:
            general_score += 12
            skills_score['details'].append(f"✅ Strong programming foundation: {', '.join(matched_programming[:5])}")
        elif len(matched_programming) >= 2:
            general_score += 8
            skills_score['details'].append(f"✅ Basic programming skills: {', '.join(matched_programming)}")
        
        # General tools and platforms
        tools_platforms = ['git', 'docker', 'aws', 'linux', 'api', 'database', 'cloud']
        matched_tools = [tool for tool in tools_platforms if tool in text_lower]
        
        if len(matched_tools) >= 3:
            general_score += 8
            skills_score['details'].append(f"✅ Modern development tools: {', '.join(matched_tools[:4])}")
        elif len(matched_tools) >= 1:
            general_score += 4
            skills_score['details'].append(f"✅ Development tools: {', '.join(matched_tools)}")
        
        return general_score
    
    def _assess_skill_diversity(self, skill_categories, skills_score):
        """Assess diversity of technical skills across categories"""
        if not skill_categories:
            return 0
        
        diversity_score = 0
        non_empty_categories = [cat for cat, skills in skill_categories.items() if skills]
        
        if len(non_empty_categories) >= 4:
            diversity_score += 3
            skills_score['details'].append("✅ Excellent skill diversity across multiple technical areas")
        elif len(non_empty_categories) >= 3:
            diversity_score += 2
            skills_score['details'].append("✅ Good skill diversity across technical areas")
        elif len(non_empty_categories) >= 2:
            diversity_score += 1
            skills_score['details'].append("✅ Moderate skill diversity")
        
        return diversity_score
    
    def _generate_overall_assessment(self, percentage):
        """Generate overall assessment based on score percentage"""
        if percentage >= 85:
            return {
                'level': 'Outstanding',
                'description': 'Exceptional resume with strong ATS optimization. Ready for immediate applications to target companies.',
                'recommendation': 'Start applying to dream companies immediately. Your resume demonstrates strong professional credentials.',
                'color': 'success'
            }
        elif percentage >= 75:
            return {
                'level': 'Excellent',
                'description': 'Strong resume with good ATS performance. Minor improvements could enhance competitiveness.',
                'recommendation': 'Apply to target positions while making suggested minor improvements in parallel.',
                'color': 'success'
            }
        elif percentage >= 65:
            return {
                'level': 'Good',
                'description': 'Solid resume foundation with room for meaningful improvements to maximize ATS performance.',
                'recommendation': 'Focus on top priority improvements before applying to highly competitive positions.',
                'color': 'warning'
            }
        elif percentage >= 50:
            return {
                'level': 'Fair',
                'description': 'Resume needs significant improvements to compete effectively in current job market.',
                'recommendation': 'Address critical weaknesses before submitting applications. Focus on quantified achievements and technical skills.',
                'color': 'warning'
            }
        else:
            return {
                'level': 'Needs Improvement',
                'description': 'Resume requires major enhancements to pass ATS filters and attract recruiter attention.',
                'recommendation': 'Complete resume overhaul needed. Focus on contact info, skills section, and quantified achievements first.',
                'color': 'error'
            }
    
    def get_score_interpretation(self, total_score, max_score):
        """Get detailed interpretation of the resume score"""
        percentage = (total_score / max_score) * 100
        
        interpretation = {
            'percentage': percentage,
            'grade': self._get_letter_grade(percentage),
            'ats_likelihood': self._get_ats_pass_likelihood(percentage),
            'competitive_level': self._get_competitive_assessment(percentage),
            'improvement_urgency': self._get_improvement_urgency(percentage)
        }
        
        return interpretation
    
    def _get_letter_grade(self, percentage):
        """Convert percentage to letter grade"""
        if percentage >= 90: return 'A+'
        elif percentage >= 85: return 'A'
        elif percentage >= 80: return 'A-'
        elif percentage >= 75: return 'B+'
        elif percentage >= 70: return 'B'
        elif percentage >= 65: return 'B-'
        elif percentage >= 60: return 'C+'
        elif percentage >= 55: return 'C'
        elif percentage >= 50: return 'C-'
        else: return 'D'
    
    def _get_ats_pass_likelihood(self, percentage):
        """Estimate likelihood of passing ATS filters"""
        if percentage >= 80: return "Very High (90-95%)"
        elif percentage >= 70: return "High (80-90%)"
        elif percentage >= 60: return "Moderate (60-75%)"
        elif percentage >= 50: return "Low (30-50%)"
        else: return "Very Low (10-25%)"
    
    def _get_competitive_assessment(self, percentage):
        """Assess competitiveness against other candidates"""
        if percentage >= 85: return "Highly Competitive - Top 10% of candidates"
        elif percentage >= 75: return "Very Competitive - Top 25% of candidates"
        elif percentage >= 65: return "Competitive - Above average candidate"
        elif percentage >= 55: return "Moderately Competitive - Average candidate pool"
        else: return "Below Average - Significant improvements needed"
    
    def _get_improvement_urgency(self, percentage):
        """Assess urgency of improvements needed"""
        if percentage >= 80: return "Low - Minor optimizations recommended"
        elif percentage >= 70: return "Medium - Targeted improvements beneficial"
        elif percentage >= 60: return "High - Several key areas need attention"
        else: return "Critical - Major overhaul required before applications"