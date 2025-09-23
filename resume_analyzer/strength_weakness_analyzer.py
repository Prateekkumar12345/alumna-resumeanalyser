"""
Strength and weakness analysis module for resume evaluation
Provides detailed analysis without social media dependencies
"""

import logging
from .config import ATS_KEYWORDS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrengthWeaknessAnalyzer:
    """Analyzes resume strengths and weaknesses with detailed explanations"""
    
    def __init__(self):
        self.ats_keywords = ATS_KEYWORDS
    
    def analyze_comprehensive_strengths_weaknesses(self, text, sections, target_role=None):
        """
        Provide detailed strength and weakness analysis with specific explanations
        
        Args:
            text (str): Resume text content
            sections (dict): Parsed resume sections
            target_role (str): Target job role for analysis
            
        Returns:
            tuple: (detailed_strengths, detailed_weaknesses)
        """
        strengths_detailed = []
        weaknesses_detailed = []
        
        # Analyze strengths with comprehensive explanations
        strengths_detailed.extend(self._analyze_experience_strengths(sections))
        strengths_detailed.extend(self._analyze_technical_strengths(text, sections, target_role))
        strengths_detailed.extend(self._analyze_achievement_strengths(sections))
        strengths_detailed.extend(self._analyze_content_quality_strengths(sections))
        strengths_detailed.extend(self._analyze_professional_presentation_strengths(sections))
        
        # Analyze weaknesses with detailed improvement guidance
        weaknesses_detailed.extend(self._analyze_contact_weaknesses(sections))
        weaknesses_detailed.extend(self._analyze_technical_weaknesses(text, sections, target_role))
        weaknesses_detailed.extend(self._analyze_experience_weaknesses(sections))
        weaknesses_detailed.extend(self._analyze_achievement_weaknesses(sections))
        weaknesses_detailed.extend(self._analyze_content_structure_weaknesses(sections))
        
        logger.info(f"Identified {len(strengths_detailed)} strengths and {len(weaknesses_detailed)} weaknesses")
        
        return strengths_detailed, weaknesses_detailed
    
    def _analyze_experience_strengths(self, sections):
        """Analyze experience-related strengths"""
        strengths = []
        experience_years = sections.get('experience_years', 0)
        experience_quality = sections.get('experience_quality', 0)
        position_count = sections.get('position_count', 0)
        
        # Substantial experience strength
        if experience_years >= 5:
            strengths.append({
                'strength': f'Extensive Professional Experience ({experience_years} years)',
                'why_its_strong': 'Demonstrates deep industry knowledge, proven track record, and career progression over multiple years',
                'ats_benefit': 'ATS systems heavily weight experience when filtering candidates, placing you in senior candidate pools',
                'competitive_advantage': 'Positions you as an experienced professional who can contribute immediately without extensive training',
                'evidence': f'Resume shows {experience_years} years of documented professional experience with career progression'
            })
        elif experience_years >= 3:
            strengths.append({
                'strength': f'Solid Professional Background ({experience_years} years)',
                'why_its_strong': 'Shows you have moved beyond entry-level and developed professional competencies',
                'ats_benefit': 'Meets experience requirements for most mid-level positions, improving application success rate',
                'competitive_advantage': 'Balances experience with adaptability, appealing to companies seeking growth-oriented professionals',
                'evidence': f'Resume demonstrates {experience_years} years of professional development and skill building'
            })
        elif experience_years >= 1:
            strengths.append({
                'strength': f'Relevant Professional Experience ({experience_years} years)',
                'why_its_strong': 'Proves you can function in professional environments and deliver results',
                'ats_benefit': 'Differentiates you from new graduates and meets minimum experience requirements',
                'competitive_advantage': 'Shows practical application of skills in real-world business contexts',
                'evidence': f'Resume shows {experience_years} years of hands-on professional experience'
            })
        
        # High-quality experience descriptions
        if experience_quality >= 70:
            strengths.append({
                'strength': 'Exceptionally Well-Crafted Experience Descriptions',
                'why_its_strong': 'Experience section uses strong action verbs, quantified achievements, and technical depth',
                'ats_benefit': 'Rich keyword density and professional language boost ATS matching scores significantly',
                'competitive_advantage': 'Detailed, results-oriented descriptions demonstrate professional communication skills',
                'evidence': f'Experience descriptions score {experience_quality}/100 for quality, showing strong professional writing'
            })
        elif experience_quality >= 50:
            strengths.append({
                'strength': 'Well-Structured Professional Experience',
                'why_its_strong': 'Experience descriptions contain good detail and demonstrate professional growth',
                'ats_benefit': 'Professional formatting and content structure are well-received by ATS parsing systems',
                'competitive_advantage': 'Shows ability to articulate achievements and responsibilities clearly',
                'evidence': f'Experience section demonstrates good professional communication with {experience_quality}/100 quality score'
            })
        
        # Leadership and career progression
        if sections.get('has_leadership'):
            strengths.append({
                'strength': 'Demonstrated Leadership Experience',
                'why_its_strong': 'Leadership roles indicate trust from management, people skills, and career advancement',
                'ats_benefit': 'Leadership keywords are highly valued by ATS systems for senior and management roles',
                'competitive_advantage': 'Leadership experience differentiates you for roles requiring team coordination and decision-making',
                'evidence': 'Resume shows leadership roles and team management responsibilities'
            })
        
        # Multiple positions showing career growth
        if position_count >= 3:
            strengths.append({
                'strength': f'Diverse Professional Background ({position_count} positions)',
                'why_its_strong': 'Multiple positions demonstrate adaptability, learning ability, and career progression',
                'ats_benefit': 'Variety of roles provides broader keyword coverage and shows professional versatility',
                'competitive_advantage': 'Diverse experience makes you valuable for roles requiring cross-functional collaboration',
                'evidence': f'Resume shows progression through {position_count} different professional roles'
            })
        
        return strengths
    
    def _analyze_technical_strengths(self, text, sections, target_role):
        """Analyze technical skills and competency strengths"""
        strengths = []
        text_lower = text.lower()
        
        # Comprehensive technical skills
        skills_count = sections.get('skills_count', 0)
        if skills_count >= 15:
            strengths.append({
                'strength': f'Extensive Technical Skill Portfolio ({skills_count} skills)',
                'why_its_strong': 'Broad technical knowledge demonstrates adaptability and continuous learning',
                'ats_benefit': 'Comprehensive skill set improves matching for diverse technical positions',
                'competitive_advantage': 'Versatility makes you valuable for multiple roles and cross-functional projects',
                'evidence': f'Resume lists {skills_count} distinct technical skills across multiple categories'
            })
        elif skills_count >= 10:
            strengths.append({
                'strength': f'Strong Technical Foundation ({skills_count} skills)',
                'why_its_strong': 'Well-rounded technical skill set shows professional competency',
                'ats_benefit': 'Good keyword coverage for technical roles improves application success',
                'competitive_advantage': 'Solid technical foundation enables you to contribute across different areas',
                'evidence': f'Resume demonstrates {skills_count} relevant technical competencies'
            })
        
        # Technical depth and sophistication
        technical_depth = sections.get('technical_depth_score', 0)
        if technical_depth >= 10:
            strengths.append({
                'strength': 'Advanced Technical Communication and Depth',
                'why_its_strong': 'Uses sophisticated technical terminology and demonstrates deep understanding of complex concepts',
                'ats_benefit': 'Advanced technical language significantly boosts ATS relevance scores for senior technical roles',
                'competitive_advantage': 'Technical depth positions you for senior roles requiring architectural thinking and system design',
                'evidence': f'Resume contains {technical_depth} advanced technical concepts and sophisticated language'
            })
        elif technical_depth >= 5:
            strengths.append({
                'strength': 'Good Technical Understanding and Communication',
                'why_its_strong': 'Demonstrates ability to work with complex technical systems and communicate technical concepts',
                'ats_benefit': 'Technical terminology improves keyword matching for technical roles',
                'competitive_advantage': 'Technical communication skills are valuable for collaborative development environments',
                'evidence': f'Resume shows {technical_depth} technical concepts indicating solid technical understanding'
            })
        
        # Role-specific technical alignment
        if target_role and target_role.lower().replace(' ', '_') in self.ats_keywords:
            role_data = self.ats_keywords[target_role.lower().replace(' ', '_')]
            core_skills = role_data.get('core_skills', [])
            matched_core = [skill for skill in core_skills if skill.lower() in text_lower]
            
            if len(matched_core) >= len(core_skills) * 0.7:  # 70% or more core skills matched
                strengths.append({
                    'strength': f'Strong {target_role} Technical Alignment',
                    'why_its_strong': f'Resume contains most core technical skills required for {target_role} positions',
                    'ats_benefit': f'High skill matching significantly improves ATS scores for {target_role} applications',
                    'competitive_advantage': f'Technical skill alignment positions you as a qualified {target_role} candidate',
                    'evidence': f'Resume matches {len(matched_core)} out of {len(core_skills)} core {target_role} skills: {", ".join(matched_core[:5])}'
                })
        
        # Skill categorization and diversity
        skill_categories = sections.get('skill_categories', {})
        non_empty_categories = [cat for cat, skills in skill_categories.items() if skills]
        if len(non_empty_categories) >= 4:
            strengths.append({
                'strength': 'Excellent Technical Skill Diversity',
                'why_its_strong': 'Skills span multiple technical categories showing full-stack competency',
                'ats_benefit': 'Diverse skill set matches keywords across multiple job categories',
                'competitive_advantage': 'Technical versatility makes you valuable for complex projects requiring multiple skill areas',
                'evidence': f'Technical skills cover {len(non_empty_categories)} major categories: {", ".join(non_empty_categories[:4])}'
            })
        
        return strengths
    
    def _analyze_achievement_strengths(self, sections):
        """Analyze quantified achievements and impact strengths"""
        strengths = []
        achievement_count = sections.get('quantified_achievements', 0)
        achievement_diversity = sections.get('achievement_diversity', 0)
        
        # Strong quantified achievements
        if achievement_count >= 4:
            strengths.append({
                'strength': f'Exceptional Quantified Impact Documentation ({achievement_count} metrics)',
                'why_its_strong': 'Multiple quantified achievements demonstrate consistent results delivery and business impact measurement',
                'ats_benefit': 'Numbers and percentages are heavily weighted by ATS systems as indicators of performance',
                'competitive_advantage': 'Quantified achievements clearly communicate value proposition to hiring managers',
                'evidence': f'Resume contains {achievement_count} specific metrics showing measurable business impact'
            })
        elif achievement_count >= 2:
            strengths.append({
                'strength': f'Good Results-Oriented Approach ({achievement_count} quantified achievements)',
                'why_its_strong': 'Quantified achievements show focus on measurable outcomes and business value',
                'ats_benefit': 'Specific metrics improve ATS scoring and demonstrate performance orientation',
                'competitive_advantage': 'Results-focused approach appeals to employers seeking impact-driven professionals',
                'evidence': f'Resume includes {achievement_count} quantified achievements demonstrating measurable results'
            })
        
        # Achievement diversity across categories
        if achievement_diversity >= 3:
            strengths.append({
                'strength': 'Diverse Impact Across Multiple Business Areas',
                'why_its_strong': 'Achievements span different categories (performance, financial, scale) showing well-rounded contribution',
                'ats_benefit': 'Variety of achievement types matches diverse job requirement keywords',
                'competitive_advantage': 'Demonstrates ability to create value across multiple business dimensions',
                'evidence': f'Achievements span {achievement_diversity} different impact categories showing comprehensive contribution'
            })
        
        return strengths
    
    def _analyze_content_quality_strengths(self, sections):
        """Analyze content quality and presentation strengths"""
        strengths = []
        word_count = sections.get('word_count', 0)
        action_verb_count = sections.get('action_verb_count', 0)
        
        # Optimal resume length
        if 400 <= word_count <= 800:
            strengths.append({
                'strength': f'Optimal Resume Length and Detail ({word_count} words)',
                'why_its_strong': 'Resume length is ideal for ATS processing while providing sufficient detail for evaluation',
                'ats_benefit': 'Optimal length ensures complete parsing by ATS systems without overwhelming filters',
                'competitive_advantage': 'Concise yet comprehensive presentation respects recruiter time while showing professionalism',
                'evidence': f'Resume contains {word_count} words, optimal for both ATS systems and human reviewers'
            })
        
        # Strong action verb usage
        if action_verb_count >= 8:
            strengths.append({
                'strength': f'Dynamic Professional Language ({action_verb_count} action verbs)',
                'why_its_strong': 'Extensive use of action verbs demonstrates proactive approach and professional communication skills',
                'ats_benefit': 'Action verbs are heavily weighted by ATS systems as indicators of professional competency',
                'competitive_advantage': 'Dynamic language creates strong impression of capability and initiative',
                'evidence': f'Resume uses {action_verb_count} different action verbs showing proactive professional approach'
            })
        
        return strengths
    
    def _analyze_professional_presentation_strengths(self, sections):
        """Analyze professional presentation and structure strengths"""
        strengths = []
        
        # Complete contact information
        if sections.get('email') and sections.get('phone'):
            strengths.append({
                'strength': 'Complete Professional Contact Information',
                'why_its_strong': 'Comprehensive contact details enable easy recruiter outreach through multiple channels',
                'ats_benefit': 'Complete contact information prevents automatic rejection and improves application processing',
                'competitive_advantage': 'Professional accessibility demonstrates attention to detail and facilitates interview scheduling',
                'evidence': 'Resume includes both email and phone contact methods for professional accessibility'
            })
        
        # Educational background
        if sections.get('has_education'):
            education_mentions = sections.get('education_mention_count', 0)
            if education_mentions >= 3:
                strengths.append({
                    'strength': 'Strong Educational Foundation and Presentation',
                    'why_its_strong': 'Well-documented educational background demonstrates formal training and continuous learning',
                    'ats_benefit': 'Educational keywords improve matching for roles requiring specific educational qualifications',
                    'competitive_advantage': 'Formal education provides credibility and shows foundational knowledge',
                    'evidence': f'Resume contains {education_mentions} educational elements showing strong academic foundation'
                })
        
        return strengths
    
    def _analyze_contact_weaknesses(self, sections):
        """Analyze contact information weaknesses"""
        weaknesses = []
        
        if not sections.get('email'):
            weaknesses.append({
                'weakness': 'Missing Professional Email Address',
                'why_problematic': 'Email is the primary communication method for recruiters and hiring managers in professional settings',
                'ats_impact': 'ATS systems flag incomplete contact information as low-quality submissions, often resulting in automatic rejection',
                'how_it_hurts': 'Applications may be automatically discarded before human review, completely eliminating job opportunities',
                'fix_priority': 'CRITICAL - Must be fixed immediately before any job applications',
                'specific_fix': 'Add a professional email address (firstname.lastname@gmail.com) prominently in the resume header',
                'timeline': 'Fix within 24 hours - this prevents all job applications from being considered'
            })
        
        if not sections.get('phone'):
            weaknesses.append({
                'weakness': 'Missing Phone Number Contact Information',
                'why_problematic': 'Phone contact provides alternative communication method and enables direct recruiter outreach',
                'ats_impact': 'Incomplete contact information reduces ATS scoring and may trigger rejection filters',
                'how_it_hurts': 'Recruiters cannot make direct contact for urgent opportunities or quick clarifications',
                'fix_priority': 'HIGH - Important for professional accessibility',
                'specific_fix': 'Add formatted phone number (XXX) XXX-XXXX in the contact section',
                'timeline': 'Add within 2-3 days to improve professional accessibility'
            })
        
        return weaknesses
    
    def _analyze_technical_weaknesses(self, text, sections, target_role):
        """Analyze technical skills and competency weaknesses"""
        weaknesses = []
        text_lower = text.lower()
        
        # Insufficient technical skills
        skills_count = sections.get('skills_count', 0)
        if skills_count < 6:
            weaknesses.append({
                'weakness': f'Limited Technical Skills Documentation ({skills_count} skills listed)',
                'why_problematic': 'Modern technical roles require diverse skill sets, and limited skills suggest narrow expertise',
                'ats_impact': 'Low keyword count significantly reduces ATS matching scores across technical job categories',
                'how_it_hurts': 'Appears less qualified compared to candidates with comprehensive technical profiles',
                'fix_priority': 'HIGH - Critical for technical role competitiveness',
                'specific_fix': 'Expand skills section to 10-15 relevant technical skills including programming languages, tools, and frameworks',
                'timeline': '1-2 weeks to research and add relevant skills you possess'
            })
        
        # Missing skills section entirely
        if not sections.get('skills_text'):
            weaknesses.append({
                'weakness': 'No Dedicated Technical Skills Section',
                'why_problematic': 'Technical skills section is essential for ATS parsing and recruiter evaluation',
                'ats_impact': 'Missing skills section severely impacts ATS keyword matching and categorization',
                'how_it_hurts': 'Resume may not be categorized correctly for technical roles, missing relevant opportunities',
                'fix_priority': 'CRITICAL - Essential for technical role applications',
                'specific_fix': 'Create prominent "Technical Skills" section with programming languages, frameworks, tools, and methodologies',
                'timeline': 'Add immediately - critical component of technical resumes'
            })
        
        # Role-specific skill gaps
        if target_role and target_role.lower().replace(' ', '_') in self.ats_keywords:
            role_data = self.ats_keywords[target_role.lower().replace(' ', '_')]
            core_skills = role_data.get('core_skills', [])
            missing_core = [skill for skill in core_skills[:6] if skill.lower() not in text_lower]
            
            if len(missing_core) >= 3:
                weaknesses.append({
                    'weakness': f'Missing Critical {target_role} Technical Skills',
                    'why_problematic': f'Core technical requirements for {target_role} positions are not demonstrated in the resume',
                    'ats_impact': f'Low skill matching significantly reduces ATS scores for {target_role} job applications',
                    'how_it_hurts': f'Resume will be filtered out of {target_role} candidate pools before human review',
                    'fix_priority': f'HIGH - Essential for {target_role} role targeting',
                    'specific_fix': f'Add these critical {target_role} skills: {", ".join(missing_core[:4])}',
                    'timeline': f'2-4 weeks to learn and add core {target_role} competencies'
                })
        
        # Low technical depth
        technical_depth = sections.get('technical_depth_score', 0)
        if technical_depth < 3:
            weaknesses.append({
                'weakness': 'Limited Technical Depth and Sophistication',
                'why_problematic': 'Technical roles require demonstration of complex problem-solving and advanced technical concepts',
                'ats_impact': 'Simple technical language results in lower ATS scoring for senior technical positions',
                'how_it_hurts': 'Appears suited only for junior roles, limiting career advancement opportunities',
                'fix_priority': 'MEDIUM - Important for senior role positioning',
                'specific_fix': 'Add technical details about system architecture, optimization challenges, and advanced concepts you\'ve worked with',
                'timeline': '1-2 weeks to revise project and experience descriptions with technical depth'
            })
        
        return weaknesses
    
    def _analyze_experience_weaknesses(self, sections):
        """Analyze experience-related weaknesses"""
        weaknesses = []
        experience_years = sections.get('experience_years', 0)
        experience_quality = sections.get('experience_quality', 0)
        
        # Insufficient work experience
        if experience_years == 0:
            weaknesses.append({
                'weakness': 'No Documented Professional Work Experience',
                'why_problematic': 'Most professional roles require demonstrated work history and practical application of skills',
                'ats_impact': 'Zero experience years results in automatic filtering for roles with experience requirements',
                'how_it_hurts': 'Limited to entry-level positions, significantly reducing job opportunity range',
                'fix_priority': 'HIGH - Critical for professional role applications',
                'specific_fix': 'Add internships, part-time work, freelance projects, or significant volunteer work with professional responsibilities',
                'timeline': '2-4 weeks to document and add relevant work experience or equivalent professional activities'
            })
        
        # Poor experience description quality
        if experience_quality < 30:
            weaknesses.append({
                'weakness': 'Poor Quality Experience Descriptions',
                'why_problematic': 'Experience descriptions lack action verbs, quantified achievements, and technical detail',
                'ats_impact': 'Weak descriptions result in low keyword density and poor ATS parsing performance',
                'how_it_hurts': 'Fails to demonstrate actual capabilities and achievements to potential employers',
                'fix_priority': 'HIGH - Critical for professional presentation',
                'specific_fix': 'Rewrite experience bullets using strong action verbs (developed, led, implemented) and add specific metrics and outcomes',
                'timeline': '1-2 weeks to completely rewrite experience descriptions with quantified achievements'
            })
        
        # Insufficient project portfolio
        project_count = sections.get('project_count', 0)
        if project_count < 2:
            weaknesses.append({
                'weakness': f'Inadequate Project Portfolio ({project_count} projects shown)',
                'why_problematic': 'Technical roles require demonstration of hands-on project work and practical skill application',
                'ats_impact': 'Limited project keywords reduce matching for roles emphasizing practical technical experience',
                'how_it_hurts': 'Cannot compete with candidates showing substantial hands-on technical project work',
                'fix_priority': 'MEDIUM - Important for demonstrating practical skills',
                'specific_fix': 'Add 2-3 detailed technical projects showing technology stack, challenges solved, and outcomes achieved',
                'timeline': '2-3 weeks to develop and document substantial technical projects'
            })
        
        return weaknesses
    
    def _analyze_achievement_weaknesses(self, sections):
        """Analyze quantified achievements weaknesses"""
        weaknesses = []
        achievement_count = sections.get('quantified_achievements', 0)
        
        if achievement_count == 0:
            weaknesses.append({
                'weakness': 'No Quantified Achievements or Impact Metrics',
                'why_problematic': 'Modern employers expect to see measurable business impact and performance indicators',
                'ats_impact': 'Missing quantified achievements significantly reduces ATS scores as systems look for performance metrics',
                'how_it_hurts': 'Resume appears as task-oriented rather than results-driven, reducing appeal to performance-focused employers',
                'fix_priority': 'HIGH - Critical for demonstrating value delivery',
                'specific_fix': 'Add specific percentages, dollar amounts, time savings, user numbers, or performance improvements to each major accomplishment',
                'timeline': '1-2 weeks to identify and quantify specific achievements with concrete metrics'
            })
        elif achievement_count == 1:
            weaknesses.append({
                'weakness': 'Insufficient Quantified Achievement Documentation',
                'why_problematic': 'Single quantified achievement suggests limited performance tracking or impact measurement',
                'ats_impact': 'Low achievement count reduces competitive scoring against candidates with multiple quantified results',
                'how_it_hurts': 'Appears to have limited business impact compared to results-oriented candidates',
                'fix_priority': 'MEDIUM - Important for competitive positioning',
                'specific_fix': 'Identify and add 2-3 additional quantified achievements across different projects and roles',
                'timeline': '1-2 weeks to review work history and quantify additional accomplishments'
            })
        
        return weaknesses
    
    def _analyze_content_structure_weaknesses(self, sections):
        """Analyze content structure and presentation weaknesses"""
        weaknesses = []
        word_count = sections.get('word_count', 0)
        action_verb_count = sections.get('action_verb_count', 0)
        
        # Resume length issues
        if word_count < 300:
            weaknesses.append({
                'weakness': f'Resume Too Brief ({word_count} words)',
                'why_problematic': 'Insufficient content suggests limited professional experience or poor communication skills',
                'ats_impact': 'Short resumes often fail to provide adequate keyword density for effective ATS matching',
                'how_it_hurts': 'Appears unprofessional and fails to demonstrate qualifications comprehensively',
                'fix_priority': 'MEDIUM - Important for professional presentation',
                'specific_fix': 'Expand project descriptions, experience details, and technical accomplishments to reach 400-800 words',
                'timeline': '1-2 weeks to add comprehensive details to existing sections'
            })
        elif word_count > 1200:
            weaknesses.append({
                'weakness': f'Resume Excessively Long ({word_count} words)',
                'why_problematic': 'Overly long resumes may overwhelm ATS parsing systems and recruiter attention',
                'ats_impact': 'Some ATS systems have difficulty parsing very long documents, potentially missing key information',
                'how_it_hurts': 'Recruiter may not read entire resume, missing key qualifications buried in excessive detail',
                'fix_priority': 'MEDIUM - Important for optimal ATS performance',
                'specific_fix': 'Condense content to 600-900 words by removing redundant information and focusing on most impactful achievements',
                'timeline': '1 week to edit and streamline content for optimal length'
            })
        
        # Insufficient action verb usage
        if action_verb_count < 4:
            weaknesses.append({
                'weakness': f'Limited Action Verb Usage ({action_verb_count} action verbs)',
                'why_problematic': 'Professional resumes require dynamic language to demonstrate proactive approach and leadership',
                'ats_impact': 'Action verbs are heavily weighted by ATS systems as indicators of professional competency',
                'how_it_hurts': 'Passive language suggests limited initiative and professional impact',
                'fix_priority': 'MEDIUM - Important for professional language quality',
                'specific_fix': 'Rewrite experience and project descriptions using strong action verbs like "developed," "led," "implemented," "optimized"',
                'timeline': '1 week to revise language throughout resume with dynamic action verbs'
            })
        
        return weaknesses