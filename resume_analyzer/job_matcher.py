"""
Job role matching and career guidance module
Provides detailed job compatibility analysis and career recommendations
"""

import logging
from .config import ATS_KEYWORDS, INDUSTRY_INSIGHTS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobRoleMatcher:
    """Analyzes resume compatibility with different job roles and provides career guidance"""
    
    def __init__(self):
        self.ats_keywords = ATS_KEYWORDS
        self.industry_insights = INDUSTRY_INSIGHTS
    
    def get_comprehensive_job_analysis(self, text, sections, target_role=None):
        """
        Generate detailed job role compatibility analysis with specific reasoning
        
        Args:
            text (str): Resume text content
            sections (dict): Parsed resume sections
            target_role (str): Optional target role for focused analysis
            
        Returns:
            dict: Comprehensive job analysis with recommendations
        """
        text_lower = text.lower()
        
        # Calculate compatibility scores for all roles
        role_compatibility = self._calculate_role_compatibility(text_lower)
        
        # Generate detailed role suggestions
        detailed_suggestions = self._generate_detailed_role_suggestions(
            role_compatibility, sections, text_lower
        )
        
        # Create career roadmap recommendations
        career_roadmap = self._create_career_roadmap(sections, detailed_suggestions)
        
        # Generate market insights
        market_analysis = self._generate_market_analysis(detailed_suggestions, sections)
        
        analysis_result = {
            'role_suggestions': detailed_suggestions,
            'career_roadmap': career_roadmap,
            'market_insights': market_analysis,
            'overall_readiness': self._assess_overall_job_readiness(sections)
        }
        
        logger.info(f"Generated comprehensive job analysis with {len(detailed_suggestions)} role suggestions")
        return analysis_result
    
    def _calculate_role_compatibility(self, text_lower):
        """Calculate compatibility scores for each role"""
        role_compatibility = {}
        
        for role_key, role_data in self.ats_keywords.items():
            # Collect all relevant keywords for the role
            core_skills = role_data.get('core_skills', [])
            frameworks = role_data.get('frameworks', [])
            methodologies = role_data.get('methodologies', [])
            platforms = role_data.get('platforms', [])
            
            all_keywords = core_skills + frameworks + methodologies + platforms
            
            # Calculate matches with weighted scoring
            matched_core = [skill for skill in core_skills if skill.lower() in text_lower]
            matched_frameworks = [fw for fw in frameworks if fw.lower() in text_lower]
            matched_methodologies = [method for method in methodologies if method.lower() in text_lower]
            matched_platforms = [platform for platform in platforms if platform.lower() in text_lower]
            
            # Weighted scoring (core skills are most important)
            core_score = len(matched_core) * 3
            framework_score = len(matched_frameworks) * 2
            methodology_score = len(matched_methodologies) * 1.5
            platform_score = len(matched_platforms) * 1
            
            total_score = core_score + framework_score + methodology_score + platform_score
            max_possible_score = len(core_skills) * 3 + len(frameworks) * 2 + len(methodologies) * 1.5 + len(platforms) * 1
            
            compatibility_percentage = (total_score / max_possible_score * 100) if max_possible_score > 0 else 0
            
            role_compatibility[role_key] = {
                'score': compatibility_percentage,
                'matched_core_skills': matched_core,
                'matched_frameworks': matched_frameworks,
                'matched_methodologies': matched_methodologies,
                'matched_platforms': matched_platforms,
                'missing_core_skills': [skill for skill in core_skills if skill.lower() not in text_lower],
                'missing_frameworks': [fw for fw in frameworks if fw.lower() not in text_lower],
                'role_data': role_data
            }
        
        return role_compatibility
    
    def _generate_detailed_role_suggestions(self, role_compatibility, sections, text_lower):
        """Generate detailed role suggestions with comprehensive analysis"""
        # Sort roles by compatibility score
        sorted_roles = sorted(role_compatibility.items(), key=lambda x: x[1]['score'], reverse=True)
        
        detailed_suggestions = []
        experience_years = sections.get('experience_years', 0)
        
        for role_key, compatibility_data in sorted_roles[:6]:  # Top 6 roles
            role_name = role_key.replace('_', ' ').title()
            score = compatibility_data['score']
            role_data = compatibility_data['role_data']
            industry_info = self.industry_insights.get(role_key, {})
            
            # Determine fit level and explanation
            fit_assessment = self._assess_role_fit(score, compatibility_data)
            
            # Generate seniority and salary expectations
            seniority_assessment = self._assess_seniority_level(experience_years, score)
            
            # Create comprehensive role analysis
            suggestion = {
                'role': role_name,
                'compatibility_score': f"{score:.1f}%",
                'fit_level': fit_assessment['level'],
                'fit_explanation': fit_assessment['explanation'],
                'readiness_timeline': fit_assessment['timeline'],
                
                'technical_alignment': {
                    'core_skills_matched': compatibility_data['matched_core_skills'],
                    'frameworks_matched': compatibility_data['matched_frameworks'],
                    'methodologies_matched': compatibility_data['matched_methodologies'],
                    'platforms_matched': compatibility_data['matched_platforms'],
                    'skill_gaps': {
                        'critical': compatibility_data['missing_core_skills'][:3],
                        'important': compatibility_data['missing_frameworks'][:3],
                        'beneficial': compatibility_data['missing_core_skills'][3:6]
                    }
                },
                
                'role_details': {
                    'daily_responsibilities': role_data.get('daily_tasks', []),
                    'required_tech_stack': role_data.get('tech_stack', 'Not specified'),
                    'typical_projects': self._generate_typical_projects(role_key),
                    'career_progression': self._generate_career_progression(role_key, experience_years)
                },
                
                'seniority_analysis': seniority_assessment,
                
                'market_insights': {
                    'growth_outlook': industry_info.get('growth_outlook', 'Stable market demand'),
                    'salary_range': industry_info.get('salary_range', 'Competitive market rates'),
                    'key_employers': industry_info.get('key_companies', []),
                    'trending_technologies': industry_info.get('trending_skills', []),
                    'job_availability': self._assess_job_availability(role_key, score)
                },
                
                'development_plan': {
                    'immediate_actions': self._generate_immediate_actions(compatibility_data, role_name),
                    'skill_development_priority': self._prioritize_skill_development(compatibility_data, role_key),
                    'learning_resources': self._suggest_learning_resources(role_key),
                    'networking_strategy': self._suggest_networking_strategy(role_key)
                },
                
                'application_strategy': {
                    'keyword_optimization': self._generate_keyword_strategy(compatibility_data, role_name),
                    'resume_customization': self._generate_resume_customization_tips(compatibility_data, role_name),
                    'portfolio_recommendations': self._generate_portfolio_recommendations(role_key),
                    'interview_preparation': self._generate_interview_prep_tips(role_key)
                }
            }
            
            detailed_suggestions.append(suggestion)
        
        return detailed_suggestions
    
    def _assess_role_fit(self, score, compatibility_data):
        """Assess role fit level and provide detailed explanation"""
        if score >= 80:
            return {
                'level': 'Excellent Fit',
                'explanation': 'Strong technical skill alignment with comprehensive role requirements. You demonstrate most core competencies.',
                'timeline': 'Ready to apply immediately - start targeting premium positions'
            }
        elif score >= 65:
            return {
                'level': 'Very Good Fit',
                'explanation': 'Solid technical foundation with minor skill gaps. Strong candidate for most positions in this role.',
                'timeline': '2-4 weeks of targeted preparation to become highly competitive'
            }
        elif score >= 50:
            return {
                'level': 'Good Fit',
                'explanation': 'Reasonable technical foundation with some important skill gaps to address. Suitable for junior to mid-level positions.',
                'timeline': '2-3 months of focused skill development to become competitive'
            }
        elif score >= 35:
            return {
                'level': 'Potential Fit',
                'explanation': 'Basic technical skills present but significant development needed. Consider entry-level or career transition roles.',
                'timeline': '4-6 months of intensive skill building required'
            }
        else:
            return {
                'level': 'Limited Fit',
                'explanation': 'Major skill gaps present. Recommend substantial preparation or consider alternative career paths.',
                'timeline': '6-12 months of comprehensive skill development needed'
            }
    
    def _assess_seniority_level(self, experience_years, compatibility_score):
        """Assess appropriate seniority level for the role"""
        # Base seniority on experience
        if experience_years >= 8:
            base_level = 'Senior/Lead'
            salary_multiplier = 1.3
        elif experience_years >= 5:
            base_level = 'Senior'
            salary_multiplier = 1.2
        elif experience_years >= 3:
            base_level = 'Mid-Level'
            salary_multiplier = 1.0
        elif experience_years >= 1:
            base_level = 'Junior'
            salary_multiplier = 0.8
        else:
            base_level = 'Entry-Level'
            salary_multiplier = 0.7
        
        # Adjust based on technical competency
        if compatibility_score >= 80:
            competency_adjustment = 'High technical competency may qualify for senior positions'
        elif compatibility_score >= 65:
            competency_adjustment = 'Good technical skills align well with stated experience level'
        elif compatibility_score < 50:
            competency_adjustment = 'Technical skills may limit to entry-level positions initially'
        else:
            competency_adjustment = 'Technical skills are appropriate for current experience level'
        
        return {
            'suggested_level': base_level,
            'competency_assessment': competency_adjustment,
            'salary_expectation': f'{salary_multiplier:.1f}x market rate for role',
            'advancement_potential': self._assess_advancement_potential(experience_years, compatibility_score)
        }
    
    def _assess_advancement_potential(self, experience_years, compatibility_score):
        """Assess career advancement potential based on experience and skills"""
        if experience_years >= 8 and compatibility_score >= 80:
            return "Excellent advancement potential - ready for senior/leadership positions"
        elif experience_years >= 5 and compatibility_score >= 70:
            return "Strong advancement potential - focus on leadership and strategic skills"
        elif experience_years >= 3 and compatibility_score >= 60:
            return "Good advancement potential - continue technical depth and leadership development"
        elif experience_years >= 1 and compatibility_score >= 50:
            return "Moderate advancement potential - focus on skill development and experience building"
        else:
            return "Limited immediate advancement potential - prioritize foundational skill development"
    
    def _generate_typical_projects(self, role_key):
        """Generate typical project examples for each role"""
        project_examples = {
            'software_developer': [
                'Web application development using modern frameworks',
                'API design and backend service implementation',
                'Database design and optimization projects',
                'Code review and quality assurance initiatives'
            ],
            'data_scientist': [
                'Predictive modeling and machine learning implementation',
                'Data pipeline development and ETL processes',
                'Business intelligence dashboard creation',
                'A/B testing and statistical analysis projects'
            ],
            'ai_engineer': [
                'Machine learning model development and deployment',
                'Computer vision and natural language processing projects',
                'AI system architecture and scalability optimization',
                'Research implementation and algorithm development'
            ],
            'full_stack_developer': [
                'End-to-end web application development',
                'Frontend user interface and user experience design',
                'Backend API and database integration',
                'Cross-platform mobile application development'
            ],
            'devops_engineer': [
                'CI/CD pipeline development and automation',
                'Infrastructure as code implementation',
                'Container orchestration and cloud deployment',
                'Monitoring and security system implementation'
            ]
        }
        
        return project_examples.get(role_key, ['Role-specific project development', 'Technical problem solving', 'System optimization', 'Team collaboration'])
    
    def _generate_career_progression(self, role_key, experience_years):
        """Generate career progression path for each role"""
        progressions = {
            'software_developer': {
                'entry': 'Junior Developer → Software Developer → Senior Developer → Tech Lead → Engineering Manager',
                'current_stage': self._determine_career_stage(experience_years, ['Junior Developer', 'Software Developer', 'Senior Developer', 'Tech Lead', 'Engineering Manager']),
                'next_milestone': 'Focus on technical leadership and mentoring junior developers'
            },
            'data_scientist': {
                'entry': 'Data Analyst → Data Scientist → Senior Data Scientist → Principal Data Scientist → Head of Data Science',
                'current_stage': self._determine_career_stage(experience_years, ['Data Analyst', 'Data Scientist', 'Senior Data Scientist', 'Principal Data Scientist', 'Head of Data Science']),
                'next_milestone': 'Develop expertise in MLOps and business strategy'
            },
            'ai_engineer': {
                'entry': 'ML Engineer → AI Engineer → Senior AI Engineer → Principal AI Engineer → AI Research Director',
                'current_stage': self._determine_career_stage(experience_years, ['ML Engineer', 'AI Engineer', 'Senior AI Engineer', 'Principal AI Engineer', 'AI Research Director']),
                'next_milestone': 'Focus on cutting-edge AI research and system architecture'
            }
        }
        
        default_progression = {
            'entry': 'Entry Level → Mid Level → Senior Level → Lead Level → Management Level',
            'current_stage': f'Estimated at {self._determine_career_stage(experience_years, ["Entry Level", "Mid Level", "Senior Level", "Lead Level", "Management Level"])}',
            'next_milestone': 'Focus on expanding technical expertise and leadership skills'
        }
        
        return progressions.get(role_key, default_progression)
    
    def _determine_career_stage(self, experience_years, stages):
        """Determine current career stage based on experience"""
        if experience_years == 0:
            return stages[0] if stages else "Entry Level"
        elif experience_years <= 2:
            return stages[1] if len(stages) > 1 else "Junior Level"
        elif experience_years <= 5:
            return stages[2] if len(stages) > 2 else "Mid Level"
        elif experience_years <= 8:
            return stages[3] if len(stages) > 3 else "Senior Level"
        else:
            return stages[4] if len(stages) > 4 else "Leadership Level"
    
    def _assess_job_availability(self, role_key, compatibility_score):
        """Assess job market availability for the role"""
        market_conditions = {
            'software_developer': 'High demand across all industries',
            'data_scientist': 'Strong demand in tech and finance sectors',
            'ai_engineer': 'Rapidly growing demand, especially in AI companies',
            'full_stack_developer': 'Consistent high demand for versatile developers',
            'devops_engineer': 'High demand as companies adopt cloud infrastructure'
        }
        
        base_availability = market_conditions.get(role_key, 'Moderate demand in various industries')
        
        if compatibility_score >= 70:
            return f"{base_availability} - Your strong skill match provides excellent opportunities"
        elif compatibility_score >= 50:
            return f"{base_availability} - Good opportunities with targeted skill development"
        else:
            return f"{base_availability} - Limited immediate opportunities, skill development recommended"
    
    def _generate_immediate_actions(self, compatibility_data, role_name):
        """Generate immediate actionable steps for role preparation"""
        actions = []
        
        missing_core = compatibility_data['missing_core_skills'][:2]
        if missing_core:
            actions.append(f"Priority learning: Start with {', '.join(missing_core)} as core {role_name} requirements")
        
        matched_skills = len(compatibility_data['matched_core_skills'])
        if matched_skills > 0:
            actions.append(f"Leverage strengths: Highlight your {matched_skills} matching core skills in applications")
        
        missing_frameworks = compatibility_data['missing_frameworks'][:2]
        if missing_frameworks:
            actions.append(f"Technical expansion: Learn {', '.join(missing_frameworks)} to improve technical stack")
        
        actions.append(f"Resume optimization: Add {role_name}-specific keywords and project examples")
        actions.append("Networking: Connect with professionals in target role through LinkedIn and industry events")
        
        return actions
    
    def _prioritize_skill_development(self, compatibility_data, role_key):
        """Create prioritized skill development plan"""
        return {
            'critical_skills': {
                'skills': compatibility_data['missing_core_skills'][:2],
                'timeline': '1-2 months',
                'reason': 'Essential for basic role competency'
            },
            'important_skills': {
                'skills': compatibility_data['missing_frameworks'][:2],
                'timeline': '2-3 months',
                'reason': 'Required for competitive positioning'
            },
            'enhancement_skills': {
                'skills': compatibility_data['missing_core_skills'][2:4],
                'timeline': '3-6 months',
                'reason': 'Advanced competency and specialization'
            }
        }
    
    def _suggest_learning_resources(self, role_key):
        """Suggest learning resources for each role"""
        resources = {
            'software_developer': [
                'FreeCodeCamp for practical coding practice',
                'LeetCode for algorithm and data structure preparation',
                'GitHub for open source contribution experience',
                'Coursera/Udemy for framework-specific courses'
            ],
            'data_scientist': [
                'Kaggle for hands-on data science competitions',
                'Coursera Data Science Specialization',
                'Python for Data Science courses',
                'Statistics and machine learning fundamentals'
            ],
            'ai_engineer': [
                'Fast.ai for practical deep learning',
                'PyTorch/TensorFlow official tutorials',
                'Papers With Code for latest research',
                'Google AI Education resources'
            ]
        }
        
        return resources.get(role_key, [
            'Industry-specific online courses',
            'Professional certification programs',
            'Hands-on project development',
            'Technical blog and documentation reading'
        ])
    
    def _suggest_networking_strategy(self, role_key):
        """Suggest networking strategies for each role"""
        strategies = {
            'software_developer': 'Join local developer meetups, contribute to open source, attend tech conferences',
            'data_scientist': 'Participate in Kaggle competitions, join data science communities, attend ML conferences',
            'ai_engineer': 'Follow AI research communities, attend AI/ML meetups, contribute to AI open source projects',
            'full_stack_developer': 'Join full-stack developer groups, participate in hackathons, showcase projects online',
            'devops_engineer': 'Engage with DevOps communities, attend cloud provider events, join infrastructure forums'
        }
        
        return strategies.get(role_key, 'Connect with industry professionals, attend relevant meetups and conferences')
    
    def _generate_keyword_strategy(self, compatibility_data, role_name):
        """Generate ATS keyword optimization strategy"""
        matched_skills = compatibility_data['matched_core_skills'] + compatibility_data['matched_frameworks']
        missing_skills = compatibility_data['missing_core_skills'][:3]
        
        return {
            'strengthen_existing': f"Emphasize these matching skills: {', '.join(matched_skills[:5])}",
            'add_missing': f"Add these critical keywords: {', '.join(missing_skills)}",
            'context_usage': f"Use {role_name} terminology naturally throughout experience descriptions",
            'density_target': f"Include {role_name} keywords 8-12 times across resume sections"
        }
    
    def _generate_resume_customization_tips(self, compatibility_data, role_name):
        """Generate role-specific resume customization advice"""
        return {
            'title_optimization': f"Consider job title variations like '{role_name}', 'Senior {role_name}', '{role_name} Specialist'",
            'skills_section': f"Lead skills section with top {role_name} requirements",
            'experience_framing': f"Frame past experience using {role_name} perspective and terminology",
            'project_selection': f"Highlight projects most relevant to {role_name} responsibilities"
        }
    
    def _generate_portfolio_recommendations(self, role_key):
        """Generate portfolio recommendations for each role"""
        portfolios = {
            'software_developer': 'GitHub with 3-5 complete applications showing different technologies',
            'data_scientist': 'Portfolio with data analysis projects, Jupyter notebooks, and visualizations',
            'ai_engineer': 'ML model projects with deployment examples and technical documentation',
            'full_stack_developer': 'Live web applications demonstrating frontend and backend integration',
            'devops_engineer': 'Infrastructure projects showing automation, monitoring, and deployment pipelines'
        }
        
        return portfolios.get(role_key, 'Professional portfolio showcasing relevant technical projects')
    
    def _generate_interview_prep_tips(self, role_key):
        """Generate interview preparation tips for each role"""
        tips = {
            'software_developer': 'Practice coding interviews, system design questions, and technical problem-solving',
            'data_scientist': 'Prepare for statistical questions, case studies, and technical presentation of your projects',
            'ai_engineer': 'Study ML algorithms deeply, prepare to discuss model architecture and deployment challenges',
            'full_stack_developer': 'Be ready to discuss both frontend and backend technologies, architecture decisions',
            'devops_engineer': 'Prepare for infrastructure scenarios, automation examples, and troubleshooting situations'
        }
        
        return tips.get(role_key, 'Prepare technical examples, practice explaining complex concepts simply, research company technology stack')
    
    def _create_career_roadmap(self, sections, detailed_suggestions):
        """Create comprehensive career development roadmap"""
        current_experience = sections.get('experience_years', 0)
        top_role = detailed_suggestions[0] if detailed_suggestions else None
        
        if not top_role:
            return {'message': 'Unable to generate career roadmap without role analysis'}
        
        return {
            'current_position': f"Professional with {current_experience} years experience",
            'recommended_target': f"{top_role['role']} ({top_role['fit_level']})",
            'development_timeline': top_role['readiness_timeline'],
            'skill_development_plan': top_role['development_plan']['skill_development_priority'],
            'milestones': {
                '30_days': 'Complete critical skill gap assessment and begin learning top priority skills',
                '90_days': 'Develop practical projects demonstrating target role competencies',
                '6_months': 'Build comprehensive portfolio and begin targeted job applications',
                '12_months': 'Secure position in target role with competitive compensation'
            },
            'success_metrics': [
                'Technical skill proficiency in core competencies',
                'Portfolio demonstrating practical application',
                'Interview performance and offer conversion rate',
                'Professional network expansion in target field'
            ]
        }
    
    def _generate_market_analysis(self, detailed_suggestions, sections):
        """Generate comprehensive market analysis"""
        if not detailed_suggestions:
            return {'message': 'Unable to generate market analysis'}
        
        top_three_roles = detailed_suggestions[:3]
        experience_years = sections.get('experience_years', 0)
        
        return {
            'market_opportunities': {
                'high_compatibility_roles': [role['role'] for role in top_three_roles if float(role['compatibility_score'].replace('%', '')) >= 60],
                'emerging_opportunities': [role['role'] for role in top_three_roles if 'AI' in role['role'] or 'Data' in role['role']],
                'stable_career_paths': [role['role'] for role in top_three_roles if role['role'] in ['Software Developer', 'Full Stack Developer']]
            },
            'salary_expectations': {
                'current_level_range': self._estimate_salary_range(experience_years, top_three_roles[0]['compatibility_score']),
                'growth_potential': 'Strong upward trajectory with skill development',
                'premium_positions': f"Top-tier companies may offer 20-30% above market rate for {top_three_roles[0]['role']} skills"
            },
            'industry_trends': {
                'hot_skills': ['Cloud Computing', 'AI/ML', 'Data Analysis', 'DevOps', 'Cybersecurity'],
                'declining_demand': ['Legacy system maintenance', 'Non-cloud infrastructure'],
                'future_growth_areas': ['AI Engineering', 'Cloud Architecture', 'Data Engineering']
            },
            'geographic_considerations': {
                'tech_hubs': ['San Francisco Bay Area', 'Seattle', 'New York', 'Austin', 'Boston'],
                'remote_opportunities': 'High for technical roles, especially post-COVID',
                'international_markets': 'Strong demand in Europe, Canada, and emerging tech markets'
            }
        }
    
    def _assess_overall_job_readiness(self, sections):
        """Assess overall job market readiness"""
        experience_years = sections.get('experience_years', 0)
        skills_count = sections.get('skills_count', 0)
        project_count = sections.get('project_count', 0)
        achievements = sections.get('quantified_achievements', 0)
        
        readiness_score = 0
        factors = []
        
        # Experience factor
        if experience_years >= 3:
            readiness_score += 30
            factors.append("Strong professional experience base")
        elif experience_years >= 1:
            readiness_score += 20
            factors.append("Some professional experience")
        else:
            factors.append("Limited professional experience - focus on projects and skills")
        
        # Skills factor
        if skills_count >= 12:
            readiness_score += 25
            factors.append("Comprehensive technical skill set")
        elif skills_count >= 6:
            readiness_score += 15
            factors.append("Good technical foundation")
        else:
            factors.append("Limited technical skills documented")
        
        # Projects factor
        if project_count >= 3:
            readiness_score += 25
            factors.append("Strong project portfolio")
        elif project_count >= 1:
            readiness_score += 15
            factors.append("Some project experience")
        else:
            factors.append("Insufficient project demonstration")
        
        # Achievements factor
        if achievements >= 2:
            readiness_score += 20
            factors.append("Quantified professional achievements")
        elif achievements >= 1:
            readiness_score += 10
            factors.append("Some measurable impact shown")
        else:
            factors.append("No quantified achievements documented")
        
        # Determine overall readiness level
        if readiness_score >= 80:
            level = "Highly Job Ready"
            recommendation = "Excellent profile for immediate job applications"
        elif readiness_score >= 60:
            level = "Job Ready"
            recommendation = "Good profile with minor improvements recommended"
        elif readiness_score >= 40:
            level = "Moderately Ready"
            recommendation = "Solid foundation but needs targeted improvements"
        else:
            level = "Needs Development"
            recommendation = "Significant improvements required before job applications"
        
        return {
            'readiness_level': level,
            'readiness_score': f"{readiness_score}/100",
            'key_factors': factors,
            'overall_recommendation': recommendation,
            'next_steps': self._generate_readiness_next_steps(readiness_score)
        }
    
    def _estimate_salary_range(self, experience_years, compatibility_score):
        """Estimate salary range based on experience and competency"""
        base_ranges = {
            0: (45000, 70000),
            1: (55000, 85000),
            2: (65000, 100000),
            3: (75000, 120000),
            5: (90000, 140000),
            8: (110000, 170000)
        }
        
        # Find appropriate experience bracket
        exp_bracket = min(experience_years, 8)
        for bracket in [0, 1, 2, 3, 5, 8]:
            if experience_years >= bracket:
                exp_bracket = bracket
        
        base_min, base_max = base_ranges[exp_bracket]
        
        # Adjust based on technical competency
        compatibility_float = float(compatibility_score.replace('%', ''))
        if compatibility_float >= 75:
            multiplier = 1.15
        elif compatibility_float >= 60:
            multiplier = 1.0
        else:
            multiplier = 0.9
        
        adjusted_min = int(base_min * multiplier)
        adjusted_max = int(base_max * multiplier)
        
        return f"${adjusted_min:,} - ${adjusted_max:,}"
    
    def _generate_readiness_next_steps(self, readiness_score):
        """Generate next steps based on readiness score"""
        if readiness_score >= 80:
            return [
                "Begin applying to target companies immediately",
                "Focus on interview preparation and negotiation",
                "Continue skill development in trending technologies",
                "Build professional network in target industry"
            ]
        elif readiness_score >= 60:
            return [
                "Address top 2-3 resume weaknesses before applying",
                "Develop 1-2 additional portfolio projects",
                "Practice technical interviewing",
                "Target mid-level positions aligned with experience"
            ]
        elif readiness_score >= 40:
            return [
                "Focus on skill development for 2-3 months",
                "Build comprehensive project portfolio",
                "Add quantified achievements to resume",
                "Consider targeted learning programs or bootcamps"
            ]
        else:
            return [
                "Complete comprehensive skill assessment",
                "Enroll in structured learning program",
                "Develop foundational projects over 6 months",
                "Build basic professional portfolio and network"
            ]