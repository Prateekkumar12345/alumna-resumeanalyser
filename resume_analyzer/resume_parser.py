"""
Resume parsing and content extraction module
Handles detailed resume section analysis and information extraction
"""

import re
import logging
from datetime import datetime
from .config import REGEX_PATTERNS, ACTION_VERBS, TECHNICAL_CONCEPTS, CURRENT_YEAR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeParser:
    """Handles comprehensive resume parsing and content extraction"""
    
    def __init__(self):
        self.patterns = REGEX_PATTERNS
        self.action_verbs = ACTION_VERBS
        self.technical_concepts = TECHNICAL_CONCEPTS
        self.current_year = CURRENT_YEAR
    
    def extract_comprehensive_sections(self, text):
        """
        Extract comprehensive resume information with improved accuracy and detailed analysis
        
        Args:
            text (str): Resume text content
            
        Returns:
            dict: Comprehensive sections data with detailed metrics
        """
        if not text:
            logger.error("No text provided for parsing")
            return {}
        
        sections = {}
        text_lower = text.lower()
        
        # Extract contact information with enhanced validation
        sections.update(self._extract_contact_info(text))
        
        # Extract and analyze skills comprehensively
        sections.update(self._extract_skills_analysis(text))
        
        # Extract and calculate experience with detailed breakdown
        sections.update(self._extract_experience_analysis(text))
        
        # Extract and analyze projects with quality assessment
        sections.update(self._extract_project_analysis(text))
        
        # Extract education information with validation
        sections.update(self._extract_education_info(text))
        
        # Analyze quantified achievements with categorization
        sections.update(self._analyze_quantified_achievements(text))
        
        # Assess technical depth and sophistication
        sections.update(self._analyze_technical_depth(text))
        
        # Analyze content quality and structure
        sections.update(self._analyze_content_quality(text))
        
        # Generate comprehensive summary statistics
        sections['analysis_summary'] = self._generate_analysis_summary(sections)
        
        logger.info(f"Extracted {len(sections)} comprehensive section categories")
        return sections
    
    def _extract_contact_info(self, text):
        """Extract and validate contact information"""
        contact_info = {}
        
        # Email extraction with validation
        emails = re.findall(self.patterns['email'], text)
        if emails:
            # Filter out potentially invalid emails
            valid_emails = [email for email in emails if self._validate_email(email)]
            contact_info['email'] = valid_emails[0] if valid_emails else None
            contact_info['email_count'] = len(valid_emails)
        else:
            contact_info['email'] = None
            contact_info['email_count'] = 0
        
        # Enhanced phone number extraction with international format support
        phone_patterns = [
            r'(\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4})',  # International format
            r'(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})',  # Standard US format
            r'(\d{5}[-.\s]?\d{5})',  # Indian format (10 digits, sometimes with separator)
            r'(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})'  # Standard format without country code
        ]
        
        phones = []
        for pattern in phone_patterns:
            phones.extend(re.findall(pattern, text))
        
        if phones:
            # Clean and format phone numbers
            clean_phones = [self._clean_phone_number(phone) for phone in phones]
            contact_info['phone'] = clean_phones[0] if clean_phones else None
            contact_info['phone_count'] = len(clean_phones)
            contact_info['phone_numbers'] = clean_phones  # Store all found numbers
        else:
            contact_info['phone'] = None
            contact_info['phone_count'] = 0
            contact_info['phone_numbers'] = []
        
        return contact_info
    
    def _clean_phone_number(self, phone):
        
        """Clean and format phone number with international support"""
        if isinstance(phone, tuple):
            phone = ''.join(phone)
    
    # Remove all non-digits except + for international numbers
        if phone.startswith('+'):
            cleaned = re.sub(r'[^\d+]', '', phone)
        else:
            cleaned = re.sub(r'[^\d]', '', phone)
    
    # Validate length
        if len(cleaned) >= 10:
            return cleaned
        return None
    
    def _extract_skills_analysis(self, text):
        """Comprehensive skills extraction and analysis"""
        skills_data = {}
        
        # Skills section extraction with multiple patterns
        skills_patterns = [
            r'(?i)(?:technical\s+)?skills?\s*:?\s*([^\n]+(?:\n(?!\s*[A-Z][^:\n]*:)[^\n]+)*)',
            r'(?i)technologies?\s*:?\s*([^\n]+(?:\n(?!\s*[A-Z][^:\n]*:)[^\n]+)*)',
            r'(?i)programming\s+languages?\s*:?\s*([^\n]+)',
            r'(?i)tools?\s*(?:and|&)?\s*technologies?\s*:?\s*([^\n]+)',
            r'(?i)core\s+competencies\s*:?\s*([^\n]+(?:\n(?!\s*[A-Z][^:\n]*:)[^\n]+)*)'
        ]
        
        all_skills_text = []
        for pattern in skills_patterns:
            matches = re.findall(pattern, text)
            all_skills_text.extend(matches)
        
        combined_skills = ' '.join(all_skills_text)
        skills_data['skills_text'] = combined_skills
        skills_data['skills_word_count'] = len(combined_skills.split()) if combined_skills else 0
        
        # Extract individual skills by parsing comma-separated values
        if combined_skills:
            # Split by common delimiters
            individual_skills = re.split(r'[,;|•\n]+', combined_skills)
            individual_skills = [skill.strip() for skill in individual_skills if skill.strip()]
            skills_data['individual_skills'] = individual_skills[:20]  # Limit to top 20
            skills_data['skills_count'] = len(individual_skills)
        else:
            skills_data['individual_skills'] = []
            skills_data['skills_count'] = 0
        
        # Categorize skills by type
        skills_data['skill_categories'] = self._categorize_skills(combined_skills)
        
        return skills_data
    
    def _extract_experience_analysis(self, text):
        """Comprehensive experience extraction and analysis"""
        experience_data = {}
        
        # Extract experience section
        experience_section_pattern = r'(?i)(experience|work\s+history|employment|professional\s+experience)(.*?)(?=(education|projects|skills|awards|certifications|$))'
        experience_match = re.search(experience_section_pattern, text, re.DOTALL | re.IGNORECASE)
        
        experience_text = experience_match.group(2) if experience_match else text
        
        # Calculate total years of experience using multiple methods
        total_experience = self._calculate_total_experience(experience_text, text)
        experience_data['experience_years'] = total_experience
        experience_data['experience_level'] = self._classify_experience_level(total_experience)
        
        # Count number of positions/roles
        position_indicators = [
            r'(?i)(software\s+engineer|developer|analyst|scientist|manager|lead|senior|junior)',
            r'(?i)(intern|internship|co-op)',
            r'(?i)(consultant|contractor|freelancer)'
        ]
        
        position_count = 0
        for pattern in position_indicators:
            position_count += len(re.findall(pattern, experience_text))
        
        experience_data['position_count'] = position_count
        experience_data['has_internship'] = bool(re.search(r'(?i)intern', experience_text))
        experience_data['has_leadership'] = bool(re.search(r'(?i)(lead|manager|supervisor|team\s+lead)', experience_text))
        
        # Analyze job descriptions quality
        experience_data['experience_quality'] = self._analyze_experience_quality(experience_text)
        
        return experience_data
    
    def _extract_project_analysis(self, text):
        """Comprehensive project extraction and analysis"""
        project_data = {}
        
        # Extract projects section
        project_section_pattern = r'(?i)(projects|personal\s+projects|portfolio|academic\s+projects)(.*?)(?=(experience|education|skills|awards|$))'
        project_match = re.search(project_section_pattern, text, re.DOTALL | re.IGNORECASE)
        
        project_count = 0
        project_descriptions = []
        project_quality_scores = []
        
        if project_match:
            project_text = project_match.group(2)
            
            # Count projects using multiple patterns
            project_patterns = [
                r'(?i)^\s*[•\-*]\s*(.+?)(?=\n|$)',  # Bullet points
                r'(?i)project\s*\d*\s*[:\-]?\s*(.+?)(?=\n|$)',  # "Project:" format
                r'(?i)^\s*\d+\.\s*(.+?)(?=\n|$)'  # Numbered items
            ]
            
            for pattern in project_patterns:
                matches = re.findall(pattern, project_text, re.MULTILINE)
                for match in matches:
                    if len(match.strip()) > 10:  # Filter out very short descriptions
                        project_descriptions.append(match.strip())
                        project_quality_scores.append(self._assess_project_quality(match))
            
            project_count = len(project_descriptions)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_projects = []
        for proj in project_descriptions:
            if proj not in seen:
                seen.add(proj)
                unique_projects.append(proj)
        
        project_data['project_count'] = len(unique_projects)
        project_data['project_descriptions'] = unique_projects[:5]  # Top 5 projects
        project_data['average_project_quality'] = sum(project_quality_scores) / len(project_quality_scores) if project_quality_scores else 0
        project_data['has_github_links'] = self._count_github_references(text)
        project_data['has_live_demos'] = self._count_demo_references(text)
        
        return project_data
    
    def _extract_education_info(self, text):
        """Extract and analyze education information"""
        education_data = {}
        
        # Enhanced education detection
        education_patterns = [
            r'(?i)(education|academic\s+background)',
            r'(?i)(bachelor|master|phd|doctorate|b\.?tech|m\.?tech|b\.?sc|m\.?sc|b\.?a|m\.?a)',
            r'(?i)(university|college|institute|school)',
            r'(?i)(degree|diploma|certification|graduate)'
        ]
        
        education_mentions = 0
        education_keywords = []
        
        for pattern in education_patterns:
            matches = re.findall(pattern, text)
            education_mentions += len(matches)
            education_keywords.extend(matches)
        
        education_data['has_education'] = education_mentions > 0
        education_data['education_mention_count'] = education_mentions
        education_data['education_keywords'] = list(set(education_keywords))[:10]
        
        # Check for GPA mentions
        gpa_pattern = r'(?i)gpa[:\s]*([0-9.]+)'
        gpa_matches = re.findall(gpa_pattern, text)
        education_data['has_gpa'] = len(gpa_matches) > 0
        education_data['gpa_value'] = gpa_matches[0] if gpa_matches else None
        
        return education_data
    
    def _analyze_quantified_achievements(self, text):
        """Analyze quantified achievements with categorization"""
        achievement_data = {}
        
        all_achievements = []
        achievement_categories = {
            'performance_metrics': [],
            'financial_impact': [],
            'scale_metrics': [],
            'time_improvements': [],
            'quality_metrics': []
        }
        
        # Enhanced achievement patterns with categorization
        for pattern in self.patterns['quantified_achievements']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                all_achievements.append(match)
                # Categorize the achievement
                if any(word in match.lower() for word in ['%', 'percent', 'increase', 'improve', 'reduce']):
                    achievement_categories['performance_metrics'].append(match)
                elif any(symbol in match for symbol in ['$', 'revenue', 'cost', 'profit', 'budget']):
                    achievement_categories['financial_impact'].append(match)
                elif any(word in match.lower() for word in ['users', 'customers', 'downloads', 'views']):
                    achievement_categories['scale_metrics'].append(match)
                elif any(word in match.lower() for word in ['time', 'speed', 'faster', 'efficiency']):
                    achievement_categories['time_improvements'].append(match)
                elif any(word in match.lower() for word in ['accuracy', 'precision', 'quality', 'score']):
                    achievement_categories['quality_metrics'].append(match)
        
        achievement_data['quantified_achievements'] = len(all_achievements)
        achievement_data['achievement_examples'] = all_achievements[:8]  # Top 8 examples
        achievement_data['achievement_categories'] = achievement_categories
        achievement_data['achievement_diversity'] = len([cat for cat in achievement_categories.values() if cat])
        
        return achievement_data
    
    def _analyze_technical_depth(self, text):
        """Analyze technical depth and sophistication of content"""
        technical_data = {}
        
        total_technical_mentions = 0
        technical_categories = {
            'architecture_design': 0,
            'performance_optimization': 0,
            'api_integration': 0,
            'ai_ml_concepts': 0,
            'cloud_concepts': 0,
            'testing_practices': 0
        }
        
        # Categorized technical concept analysis
        concept_mappings = {
            'architecture_design': [r'(?i)(?:algorithm|data structure|system design|architecture|design pattern)'],
            'performance_optimization': [r'(?i)(?:optimization|performance|scalability|efficiency|caching)'],
            'api_integration': [r'(?i)(?:api|rest|graphql|microservice|integration|webhook)'],
            'ai_ml_concepts': [r'(?i)(?:machine learning|artificial intelligence|deep learning|neural network|model)'],
            'cloud_concepts': [r'(?i)(?:cloud|aws|azure|docker|kubernetes|containerization)'],
            'testing_practices': [r'(?i)(?:test driven|unit test|integration test|automated testing|ci/cd)']
        }
        
        for category, patterns in concept_mappings.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text))
                technical_categories[category] = matches
                total_technical_mentions += matches
        
        technical_data['technical_depth_score'] = total_technical_mentions
        technical_data['technical_categories'] = technical_categories
        technical_data['technical_sophistication'] = self._classify_technical_level(total_technical_mentions)
        
        return technical_data
    
    def _analyze_content_quality(self, text):
        """Analyze overall content quality and structure"""
        quality_data = {}
        
        # Basic statistics
        word_count = len(text.split())
        sentence_count = len(re.split(r'[.!?]+', text))
        paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
        
        quality_data['word_count'] = word_count
        quality_data['sentence_count'] = sentence_count
        quality_data['paragraph_count'] = paragraph_count
        quality_data['avg_words_per_sentence'] = word_count / sentence_count if sentence_count > 0 else 0
        
        # Action verb usage analysis
        action_verb_count = sum(1 for verb in self.action_verbs if verb.lower() in text.lower())
        quality_data['action_verb_count'] = action_verb_count
        quality_data['action_verb_density'] = action_verb_count / word_count if word_count > 0 else 0
        
        # Content structure analysis
        section_headers = len(re.findall(r'(?i)^[A-Z][A-Z\s]+$', text, re.MULTILINE))
        quality_data['section_headers'] = section_headers
        quality_data['structure_score'] = min(10, section_headers * 2)
        
        return quality_data
    
    # Helper methods
    def _validate_email(self, email):
        """Validate email format and common patterns"""
        if '@' not in email or '.' not in email:
            return False
        if email.count('@') != 1:
            return False
        return True
    
    def _clean_phone_number(self, phone):
        """Clean and format phone number"""
        if isinstance(phone, tuple):
            phone = ''.join(phone)
        # Remove all non-digits except +
        cleaned = re.sub(r'[^\d+]', '', phone)
        return cleaned if len(cleaned) >= 10 else None
    
    def _categorize_skills(self, skills_text):
        """Categorize skills into different types"""
        if not skills_text:
            return {}
        
        categories = {
            'programming_languages': [],
            'frameworks_libraries': [],
            'databases': [],
            'tools_platforms': [],
            'methodologies': []
        }
        
        # Define keyword mappings
        skill_mappings = {
            'programming_languages': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'go', 'rust', 'swift'],
            'frameworks_libraries': ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'express', 'tensorflow', 'pytorch'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'elasticsearch'],
            'tools_platforms': ['aws', 'azure', 'docker', 'kubernetes', 'jenkins', 'git', 'linux'],
            'methodologies': ['agile', 'scrum', 'devops', 'ci/cd', 'tdd', 'microservices']
        }
        
        skills_lower = skills_text.lower()
        for category, keywords in skill_mappings.items():
            for keyword in keywords:
                if keyword in skills_lower:
                    categories[category].append(keyword.title())
        
        return categories
    
    def _calculate_total_experience(self, experience_text, full_text):
        """Calculate total years of experience using multiple methods"""
        total_experience = 0
        
        # Method 1: Extract year ranges from experience section
        year_ranges = re.findall(self.patterns['year_ranges'], experience_text.lower())
        for start_year, end_year in year_ranges:
            try:
                start = int(start_year)
                end = self.current_year if end_year in ['present', 'current'] else int(end_year)
                if end >= start:  # Sanity check
                    experience_years = end - start
                    total_experience = max(total_experience, experience_years)
            except ValueError:
                continue
        
        # Method 2: Look for explicit experience mentions
        exp_mentions = re.findall(self.patterns['experience_years'], full_text)
        if exp_mentions:
            try:
                mentioned_years = int(exp_mentions[0])
                total_experience = max(total_experience, mentioned_years)
            except ValueError:
                pass
        
        # Method 3: Count overlapping employment periods
        all_years = re.findall(r'\b(20\d{2})\b', experience_text)
        if len(all_years) >= 2:
            years = [int(year) for year in all_years if 2000 <= int(year) <= self.current_year]
            if years:
                span_years = max(years) - min(years)
                total_experience = max(total_experience, span_years)
        
        return total_experience
    
    def _classify_experience_level(self, years):
        """Classify experience level based on years"""
        if years == 0:
            return "Entry Level / New Graduate"
        elif years <= 2:
            return "Junior Level"
        elif years <= 5:
            return "Mid Level"
        elif years <= 10:
            return "Senior Level"
        else:
            return "Expert / Leadership Level"
    
    def _analyze_experience_quality(self, experience_text):
        """Analyze the quality of experience descriptions"""
        if not experience_text:
            return 0
        
        quality_score = 0
        
        # Check for action verbs
        action_verb_count = sum(1 for verb in self.action_verbs if verb.lower() in experience_text.lower())
        quality_score += min(20, action_verb_count * 2)
        
        # Check for quantified achievements
        achievement_count = sum(1 for pattern in self.patterns['quantified_achievements'] 
                              for match in re.findall(pattern, experience_text, re.IGNORECASE))
        quality_score += min(30, achievement_count * 5)
        
        # Check for technical depth
        technical_mentions = sum(1 for pattern in self.technical_concepts 
                               for match in re.findall(pattern, experience_text))
        quality_score += min(25, technical_mentions * 3)
        
        # Check for leadership/collaboration indicators
        leadership_keywords = ['led', 'managed', 'coordinated', 'collaborated', 'mentored', 'trained']
        leadership_count = sum(1 for keyword in leadership_keywords if keyword in experience_text.lower())
        quality_score += min(15, leadership_count * 3)
        
        return min(100, quality_score)
    
    def _assess_project_quality(self, project_description):
        """Assess the quality of a project description"""
        if not project_description:
            return 0
        
        score = 0
        description_lower = project_description.lower()
        
        # Length and detail
        if len(project_description) > 100:
            score += 20
        elif len(project_description) > 50:
            score += 10
        
        # Technical keywords
        tech_keywords = ['built', 'developed', 'implemented', 'created', 'designed', 'using', 'with']
        tech_count = sum(1 for keyword in tech_keywords if keyword in description_lower)
        score += min(30, tech_count * 5)
        
        # Specific technologies mentioned
        if any(tech in description_lower for tech in ['python', 'javascript', 'react', 'node', 'sql', 'api']):
            score += 20
        
        # Results or impact mentioned
        if any(word in description_lower for word in ['achieved', 'improved', 'increased', 'reduced', 'optimized']):
            score += 15
        
        # URLs or links mentioned
        if any(indicator in description_lower for indicator in ['github', 'demo', 'live', 'deployed', 'hosted']):
            score += 15
        
        return min(100, score)
    
    def _count_github_references(self, text):
        """Count GitHub references in the text"""
        github_pattern = r'(?i)github'
        return len(re.findall(github_pattern, text))
    
    def _count_demo_references(self, text):
        """Count demo/live project references"""
        demo_patterns = [r'(?i)demo', r'(?i)live\s+(?:site|app|version)', r'(?i)deployed', r'(?i)hosted']
        return sum(len(re.findall(pattern, text)) for pattern in demo_patterns)
    
    def _classify_technical_level(self, technical_mentions):
        """Classify technical sophistication level"""
        if technical_mentions >= 15:
            return "Advanced Technical Depth"
        elif technical_mentions >= 8:
            return "Good Technical Understanding"
        elif technical_mentions >= 3:
            return "Basic Technical Awareness"
        else:
            return "Limited Technical Depth"
    
    def _generate_analysis_summary(self, sections):
        """Generate comprehensive analysis summary"""
        summary = {
            'total_data_points': len(sections),
            'completeness_score': 0,
            'technical_readiness': 0,
            'professional_maturity': 0,
            'key_strengths': [],
            'critical_gaps': []
        }
        
        # Calculate completeness score
        essential_sections = ['email', 'skills_text', 'experience_years', 'project_count']
        completeness = sum(1 for section in essential_sections if sections.get(section))
        summary['completeness_score'] = (completeness / len(essential_sections)) * 100
        
        # Calculate technical readiness
        technical_indicators = [
            sections.get('skills_count', 0) >= 8,
            sections.get('technical_depth_score', 0) >= 5,
            sections.get('project_count', 0) >= 2,
            sections.get('quantified_achievements', 0) >= 1
        ]
        summary['technical_readiness'] = (sum(technical_indicators) / len(technical_indicators)) * 100
        
        # Calculate professional maturity
        maturity_indicators = [
            sections.get('experience_years', 0) >= 1,
            sections.get('action_verb_count', 0) >= 5,
            sections.get('has_education', False),
            sections.get('quantified_achievements', 0) >= 2
        ]
        summary['professional_maturity'] = (sum(maturity_indicators) / len(maturity_indicators)) * 100
        
        return summary


