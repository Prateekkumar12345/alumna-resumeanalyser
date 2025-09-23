"""
Configuration module for Resume Analyzer
Contains all constants, role definitions, and industry insights
"""

from datetime import datetime

# ATS Keywords and role definitions
ATS_KEYWORDS = {
    'software_developer': {
        'core_skills': ['Python', 'Java', 'JavaScript', 'C++', 'SQL', 'Git', 'API', 'Database'],
        'frameworks': ['React', 'Node.js', 'Spring', 'Django', 'Flask', 'Express'],
        'methodologies': ['Agile', 'Scrum', 'Testing', 'DevOps', 'CI/CD'],
        'platforms': ['AWS', 'Azure', 'Docker', 'Linux', 'Cloud'],
        'daily_tasks': ['Code development', 'Bug fixing', 'Code reviews', 'System design', 'Testing'],
        'tech_stack': 'Frontend: React/Angular/Vue, Backend: Node.js/Python/Java, Database: MySQL/PostgreSQL, Cloud: AWS/Azure'
    },
    'data_scientist': {
        'core_skills': ['Python', 'R', 'SQL', 'Statistics', 'Machine Learning', 'Deep Learning'],
        'frameworks': ['Pandas', 'NumPy', 'Scikit-learn', 'TensorFlow', 'PyTorch', 'Matplotlib'],
        'methodologies': ['Data Analysis', 'Statistical Modeling', 'A/B Testing', 'ETL', 'Feature Engineering'],
        'platforms': ['Jupyter', 'AWS', 'Big Data', 'Spark', 'Hadoop'],
        'daily_tasks': ['Data cleaning', 'Model building', 'Data visualization', 'Statistical analysis', 'Reporting'],
        'tech_stack': 'Languages: Python/R, ML Libraries: Scikit-learn/TensorFlow, Visualization: Matplotlib/Seaborn/Tableau, Cloud: AWS/GCP'
    },
    'ai_engineer': {
        'core_skills': ['Machine Learning', 'Deep Learning', 'Neural Networks', 'AI', 'Python', 'Mathematics'],
        'frameworks': ['TensorFlow', 'PyTorch', 'Keras', 'OpenCV', 'Transformers', 'CUDA'],
        'methodologies': ['MLOps', 'Model Training', 'Computer Vision', 'NLP', 'Research', 'Algorithm Design'],
        'platforms': ['GPU Computing', 'Cloud ML', 'Docker', 'Kubernetes', 'MLflow'],
        'daily_tasks': ['Model development', 'Algorithm optimization', 'Research implementation', 'Performance tuning'],
        'tech_stack': 'ML: TensorFlow/PyTorch, CV: OpenCV/YOLO, NLP: Transformers/spaCy, Deployment: Docker/Kubernetes/MLflow'
    },
    'full_stack_developer': {
        'core_skills': ['JavaScript', 'TypeScript', 'HTML', 'CSS', 'SQL', 'Git', 'REST API'],
        'frameworks': ['React', 'Angular', 'Vue.js', 'Node.js', 'Express', 'MongoDB', 'PostgreSQL'],
        'methodologies': ['Responsive Design', 'Agile', 'Testing', 'GraphQL', 'Microservices'],
        'platforms': ['Docker', 'AWS', 'Heroku', 'Netlify', 'Vercel'],
        'daily_tasks': ['Frontend development', 'Backend API development', 'Database design', 'UI/UX implementation'],
        'tech_stack': 'Frontend: React/Angular + TypeScript, Backend: Node.js/Express, Database: MongoDB/PostgreSQL, Deployment: AWS/Docker'
    },
    'devops_engineer': {
        'core_skills': ['Linux', 'Bash', 'Python', 'Git', 'Networking', 'Security'],
        'frameworks': ['Docker', 'Kubernetes', 'Jenkins', 'Terraform', 'Ansible', 'Prometheus'],
        'methodologies': ['CI/CD', 'Infrastructure as Code', 'Monitoring', 'Automation', 'Containerization'],
        'platforms': ['AWS', 'Azure', 'GCP', 'Linux', 'Cloud Computing'],
        'daily_tasks': ['Infrastructure management', 'Pipeline automation', 'Monitoring setup', 'Security implementation'],
        'tech_stack': 'Cloud: AWS/Azure/GCP, Containers: Docker/Kubernetes, CI/CD: Jenkins/GitLab, IaC: Terraform/Ansible'
    }
}

# Industry insights and market data
INDUSTRY_INSIGHTS = {
    'software_developer': {
        'growth_outlook': 'High demand with 22% projected growth',
        'salary_range': '$70k-150k+ depending on experience',
        'key_companies': ['Google', 'Microsoft', 'Amazon', 'Meta', 'Netflix', 'Startups'],
        'trending_skills': ['Cloud Native', 'Microservices', 'React', 'TypeScript', 'GraphQL']
    },
    'data_scientist': {
        'growth_outlook': 'Excellent growth with increasing data-driven decisions',
        'salary_range': '$80k-180k+ based on specialization',
        'key_companies': ['Netflix', 'Uber', 'Airbnb', 'LinkedIn', 'Banking sector'],
        'trending_skills': ['MLOps', 'Deep Learning', 'Big Data', 'Cloud ML', 'AutoML']
    },
    'ai_engineer': {
        'growth_outlook': 'Explosive growth with AI revolution',
        'salary_range': '$90k-200k+ for specialized roles',
        'key_companies': ['OpenAI', 'Google DeepMind', 'Tesla', 'NVIDIA', 'Research labs'],
        'trending_skills': ['Transformers', 'LLMs', 'Computer Vision', 'Reinforcement Learning', 'MLOps']
    }
}

# Scoring configuration (removed social media components)
SCORING_CONFIG = {
    'contact_info': {'max': 15, 'weight': 0.15},
    'technical_skills': {'max': 30, 'weight': 0.30},
    'experience_quality': {'max': 25, 'weight': 0.25},
    'quantified_achievements': {'max': 20, 'weight': 0.20},
    'content_optimization': {'max': 10, 'weight': 0.10}
}

# Current year for experience calculations
CURRENT_YEAR = datetime.now().year

# Regular expression patterns
REGEX_PATTERNS = {
    'email': r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
    'phone': r'(\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}|\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})',
    'year_ranges': r'(\d{4})\s*[-–—]\s*(\d{4}|present|current)',
    'experience_years': r'(?i)(\d{1,2})\s+years?\s+(?:of\s+)?experience',
    'quantified_achievements': [
        r'\d+%\s+(?:increase|improvement|growth|reduction)',
        r'(?:increased|improved|reduced|optimized)[^.\n]*\d+%',
        r'\$\d+(?:k|,\d{3})*(?:\s+(?:saved|revenue|profit))?',
        r'\d+(?:k|,\d{3})*\+?\s+(?:users|customers|downloads|views)',
        r'\d+(?:st|nd|rd|th)\s+(?:place|position|rank)',
        r'(?:accuracy|precision|recall)\s+of\s+\d+%'
    ]
}

# Action verbs for content analysis
ACTION_VERBS = [
    'developed', 'built', 'created', 'implemented', 'designed', 'led', 'managed', 
    'optimized', 'engineered', 'architected', 'delivered', 'achieved', 'improved',
    'streamlined', 'automated', 'collaborated', 'spearheaded', 'coordinated'
]

# Advanced technical concepts for depth analysis
TECHNICAL_CONCEPTS = [
    r'(?i)(?:algorithm|data structure|system design|architecture)',
    r'(?i)(?:optimization|performance|scalability|efficiency)',
    r'(?i)(?:api|microservice|database design|security)',
    r'(?i)(?:machine learning|artificial intelligence|deep learning)',
    r'(?i)(?:cloud native|containerization|orchestration)',
    r'(?i)(?:test driven|continuous integration|deployment pipeline)'
]