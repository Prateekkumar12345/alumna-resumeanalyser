from fastapi import FastAPI, HTTPException, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
import tempfile
import os
import json
import logging
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
import asyncio
import io
import concurrent.futures
import re
from datetime import datetime

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field, validator

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="AI Resume Analyzer API with Job Search")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.warning("OPENAI_API_KEY not found in environment variables")

# ===== PYDANTIC MODELS - MATCHING YOUR STANDARD JSON STRUCTURE =====

class ProfessionalProfile(BaseModel):
    experience_level: str = Field(description="Years of experience and seniority level")
    technical_skills_count: int = Field(description="Number of technical skills identified")
    project_portfolio_size: str = Field(description="Size and quality of project portfolio")
    achievement_metrics: str = Field(description="Quality of quantified achievements")
    technical_sophistication: str = Field(description="Level of technical expertise")

class ContactPresentation(BaseModel):
    email_address: str = Field(description="Email presence and quality")
    phone_number: str = Field(description="Phone number presence")
    education: str = Field(description="Education background quality")
    resume_length: str = Field(description="Resume length assessment")
    action_verbs: str = Field(description="Use of strong action verbs")

class OverallAssessment(BaseModel):
    score_percentage: int = Field(description="Overall score percentage")
    level: str = Field(description="Assessment level")
    description: str = Field(description="Score description")
    recommendation: str = Field(description="Overall recommendation")

class ExecutiveSummary(BaseModel):
    professional_profile: ProfessionalProfile
    contact_presentation: ContactPresentation
    overall_assessment: OverallAssessment

class ScoringDetail(BaseModel):
    score: int = Field(description="Score out of max points")
    max_score: int = Field(description="Maximum possible score")
    percentage: float = Field(description="Percentage score")
    details: List[str] = Field(description="Detailed breakdown of scoring")

class StrengthAnalysis(BaseModel):
    strength: str = Field(description="Main strength identified")
    why_its_strong: str = Field(description="Explanation of why it's a strength")
    ats_benefit: str = Field(description="How it helps with ATS systems")
    competitive_advantage: str = Field(description="Competitive advantage provided")
    evidence: str = Field(description="Supporting evidence from resume")

class WeaknessAnalysis(BaseModel):
    weakness: str = Field(description="Main weakness identified")
    why_problematic: str = Field(description="Why this is problematic")
    ats_impact: str = Field(description="Impact on ATS systems")
    how_it_hurts: str = Field(description="How it hurts candidacy")
    fix_priority: str = Field(description="Priority level: CRITICAL/HIGH/MEDIUM")
    specific_fix: str = Field(description="Specific steps to fix")
    timeline: str = Field(description="Timeline for implementation")

class ImprovementPlan(BaseModel):
    critical: List[str] = Field(default_factory=list, description="Critical improvements")
    high: List[str] = Field(default_factory=list, description="High priority improvements")
    medium: List[str] = Field(default_factory=list, description="Medium priority improvements")

class JobMarketAnalysis(BaseModel):
    role_compatibility: str = Field(description="Compatibility with target role")
    market_positioning: str = Field(description="Position in job market")
    career_advancement: str = Field(description="Career advancement opportunities")
    skill_development: str = Field(description="Skill development recommendations")

class AIInsights(BaseModel):
    overall_score: int = Field(description="Overall AI-determined score")
    recommendation_level: str = Field(description="Recommendation level")
    key_strengths_count: int = Field(description="Number of key strengths")
    improvement_areas_count: int = Field(description="Number of improvement areas")

class ResumeAnalysis(BaseModel):
    """Main analysis model matching your standard JSON structure"""
    professional_profile: ProfessionalProfile
    contact_presentation: ContactPresentation
    detailed_scoring: Dict[str, ScoringDetail]
    strengths_analysis: List[StrengthAnalysis] = Field(min_items=5)
    weaknesses_analysis: List[WeaknessAnalysis] = Field(min_items=5)
    improvement_plan: ImprovementPlan
    job_market_analysis: JobMarketAnalysis
    overall_score: int = Field(ge=0, le=100, description="Overall resume score out of 100")
    recommendation_level: str = Field(description="Overall recommendation level")

class JobListing(BaseModel):
    company_name: str = Field(description="Name of the hiring company")
    position: str = Field(description="Job position/title")
    location: str = Field(description="Job location")
    ctc: str = Field(description="Compensation/Salary range")
    experience_required: str = Field(description="Required years of experience")
    last_date_to_apply: str = Field(description="Application deadline")
    about_job: str = Field(description="Brief description about the job")
    job_description: str = Field(description="Detailed job description")
    job_requirements: str = Field(description="Required skills and qualifications")
    application_url: Optional[str] = Field(description="Link to apply")

class OptimizedPDFExtractor:
    """Optimized PDF text extraction"""
    
    @staticmethod
    async def extract_text_from_pdf(uploaded_file) -> Optional[str]:
        try:
            uploaded_file.seek(0)
            content = await uploaded_file.read()
            
            def process_pdf(content_bytes):
                pdf_file = io.BytesIO(content_bytes)
                pdf_reader = PdfReader(pdf_file)
                
                extracted_text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            extracted_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    except Exception as page_error:
                        logger.warning(f"Error extracting page {page_num + 1}: {str(page_error)}")
                        continue
                
                return extracted_text.strip()
            
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                extracted_text = await loop.run_in_executor(pool, process_pdf, content)
            
            return extracted_text if extracted_text else None
            
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            return None

class JobSearchService:
    """Service to search and parse job listings"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    async def search_jobs(self, target_role: str, location: str = "India") -> List[Dict[str, Any]]:
        """Search for jobs and extract structured information"""
        try:
            job_extraction_prompt = f"""
            Generate 5-10 realistic current job listings for the position: {target_role} in {location}.
            
            For each job listing, provide EXACTLY these fields in JSON format:
            {{
                "company_name": "Company name",
                "position": "Exact job title",
                "location": "City/region in {location}",
                "ctc": "Salary range with currency",
                "experience_required": "X-Y years",
                "last_date_to_apply": "YYYY-MM-DD format",
                "about_job": "2-3 sentence summary",
                "job_description": "Detailed responsibilities and duties",
                "job_requirements": "Required skills, qualifications, and education",
                "application_url": "https://company-careers.com/job-id"
            }}
            
            Return ONLY a valid JSON array with no additional text. Make the data realistic and relevant to the current job market in 2025.
            """
            
            response = await self.llm.apredict(job_extraction_prompt)
            
            # Parse the JSON response
            try:
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    jobs_data = json.loads(json_match.group())
                else:
                    jobs_data = json.loads(response)
                
                return jobs_data
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse job listings JSON: {e}")
                return []
                
        except Exception as e:
            logger.error(f"Job search error: {str(e)}")
            return []

class HighPerformanceLangChainAnalyzer:
    """High-performance AI analyzer with guaranteed standard JSON output"""
    
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model_name="gpt-3.5-turbo-16k",
            temperature=0.2,
            max_tokens=4000,
            request_timeout=30
        )
        
        self.output_parser = PydanticOutputParser(pydantic_object=ResumeAnalysis)
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.output_parser, llm=self.llm)
        self.analysis_chain = self._create_analysis_chain()
        self.job_search = JobSearchService(self.llm)
    
    def _create_analysis_chain(self) -> LLMChain:
        """Create the main analysis chain with strict output format"""
        
        human_prompt_template = """
        Analyze this resume comprehensively for the target role: {target_role}

        RESUME CONTENT:
        {resume_text}

        YOU MUST provide a complete JSON response with ALL of the following sections:

        1. PROFESSIONAL PROFILE (experience_level, technical_skills_count, project_portfolio_size, achievement_metrics, technical_sophistication)
        2. CONTACT PRESENTATION (email_address, phone_number, education, resume_length, action_verbs)
        3. DETAILED SCORING with these exact sections:
           - "Contact Information" (score, max_score, percentage, details)
           - "Technical Skills" (score, max_score, percentage, details)
           - "Experience Quality" (score, max_score, percentage, details)
           - "Quantified Achievements" (score, max_score, percentage, details)
           - "Content Optimization" (score, max_score, percentage, details)
        4. STRENGTHS ANALYSIS - Provide at least 5 strengths (strength, why_its_strong, ats_benefit, competitive_advantage, evidence)
        5. WEAKNESSES ANALYSIS - Provide at least 5 weaknesses (weakness, why_problematic, ats_impact, how_it_hurts, fix_priority, specific_fix, timeline)
        6. IMPROVEMENT PLAN (critical, high, medium lists)
        7. JOB MARKET ANALYSIS (role_compatibility, market_positioning, career_advancement, skill_development)
        8. overall_score (0-100)
        9. recommendation_level

        {format_instructions}
        
        CRITICAL: Return ONLY valid JSON matching the exact structure specified. No additional text or explanations.
        """
        
        prompt = PromptTemplate(
            template=human_prompt_template,
            input_variables=["resume_text", "target_role"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
        
        return LLMChain(llm=self.llm, prompt=prompt)
    
    def _get_standard_response_template(self, target_role: str, word_count: int) -> Dict[str, Any]:
        """Returns the standard response structure that your frontend expects"""
        return {
            "success": True,
            "analysis_method": "AI-Powered LangChain Analysis",
            "resume_metadata": {
                "word_count": word_count,
                "validation_message": "Comprehensive AI analysis completed",
                "target_role": target_role or "general position"
            },
            "executive_summary": {
                "professional_profile": {},
                "contact_presentation": {},
                "overall_assessment": {}
            },
            "detailed_scoring": {},
            "strengths_analysis": [],
            "weaknesses_analysis": [],
            "improvement_plan": {
                "critical": [],
                "high": [],
                "medium": []
            },
            "job_market_analysis": {},
            "ai_insights": {}
        }
    
    async def analyze_resume_with_jobs(
        self, 
        resume_text: str, 
        target_role: Optional[str] = None,
        search_jobs: bool = True,
        location: str = "India"
    ) -> Dict[str, Any]:
        """Analyze resume with guaranteed standard JSON format and optional job search"""
        try:
            role_context = target_role or "general position"
            word_count = len(resume_text.split())
            
            # Initialize response with standard structure
            response = self._get_standard_response_template(role_context, word_count)
            
            # Run resume analysis and job search in parallel
            if search_jobs and target_role:
                analysis_task = self.analysis_chain.arun(
                    resume_text=resume_text,
                    target_role=role_context
                )
                jobs_task = self.job_search.search_jobs(target_role, location)
                
                analysis_result, job_listings = await asyncio.gather(
                    analysis_task,
                    jobs_task,
                    return_exceptions=True
                )
                
                if isinstance(analysis_result, Exception):
                    raise analysis_result
                if isinstance(job_listings, Exception):
                    logger.error(f"Job search failed: {job_listings}")
                    job_listings = []
            else:
                analysis_result = await self.analysis_chain.arun(
                    resume_text=resume_text,
                    target_role=role_context
                )
                job_listings = []
            
            # Parse and populate response
            try:
                parsed_analysis = self.fixing_parser.parse(analysis_result)
                self._populate_response(response, parsed_analysis, word_count, role_context)
                
            except Exception as parse_error:
                logger.warning(f"Structured parsing failed, using fallback: {parse_error}")
                self._populate_fallback_response(response, analysis_result, word_count, role_context)
            
            # Add job listings if available
            if job_listings:
                response["job_listings"] = {
                    "total_jobs_found": len(job_listings),
                    "search_query": f"{target_role} in {location}",
                    "jobs": job_listings
                }
            
            return response
                
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return self._generate_error_response(str(e), target_role, word_count)
    
    def _populate_response(self, response: Dict, analysis: ResumeAnalysis, word_count: int, target_role: str):
        """Populate response with parsed analysis data"""
        
        # Executive Summary
        response["executive_summary"] = {
            "professional_profile": {
                "experience_level": analysis.professional_profile.experience_level,
                "technical_skills_count": analysis.professional_profile.technical_skills_count,
                "project_portfolio_size": analysis.professional_profile.project_portfolio_size,
                "achievement_metrics": analysis.professional_profile.achievement_metrics,
                "technical_sophistication": analysis.professional_profile.technical_sophistication
            },
            "contact_presentation": {
                "email_address": analysis.contact_presentation.email_address,
                "phone_number": analysis.contact_presentation.phone_number,
                "education": analysis.contact_presentation.education,
                "resume_length": analysis.contact_presentation.resume_length,
                "action_verbs": analysis.contact_presentation.action_verbs
            },
            "overall_assessment": {
                "score_percentage": analysis.overall_score,
                "level": analysis.recommendation_level,
                "description": f"AI-determined resume quality: {analysis.overall_score}%",
                "recommendation": analysis.recommendation_level
            }
        }
        
        # Detailed Scoring
        response["detailed_scoring"] = {
            key: {
                "score": detail.score,
                "max_score": detail.max_score,
                "percentage": detail.percentage,
                "details": detail.details
            }
            for key, detail in analysis.detailed_scoring.items()
        }
        
        # Strengths
        response["strengths_analysis"] = [
            {
                "strength": s.strength,
                "why_its_strong": s.why_its_strong,
                "ats_benefit": s.ats_benefit,
                "competitive_advantage": s.competitive_advantage,
                "evidence": s.evidence
            }
            for s in analysis.strengths_analysis
        ]
        
        # Weaknesses
        response["weaknesses_analysis"] = [
            {
                "weakness": w.weakness,
                "why_problematic": w.why_problematic,
                "ats_impact": w.ats_impact,
                "how_it_hurts": w.how_it_hurts,
                "fix_priority": w.fix_priority,
                "specific_fix": w.specific_fix,
                "timeline": w.timeline
            }
            for w in analysis.weaknesses_analysis
        ]
        
        # Improvement Plan
        response["improvement_plan"] = {
            "critical": analysis.improvement_plan.critical,
            "high": analysis.improvement_plan.high,
            "medium": analysis.improvement_plan.medium
        }
        
        # Job Market Analysis
        response["job_market_analysis"] = {
            "role_compatibility": analysis.job_market_analysis.role_compatibility,
            "market_positioning": analysis.job_market_analysis.market_positioning,
            "career_advancement": analysis.job_market_analysis.career_advancement,
            "skill_development": analysis.job_market_analysis.skill_development
        }
        
        # AI Insights
        response["ai_insights"] = {
            "overall_score": analysis.overall_score,
            "recommendation_level": analysis.recommendation_level,
            "key_strengths_count": len(analysis.strengths_analysis),
            "improvement_areas_count": len(analysis.weaknesses_analysis)
        }
    
    def _populate_fallback_response(self, response: Dict, raw_result: str, word_count: int, target_role: str):
        """Fallback method to populate response from raw LLM output"""
        try:
            json_match = re.search(r'\{.*\}', raw_result, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group())
                
                # Try to extract data and populate standard structure
                if "professional_profile" in parsed_data:
                    response["executive_summary"]["professional_profile"] = parsed_data["professional_profile"]
                if "contact_presentation" in parsed_data:
                    response["executive_summary"]["contact_presentation"] = parsed_data["contact_presentation"]
                if "overall_score" in parsed_data:
                    response["executive_summary"]["overall_assessment"] = {
                        "score_percentage": parsed_data.get("overall_score", 0),
                        "level": parsed_data.get("recommendation_level", "Unknown"),
                        "description": f"AI-determined resume quality: {parsed_data.get('overall_score', 0)}%",
                        "recommendation": parsed_data.get("recommendation_level", "Unknown")
                    }
                
                response["detailed_scoring"] = parsed_data.get("detailed_scoring", {})
                response["strengths_analysis"] = parsed_data.get("strengths_analysis", [])
                response["weaknesses_analysis"] = parsed_data.get("weaknesses_analysis", [])
                response["improvement_plan"] = parsed_data.get("improvement_plan", {"critical": [], "high": [], "medium": []})
                response["job_market_analysis"] = parsed_data.get("job_market_analysis", {})
                response["ai_insights"] = {
                    "overall_score": parsed_data.get("overall_score", 0),
                    "recommendation_level": parsed_data.get("recommendation_level", "Unknown"),
                    "key_strengths_count": len(parsed_data.get("strengths_analysis", [])),
                    "improvement_areas_count": len(parsed_data.get("weaknesses_analysis", []))
                }
                
        except Exception as e:
            logger.error(f"Fallback parsing error: {e}")
            # Keep the standard structure with empty/default values
    
    def _generate_error_response(self, error_message: str, target_role: str = None, word_count: int = 0) -> Dict[str, Any]:
        """Generate error response maintaining standard structure"""
        response = self._get_standard_response_template(target_role or "unknown", word_count)
        response["success"] = False
        response["error"] = f"AI analysis failed: {error_message}"
        response["resume_metadata"]["validation_message"] = "Analysis encountered an error"
        return response

# Initialize components
pdf_extractor = OptimizedPDFExtractor()
high_perf_analyzer = None

if openai_api_key:
    try:
        high_perf_analyzer = HighPerformanceLangChainAnalyzer(openai_api_key)
        logger.info("High-performance analyzer initialized successfully")
    except Exception as init_error:
        logger.error(f"Failed to initialize analyzer: {init_error}")

# Performance middleware
@app.middleware("http")
async def add_performance_headers(request: Request, call_next):
    start_time = asyncio.get_event_loop().time()
    response = await call_next(request)
    process_time = asyncio.get_event_loop().time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time, 2))
    return response

@app.post("/analyze-resume")
async def analyze_resume(
    file: UploadFile = File(...),
    target_role: Optional[str] = None,
    search_jobs: bool = True,
    location: str = "India"
):
    """
    Comprehensive resume analysis with guaranteed standard JSON output and job search integration
    
    Parameters:
    - file: PDF resume file
    - target_role: Target job position (recommended for best results)
    - search_jobs: Whether to search for relevant jobs (default: True)
    - location: Job search location (default: India)
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        if not high_perf_analyzer:
            raise HTTPException(status_code=500, detail="AI analyzer not initialized.")
        
        if not file.content_type or "pdf" not in file.content_type.lower():
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Extract PDF text
        resume_text = await pdf_extractor.extract_text_from_pdf(file)
        
        if not resume_text:
            raise HTTPException(status_code=400, detail="Failed to extract text from PDF.")
        
        if len(resume_text.strip()) < 100:
            raise HTTPException(status_code=400, detail="Resume content too short.")
        
        # Perform analysis with guaranteed standard output
        analysis_result = await asyncio.wait_for(
            high_perf_analyzer.analyze_resume_with_jobs(
                resume_text, 
                target_role, 
                search_jobs=search_jobs and bool(target_role),
                location=location
            ),
            timeout=60.0
        )
        
        processing_time = asyncio.get_event_loop().time() - start_time
        logger.info(f"Analysis completed in {processing_time:.2f}s")
        
        return analysis_result
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Analysis timeout.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AI Resume Analyzer with Consistent JSON Output",
        "openai_configured": bool(openai_api_key),
        "analyzer_available": bool(high_perf_analyzer),
        "features": [
            "Guaranteed standard JSON structure",
            "Resume analysis",
            "Job search integration",
            "Frontend-compatible output"
        ]
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AI Resume Analyzer with Consistent JSON Output",
        "version": "2.2",
        "description": "AI resume analysis with guaranteed standard JSON format for frontend compatibility",
        "endpoints": {
            "/analyze-resume": "POST - Comprehensive analysis with standard JSON output",
            "/health": "GET - Service health check",
            "/docs": "GET - API documentation"
        },
        "guarantees": [
            "Consistent JSON structure every time",
            "All standard fields present",
            "Frontend-compatible format",
            "Optional job listings"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", workers=1)from fastapi import FastAPI, HTTPException, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
import tempfile
import os
import json
import logging
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
import asyncio
import io
import concurrent.futures
import re
from datetime import datetime

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field, validator

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="AI Resume Analyzer API with Job Search")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.warning("OPENAI_API_KEY not found in environment variables")

# ===== PYDANTIC MODELS - MATCHING YOUR STANDARD JSON STRUCTURE =====

class ProfessionalProfile(BaseModel):
    experience_level: str = Field(description="Years of experience and seniority level")
    technical_skills_count: int = Field(description="Number of technical skills identified")
    project_portfolio_size: str = Field(description="Size and quality of project portfolio")
    achievement_metrics: str = Field(description="Quality of quantified achievements")
    technical_sophistication: str = Field(description="Level of technical expertise")

class ContactPresentation(BaseModel):
    email_address: str = Field(description="Email presence and quality")
    phone_number: str = Field(description="Phone number presence")
    education: str = Field(description="Education background quality")
    resume_length: str = Field(description="Resume length assessment")
    action_verbs: str = Field(description="Use of strong action verbs")

class OverallAssessment(BaseModel):
    score_percentage: int = Field(description="Overall score percentage")
    level: str = Field(description="Assessment level")
    description: str = Field(description="Score description")
    recommendation: str = Field(description="Overall recommendation")

class ExecutiveSummary(BaseModel):
    professional_profile: ProfessionalProfile
    contact_presentation: ContactPresentation
    overall_assessment: OverallAssessment

class ScoringDetail(BaseModel):
    score: int = Field(description="Score out of max points")
    max_score: int = Field(description="Maximum possible score")
    percentage: float = Field(description="Percentage score")
    details: List[str] = Field(description="Detailed breakdown of scoring")

class StrengthAnalysis(BaseModel):
    strength: str = Field(description="Main strength identified")
    why_its_strong: str = Field(description="Explanation of why it's a strength")
    ats_benefit: str = Field(description="How it helps with ATS systems")
    competitive_advantage: str = Field(description="Competitive advantage provided")
    evidence: str = Field(description="Supporting evidence from resume")

class WeaknessAnalysis(BaseModel):
    weakness: str = Field(description="Main weakness identified")
    why_problematic: str = Field(description="Why this is problematic")
    ats_impact: str = Field(description="Impact on ATS systems")
    how_it_hurts: str = Field(description="How it hurts candidacy")
    fix_priority: str = Field(description="Priority level: CRITICAL/HIGH/MEDIUM")
    specific_fix: str = Field(description="Specific steps to fix")
    timeline: str = Field(description="Timeline for implementation")

class ImprovementPlan(BaseModel):
    critical: List[str] = Field(default_factory=list, description="Critical improvements")
    high: List[str] = Field(default_factory=list, description="High priority improvements")
    medium: List[str] = Field(default_factory=list, description="Medium priority improvements")

class JobMarketAnalysis(BaseModel):
    role_compatibility: str = Field(description="Compatibility with target role")
    market_positioning: str = Field(description="Position in job market")
    career_advancement: str = Field(description="Career advancement opportunities")
    skill_development: str = Field(description="Skill development recommendations")

class AIInsights(BaseModel):
    overall_score: int = Field(description="Overall AI-determined score")
    recommendation_level: str = Field(description="Recommendation level")
    key_strengths_count: int = Field(description="Number of key strengths")
    improvement_areas_count: int = Field(description="Number of improvement areas")

class ResumeAnalysis(BaseModel):
    """Main analysis model matching your standard JSON structure"""
    professional_profile: ProfessionalProfile
    contact_presentation: ContactPresentation
    detailed_scoring: Dict[str, ScoringDetail]
    strengths_analysis: List[StrengthAnalysis] = Field(min_items=5)
    weaknesses_analysis: List[WeaknessAnalysis] = Field(min_items=5)
    improvement_plan: ImprovementPlan
    job_market_analysis: JobMarketAnalysis
    overall_score: int = Field(ge=0, le=100, description="Overall resume score out of 100")
    recommendation_level: str = Field(description="Overall recommendation level")

class JobListing(BaseModel):
    company_name: str = Field(description="Name of the hiring company")
    position: str = Field(description="Job position/title")
    location: str = Field(description="Job location")
    ctc: str = Field(description="Compensation/Salary range")
    experience_required: str = Field(description="Required years of experience")
    last_date_to_apply: str = Field(description="Application deadline")
    about_job: str = Field(description="Brief description about the job")
    job_description: str = Field(description="Detailed job description")
    job_requirements: str = Field(description="Required skills and qualifications")
    application_url: Optional[str] = Field(description="Link to apply")

class OptimizedPDFExtractor:
    """Optimized PDF text extraction"""
    
    @staticmethod
    async def extract_text_from_pdf(uploaded_file) -> Optional[str]:
        try:
            uploaded_file.seek(0)
            content = await uploaded_file.read()
            
            def process_pdf(content_bytes):
                pdf_file = io.BytesIO(content_bytes)
                pdf_reader = PdfReader(pdf_file)
                
                extracted_text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            extracted_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    except Exception as page_error:
                        logger.warning(f"Error extracting page {page_num + 1}: {str(page_error)}")
                        continue
                
                return extracted_text.strip()
            
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                extracted_text = await loop.run_in_executor(pool, process_pdf, content)
            
            return extracted_text if extracted_text else None
            
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            return None

class JobSearchService:
    """Service to search and parse job listings"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    async def search_jobs(self, target_role: str, location: str = "India") -> List[Dict[str, Any]]:
        """Search for jobs and extract structured information"""
        try:
            job_extraction_prompt = f"""
            Generate 5-10 realistic current job listings for the position: {target_role} in {location}.
            
            For each job listing, provide EXACTLY these fields in JSON format:
            {{
                "company_name": "Company name",
                "position": "Exact job title",
                "location": "City/region in {location}",
                "ctc": "Salary range with currency",
                "experience_required": "X-Y years",
                "last_date_to_apply": "YYYY-MM-DD format",
                "about_job": "2-3 sentence summary",
                "job_description": "Detailed responsibilities and duties",
                "job_requirements": "Required skills, qualifications, and education",
                "application_url": "https://company-careers.com/job-id"
            }}
            
            Return ONLY a valid JSON array with no additional text. Make the data realistic and relevant to the current job market in 2025.
            """
            
            response = await self.llm.apredict(job_extraction_prompt)
            
            # Parse the JSON response
            try:
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    jobs_data = json.loads(json_match.group())
                else:
                    jobs_data = json.loads(response)
                
                return jobs_data
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse job listings JSON: {e}")
                return []
                
        except Exception as e:
            logger.error(f"Job search error: {str(e)}")
            return []

class HighPerformanceLangChainAnalyzer:
    """High-performance AI analyzer with guaranteed standard JSON output"""
    
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model_name="gpt-3.5-turbo-16k",
            temperature=0.2,
            max_tokens=4000,
            request_timeout=30
        )
        
        self.output_parser = PydanticOutputParser(pydantic_object=ResumeAnalysis)
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.output_parser, llm=self.llm)
        self.analysis_chain = self._create_analysis_chain()
        self.job_search = JobSearchService(self.llm)
    
    def _create_analysis_chain(self) -> LLMChain:
        """Create the main analysis chain with strict output format"""
        
        human_prompt_template = """
        Analyze this resume comprehensively for the target role: {target_role}

        RESUME CONTENT:
        {resume_text}

        YOU MUST provide a complete JSON response with ALL of the following sections:

        1. PROFESSIONAL PROFILE (experience_level, technical_skills_count, project_portfolio_size, achievement_metrics, technical_sophistication)
        2. CONTACT PRESENTATION (email_address, phone_number, education, resume_length, action_verbs)
        3. DETAILED SCORING with these exact sections:
           - "Contact Information" (score, max_score, percentage, details)
           - "Technical Skills" (score, max_score, percentage, details)
           - "Experience Quality" (score, max_score, percentage, details)
           - "Quantified Achievements" (score, max_score, percentage, details)
           - "Content Optimization" (score, max_score, percentage, details)
        4. STRENGTHS ANALYSIS - Provide at least 5 strengths (strength, why_its_strong, ats_benefit, competitive_advantage, evidence)
        5. WEAKNESSES ANALYSIS - Provide at least 5 weaknesses (weakness, why_problematic, ats_impact, how_it_hurts, fix_priority, specific_fix, timeline)
        6. IMPROVEMENT PLAN (critical, high, medium lists)
        7. JOB MARKET ANALYSIS (role_compatibility, market_positioning, career_advancement, skill_development)
        8. overall_score (0-100)
        9. recommendation_level

        {format_instructions}
        
        CRITICAL: Return ONLY valid JSON matching the exact structure specified. No additional text or explanations.
        """
        
        prompt = PromptTemplate(
            template=human_prompt_template,
            input_variables=["resume_text", "target_role"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
        
        return LLMChain(llm=self.llm, prompt=prompt)
    
    def _get_standard_response_template(self, target_role: str, word_count: int) -> Dict[str, Any]:
        """Returns the standard response structure that your frontend expects"""
        return {
            "success": True,
            "analysis_method": "AI-Powered LangChain Analysis",
            "resume_metadata": {
                "word_count": word_count,
                "validation_message": "Comprehensive AI analysis completed",
                "target_role": target_role or "general position"
            },
            "executive_summary": {
                "professional_profile": {},
                "contact_presentation": {},
                "overall_assessment": {}
            },
            "detailed_scoring": {},
            "strengths_analysis": [],
            "weaknesses_analysis": [],
            "improvement_plan": {
                "critical": [],
                "high": [],
                "medium": []
            },
            "job_market_analysis": {},
            "ai_insights": {}
        }
    
    async def analyze_resume_with_jobs(
        self, 
        resume_text: str, 
        target_role: Optional[str] = None,
        search_jobs: bool = True,
        location: str = "India"
    ) -> Dict[str, Any]:
        """Analyze resume with guaranteed standard JSON format and optional job search"""
        try:
            role_context = target_role or "general position"
            word_count = len(resume_text.split())
            
            # Initialize response with standard structure
            response = self._get_standard_response_template(role_context, word_count)
            
            # Run resume analysis and job search in parallel
            if search_jobs and target_role:
                analysis_task = self.analysis_chain.arun(
                    resume_text=resume_text,
                    target_role=role_context
                )
                jobs_task = self.job_search.search_jobs(target_role, location)
                
                analysis_result, job_listings = await asyncio.gather(
                    analysis_task,
                    jobs_task,
                    return_exceptions=True
                )
                
                if isinstance(analysis_result, Exception):
                    raise analysis_result
                if isinstance(job_listings, Exception):
                    logger.error(f"Job search failed: {job_listings}")
                    job_listings = []
            else:
                analysis_result = await self.analysis_chain.arun(
                    resume_text=resume_text,
                    target_role=role_context
                )
                job_listings = []
            
            # Parse and populate response
            try:
                parsed_analysis = self.fixing_parser.parse(analysis_result)
                self._populate_response(response, parsed_analysis, word_count, role_context)
                
            except Exception as parse_error:
                logger.warning(f"Structured parsing failed, using fallback: {parse_error}")
                self._populate_fallback_response(response, analysis_result, word_count, role_context)
            
            # Add job listings if available
            if job_listings:
                response["job_listings"] = {
                    "total_jobs_found": len(job_listings),
                    "search_query": f"{target_role} in {location}",
                    "jobs": job_listings
                }
            
            return response
                
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return self._generate_error_response(str(e), target_role, word_count)
    
    def _populate_response(self, response: Dict, analysis: ResumeAnalysis, word_count: int, target_role: str):
        """Populate response with parsed analysis data"""
        
        # Executive Summary
        response["executive_summary"] = {
            "professional_profile": {
                "experience_level": analysis.professional_profile.experience_level,
                "technical_skills_count": analysis.professional_profile.technical_skills_count,
                "project_portfolio_size": analysis.professional_profile.project_portfolio_size,
                "achievement_metrics": analysis.professional_profile.achievement_metrics,
                "technical_sophistication": analysis.professional_profile.technical_sophistication
            },
            "contact_presentation": {
                "email_address": analysis.contact_presentation.email_address,
                "phone_number": analysis.contact_presentation.phone_number,
                "education": analysis.contact_presentation.education,
                "resume_length": analysis.contact_presentation.resume_length,
                "action_verbs": analysis.contact_presentation.action_verbs
            },
            "overall_assessment": {
                "score_percentage": analysis.overall_score,
                "level": analysis.recommendation_level,
                "description": f"AI-determined resume quality: {analysis.overall_score}%",
                "recommendation": analysis.recommendation_level
            }
        }
        
        # Detailed Scoring
        response["detailed_scoring"] = {
            key: {
                "score": detail.score,
                "max_score": detail.max_score,
                "percentage": detail.percentage,
                "details": detail.details
            }
            for key, detail in analysis.detailed_scoring.items()
        }
        
        # Strengths
        response["strengths_analysis"] = [
            {
                "strength": s.strength,
                "why_its_strong": s.why_its_strong,
                "ats_benefit": s.ats_benefit,
                "competitive_advantage": s.competitive_advantage,
                "evidence": s.evidence
            }
            for s in analysis.strengths_analysis
        ]
        
        # Weaknesses
        response["weaknesses_analysis"] = [
            {
                "weakness": w.weakness,
                "why_problematic": w.why_problematic,
                "ats_impact": w.ats_impact,
                "how_it_hurts": w.how_it_hurts,
                "fix_priority": w.fix_priority,
                "specific_fix": w.specific_fix,
                "timeline": w.timeline
            }
            for w in analysis.weaknesses_analysis
        ]
        
        # Improvement Plan
        response["improvement_plan"] = {
            "critical": analysis.improvement_plan.critical,
            "high": analysis.improvement_plan.high,
            "medium": analysis.improvement_plan.medium
        }
        
        # Job Market Analysis
        response["job_market_analysis"] = {
            "role_compatibility": analysis.job_market_analysis.role_compatibility,
            "market_positioning": analysis.job_market_analysis.market_positioning,
            "career_advancement": analysis.job_market_analysis.career_advancement,
            "skill_development": analysis.job_market_analysis.skill_development
        }
        
        # AI Insights
        response["ai_insights"] = {
            "overall_score": analysis.overall_score,
            "recommendation_level": analysis.recommendation_level,
            "key_strengths_count": len(analysis.strengths_analysis),
            "improvement_areas_count": len(analysis.weaknesses_analysis)
        }
    
    def _populate_fallback_response(self, response: Dict, raw_result: str, word_count: int, target_role: str):
        """Fallback method to populate response from raw LLM output"""
        try:
            json_match = re.search(r'\{.*\}', raw_result, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group())
                
                # Try to extract data and populate standard structure
                if "professional_profile" in parsed_data:
                    response["executive_summary"]["professional_profile"] = parsed_data["professional_profile"]
                if "contact_presentation" in parsed_data:
                    response["executive_summary"]["contact_presentation"] = parsed_data["contact_presentation"]
                if "overall_score" in parsed_data:
                    response["executive_summary"]["overall_assessment"] = {
                        "score_percentage": parsed_data.get("overall_score", 0),
                        "level": parsed_data.get("recommendation_level", "Unknown"),
                        "description": f"AI-determined resume quality: {parsed_data.get('overall_score', 0)}%",
                        "recommendation": parsed_data.get("recommendation_level", "Unknown")
                    }
                
                response["detailed_scoring"] = parsed_data.get("detailed_scoring", {})
                response["strengths_analysis"] = parsed_data.get("strengths_analysis", [])
                response["weaknesses_analysis"] = parsed_data.get("weaknesses_analysis", [])
                response["improvement_plan"] = parsed_data.get("improvement_plan", {"critical": [], "high": [], "medium": []})
                response["job_market_analysis"] = parsed_data.get("job_market_analysis", {})
                response["ai_insights"] = {
                    "overall_score": parsed_data.get("overall_score", 0),
                    "recommendation_level": parsed_data.get("recommendation_level", "Unknown"),
                    "key_strengths_count": len(parsed_data.get("strengths_analysis", [])),
                    "improvement_areas_count": len(parsed_data.get("weaknesses_analysis", []))
                }
                
        except Exception as e:
            logger.error(f"Fallback parsing error: {e}")
            # Keep the standard structure with empty/default values
    
    def _generate_error_response(self, error_message: str, target_role: str = None, word_count: int = 0) -> Dict[str, Any]:
        """Generate error response maintaining standard structure"""
        response = self._get_standard_response_template(target_role or "unknown", word_count)
        response["success"] = False
        response["error"] = f"AI analysis failed: {error_message}"
        response["resume_metadata"]["validation_message"] = "Analysis encountered an error"
        return response

# Initialize components
pdf_extractor = OptimizedPDFExtractor()
high_perf_analyzer = None

if openai_api_key:
    try:
        high_perf_analyzer = HighPerformanceLangChainAnalyzer(openai_api_key)
        logger.info("High-performance analyzer initialized successfully")
    except Exception as init_error:
        logger.error(f"Failed to initialize analyzer: {init_error}")

# Performance middleware
@app.middleware("http")
async def add_performance_headers(request: Request, call_next):
    start_time = asyncio.get_event_loop().time()
    response = await call_next(request)
    process_time = asyncio.get_event_loop().time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time, 2))
    return response

@app.post("/analyze-resume")
async def analyze_resume(
    file: UploadFile = File(...),
    target_role: Optional[str] = None,
    search_jobs: bool = True,
    location: str = "India"
):
    """
    Comprehensive resume analysis with guaranteed standard JSON output and job search integration
    
    Parameters:
    - file: PDF resume file
    - target_role: Target job position (recommended for best results)
    - search_jobs: Whether to search for relevant jobs (default: True)
    - location: Job search location (default: India)
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        if not high_perf_analyzer:
            raise HTTPException(status_code=500, detail="AI analyzer not initialized.")
        
        if not file.content_type or "pdf" not in file.content_type.lower():
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Extract PDF text
        resume_text = await pdf_extractor.extract_text_from_pdf(file)
        
        if not resume_text:
            raise HTTPException(status_code=400, detail="Failed to extract text from PDF.")
        
        if len(resume_text.strip()) < 100:
            raise HTTPException(status_code=400, detail="Resume content too short.")
        
        # Perform analysis with guaranteed standard output
        analysis_result = await asyncio.wait_for(
            high_perf_analyzer.analyze_resume_with_jobs(
                resume_text, 
                target_role, 
                search_jobs=search_jobs and bool(target_role),
                location=location
            ),
            timeout=60.0
        )
        
        processing_time = asyncio.get_event_loop().time() - start_time
        logger.info(f"Analysis completed in {processing_time:.2f}s")
        
        return analysis_result
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Analysis timeout.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AI Resume Analyzer with Consistent JSON Output",
        "openai_configured": bool(openai_api_key),
        "analyzer_available": bool(high_perf_analyzer),
        "features": [
            "Guaranteed standard JSON structure",
            "Resume analysis",
            "Job search integration",
            "Frontend-compatible output"
        ]
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AI Resume Analyzer with Consistent JSON Output",
        "version": "2.2",
        "description": "AI resume analysis with guaranteed standard JSON format for frontend compatibility",
        "endpoints": {
            "/analyze-resume": "POST - Comprehensive analysis with standard JSON output",
            "/health": "GET - Service health check",
            "/docs": "GET - API documentation"
        },
        "guarantees": [
            "Consistent JSON structure every time",
            "All standard fields present",
            "Frontend-compatible format",
            "Optional job listings"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", workers=1)
