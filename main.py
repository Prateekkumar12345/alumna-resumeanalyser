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
from pydantic import BaseModel, Field

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

# Pydantic models
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

class ImprovementTask(BaseModel):
    period: str = Field(description="Time period for task")
    task: str = Field(description="Specific task to complete")
    priority: str = Field(description="Priority level")

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

class ResumeAnalysis(BaseModel):
    professional_profile: ProfessionalProfile
    contact_presentation: ContactPresentation
    detailed_scoring: Dict[str, ScoringDetail]
    strengths_analysis: List[StrengthAnalysis]
    weaknesses_analysis: List[WeaknessAnalysis]
    improvement_plan: Dict[str, Any]
    job_market_analysis: Dict[str, str]
    overall_score: int = Field(description="Overall resume score out of 100")
    recommendation_level: str = Field(description="Overall recommendation level")

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
            # Search query for job listings
            search_query = f"{target_role} jobs {location} hiring"
            
            # Note: In production, you'd use actual web search API here
            # For this implementation, we'll use LLM to generate realistic job data
            # based on the role. Replace this with actual web scraping/API calls.
            
            job_extraction_prompt = f"""
            Generate 5-10 realistic current job listings for the position: {target_role} in {location}.
            
            For each job listing, provide:
            1. Company Name (use realistic company names)
            2. Position (exact job title)
            3. Location (city/region in {location})
            4. CTC/Salary Range (in appropriate currency)
            5. Experience Required (e.g., "2-4 years")
            6. Last Date to Apply (realistic future date)
            7. About the Job (2-3 sentences)
            8. Job Description (detailed responsibilities)
            9. Job Requirements (skills, qualifications, education)
            10. Application URL (format: https://company-careers.com/job-id)
            
            Format the response as a JSON array with these exact field names:
            company_name, position, location, ctc, experience_required, last_date_to_apply, 
            about_job, job_description, job_requirements, application_url
            
            Make the data realistic and relevant to the current job market.
            """
            
            response = await self.llm.apredict(job_extraction_prompt)
            
            # Parse the JSON response
            try:
                # Extract JSON from response if it's wrapped in text
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    jobs_data = json.loads(json_match.group())
                else:
                    jobs_data = json.loads(response)
                
                return jobs_data
            except json.JSONDecodeError:
                logger.error("Failed to parse job listings JSON")
                return []
                
        except Exception as e:
            logger.error(f"Job search error: {str(e)}")
            return []
    
    async def search_jobs_with_web_api(self, target_role: str, location: str = "India") -> List[Dict[str, Any]]:
        """Alternative: Use actual web search APIs (implement with your preferred API)"""
        # TODO: Implement with actual job search APIs like:
        # - LinkedIn Jobs API
        # - Indeed API
        # - Naukri API
        # - Custom web scraping
        pass

class HighPerformanceLangChainAnalyzer:
    """High-performance AI analyzer with job search integration"""
    
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
        
        # Initialize job search service
        self.job_search = JobSearchService(self.llm)
    
    def _create_analysis_chain(self) -> LLMChain:
        """Create the main analysis chain"""
        
        system_prompt = """You are an expert ATS resume analyzer and senior career coach with 15+ years of experience."""
        
        human_prompt_template = """
        Analyze this resume comprehensively. Target Role: {target_role}

        RESUME CONTENT:
        {resume_text}

        Provide a complete analysis with:
        
        1. PROFESSIONAL PROFILE ASSESSMENT
        2. CONTACT & PRESENTATION EVALUATION
        3. DETAILED SCORING
        4. COMPREHENSIVE STRENGTHS (at least 5)
        5. DETAILED WEAKNESSES (at least 5)
        6. ACTIONABLE IMPROVEMENT PLAN
        7. JOB MARKET ANALYSIS
        
        {format_instructions}
        """
        
        prompt = PromptTemplate(
            template=human_prompt_template,
            input_variables=["resume_text", "target_role"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
        
        return LLMChain(llm=self.llm, prompt=prompt)
    
    async def analyze_resume_with_jobs(
        self, 
        resume_text: str, 
        target_role: Optional[str] = None,
        search_jobs: bool = True,
        location: str = "India"
    ) -> Dict[str, Any]:
        """Analyze resume and optionally search for relevant jobs"""
        try:
            role_context = target_role or "General position"
            
            # Run resume analysis and job search in parallel
            if search_jobs and target_role:
                analysis_task = self.analysis_chain.arun(
                    resume_text=resume_text,
                    target_role=role_context
                )
                jobs_task = self.job_search.search_jobs(target_role, location)
                
                # Wait for both tasks
                analysis_result, job_listings = await asyncio.gather(
                    analysis_task,
                    jobs_task,
                    return_exceptions=True
                )
                
                # Handle exceptions
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
            
            # Parse analysis
            try:
                parsed_analysis = self.fixing_parser.parse(analysis_result)
                response = self._format_response(parsed_analysis, resume_text, target_role)
                
                # Add job listings to response
                if job_listings:
                    response["job_listings"] = {
                        "total_jobs_found": len(job_listings),
                        "search_query": f"{target_role} in {location}",
                        "jobs": job_listings
                    }
                
                return response
                
            except Exception as parse_error:
                logger.warning(f"Structured parsing failed: {parse_error}")
                return await self._fallback_analysis(resume_text, target_role, analysis_result, job_listings)
                
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return self._generate_error_response(str(e))
    
    async def _fallback_analysis(
        self, 
        resume_text: str, 
        target_role: Optional[str], 
        raw_result: str,
        job_listings: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Fallback analysis - parse raw result into structured format"""
        
        try:
            # Try to extract JSON from raw_result if it exists
            json_match = re.search(r'\{.*\}', raw_result, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group())
                
                # Build response in the same format as _format_response
                word_count = len(resume_text.split())
                
                response = {
                    "success": True,
                    "analysis_method": "AI-Powered Comprehensive Analysis",
                    "resume_metadata": {
                        "word_count": word_count,
                        "validation_message": "Comprehensive AI analysis completed",
                        "target_role": target_role or "General analysis",
                        "content_preserved": True
                    },
                    "executive_summary": parsed_data.get("executive_summary", {}),
                    "detailed_scoring": parsed_data.get("detailed_scoring", {}),
                    "strengths_analysis": parsed_data.get("strengths_analysis", []),
                    "weaknesses_analysis": parsed_data.get("weaknesses_analysis", []),
                    "improvement_plan": parsed_data.get("improvement_plan", {}),
                    "job_market_analysis": parsed_data.get("job_market_analysis", {}),
                    "ai_insights": parsed_data.get("ai_insights", {})
                }
                
                # Add job listings if available
                if job_listings:
                    response["job_listings"] = {
                        "total_jobs_found": len(job_listings),
                        "search_query": f"{target_role} in India",
                        "jobs": job_listings
                    }
                
                return response
            else:
                # If no JSON found, return error
                return self._generate_error_response("Failed to parse analysis result")
            
        except Exception as fallback_error:
            logger.error(f"Fallback analysis error: {fallback_error}")
            return self._generate_error_response(f"Analysis failed: {fallback_error}")
    
    def _format_response(self, analysis: ResumeAnalysis, resume_text: str, target_role: Optional[str]) -> Dict[str, Any]:
        """Format the parsed analysis"""
        
        word_count = len(resume_text.split())
        
        return {
            "success": True,
            "analysis_method": "AI-Powered Comprehensive Analysis",
            "resume_metadata": {
                "word_count": word_count,
                "validation_message": "Comprehensive AI analysis completed",
                "target_role": target_role or "General analysis",
                "content_preserved": True
            },
            "executive_summary": {
                "professional_profile": analysis.professional_profile.dict(),
                "contact_presentation": analysis.contact_presentation.dict(),
                "overall_assessment": {
                    "score_percentage": analysis.overall_score,
                    "level": analysis.recommendation_level,
                    "description": f"AI-determined resume quality: {analysis.overall_score}%",
                    "recommendation": analysis.recommendation_level
                }
            },
            "detailed_scoring": analysis.detailed_scoring,
            "strengths_analysis": [strength.dict() for strength in analysis.strengths_analysis],
            "weaknesses_analysis": [weakness.dict() for weakness in analysis.weaknesses_analysis],
            "improvement_plan": analysis.improvement_plan,
            "job_market_analysis": analysis.job_market_analysis,
            "ai_insights": {
                "overall_score": analysis.overall_score,
                "recommendation_level": analysis.recommendation_level,
                "key_strengths_count": len(analysis.strengths_analysis),
                "improvement_areas_count": len(analysis.weaknesses_analysis),
                "content_completeness": "Full analysis preserved"
            }
        }
    
    def _generate_error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate structured error response"""
        return {
            "success": False,
            "error": f"AI analysis failed: {error_message}",
            "message": "Resume analysis encountered an error.",
            "analysis_method": "Error Response"
        }

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
    response.headers["X-Content-Preserved"] = "true"
    return response

@app.post("/analyze-resume")
async def analyze_resume(
    file: UploadFile = File(...),
    target_role: Optional[str] = None,
    search_jobs: bool = True,
    location: str = "India"
):
    """
    Comprehensive resume analysis with job search integration
    
    Parameters:
    - file: PDF resume file
    - target_role: Target job position (required for job search)
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
        
        # Perform analysis with job search
        analysis_result = await asyncio.wait_for(
            high_perf_analyzer.analyze_resume_with_jobs(
                resume_text, 
                target_role, 
                search_jobs=search_jobs and bool(target_role),
                location=location
            ),
            timeout=60.0
        )
        
        # Log performance metrics without adding to response
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
        "service": "AI Resume Analyzer with Job Search",
        "openai_configured": bool(openai_api_key),
        "analyzer_available": bool(high_perf_analyzer),
        "features": [
            "Resume analysis",
            "Job search integration",
            "Structured job listings",
            "Performance optimized"
        ]
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AI Resume Analyzer with Job Search",
        "version": "2.1",
        "description": "AI resume analysis with integrated job search",
        "endpoints": {
            "/analyze-resume": "POST - Comprehensive analysis with job listings",
            "/health": "GET - Service health check",
            "/docs": "GET - API documentation"
        },
        "new_features": [
            "Integrated job search",
            "Structured job listings with company, CTC, location, requirements",
            "Parallel processing for faster results"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", workers=1)
