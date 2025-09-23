from fastapi import FastAPI, HTTPException, File, UploadFile
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
app = FastAPI(title="AI Resume Analyzer API - LangChain Powered")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client through LangChain
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.warning("OPENAI_API_KEY not found in environment variables")

# Pydantic models for structured output
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

class PDFExtractor:
    """Enhanced PDF text extraction with error handling"""
    
    @staticmethod
    async def extract_text_from_pdf(uploaded_file) -> Optional[str]:
        try:
            # Reset file pointer
            uploaded_file.seek(0)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                content = await uploaded_file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            try:
                with open(tmp_file_path, 'rb') as pdf_file:
                    pdf_reader = PdfReader(pdf_file)
                    extracted_text = ""
                    
                    for page_num, page in enumerate(pdf_reader.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                extracted_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                        except Exception as page_error:
                            logger.warning(f"Error extracting page {page_num + 1}: {str(page_error)}")
                    
                    return extracted_text.strip()
            finally:
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            return None

class LangChainResumeAnalyzer:
    """Advanced AI-powered resume analysis using LangChain"""
    
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model_name="gpt-4",
            temperature=0.2,
            max_tokens=4000
        )
        
        # Create output parser
        self.output_parser = PydanticOutputParser(pydantic_object=ResumeAnalysis)
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.output_parser, llm=self.llm)
        
        # Create analysis chain
        self.analysis_chain = self._create_analysis_chain()
    
    def _create_analysis_chain(self) -> LLMChain:
        """Create the main analysis chain with structured prompts"""
        
        system_prompt = """You are an expert ATS (Applicant Tracking System) resume analyzer and senior career coach with 15+ years of experience. You have deep knowledge of:
        - ATS optimization and keyword matching
        - Industry-specific requirements across all sectors
        - Quantified achievement identification
        - Professional presentation standards
        - Job market trends and salary expectations
        
        Your analysis must be thorough, specific, and actionable. Always provide concrete examples and evidence-based recommendations."""
        
        human_prompt_template = """
        Analyze this resume comprehensively. Target Role: {target_role}

        RESUME CONTENT:
        {resume_text}

        Provide a complete analysis with:
        
        1. PROFESSIONAL PROFILE ASSESSMENT:
        - Determine exact experience level (e.g., "3-5 years", "Senior level 8+ years")
        - Count and categorize technical skills
        - Evaluate project portfolio depth and complexity
        - Identify and rate quantified achievements
        - Assess technical sophistication level
        
        2. CONTACT & PRESENTATION EVALUATION:
        - Verify contact information completeness
        - Evaluate resume structure and formatting
        - Assess professional presentation quality
        - Analyze content optimization for ATS
        
        3. DETAILED SCORING (provide exact scores):
        - Contact Information: Score out of 15 points
        - Technical Skills: Score out of 30 points  
        - Experience Quality: Score out of 25 points
        - Quantified Achievements: Score out of 20 points
        - Content Optimization: Score out of 10 points
        
        4. COMPREHENSIVE STRENGTHS (identify at least 5):
        For each strength, explain why it's strong, ATS benefits, competitive advantages, and supporting evidence.
        
        5. DETAILED WEAKNESSES (identify at least 5):
        For each weakness, explain the problem, ATS impact, how it hurts candidacy, fix priority (CRITICAL/HIGH/MEDIUM), specific solutions, and implementation timeline.
        
        6. ACTIONABLE IMPROVEMENT PLAN:
        - Critical fixes requiring immediate attention
        - High priority improvements (1-2 months)
        - Medium priority enhancements (2-6 months)
        - Specific timeline with tasks and priorities
        
        7. JOB MARKET ANALYSIS:
        - Role compatibility assessment
        - Market positioning insights
        - Career advancement recommendations
        - Skill development priorities
        
        Provide specific, evidence-based analysis with concrete examples from the resume content.
        
        {format_instructions}
        """
        
        prompt = PromptTemplate(
            template=human_prompt_template,
            input_variables=["resume_text", "target_role"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
        
        return LLMChain(llm=self.llm, prompt=prompt)
    
    async def analyze_resume(self, resume_text: str, target_role: Optional[str] = None) -> Dict[str, Any]:
        """Main analysis method using LangChain"""
        try:
            # Prepare inputs
            role_context = target_role or "General position"
            
            # Run the analysis chain
            result = await self.analysis_chain.arun(
                resume_text=resume_text,
                target_role=role_context
            )
            
            # Parse the structured output
            try:
                parsed_analysis = self.fixing_parser.parse(result)
                return self._format_response(parsed_analysis, resume_text, target_role)
            except Exception as parse_error:
                logger.warning(f"Structured parsing failed: {parse_error}")
                return await self._fallback_analysis(resume_text, target_role, result)
                
        except Exception as e:
            logger.error(f"LangChain analysis error: {str(e)}")
            return self._generate_error_response(str(e))
    
    async def _fallback_analysis(self, resume_text: str, target_role: Optional[str], raw_result: str) -> Dict[str, Any]:
        """Fallback analysis when structured parsing fails"""
        
        # Create a simplified analysis chain for fallback
        fallback_prompt = PromptTemplate(
            template="""
            Based on this resume analysis, provide a JSON response with key insights:
            
            Resume: {resume_text}
            Target Role: {target_role}
            Analysis: {raw_result}
            
            Focus on actionable recommendations and specific improvements needed.
            """,
            input_variables=["resume_text", "target_role", "raw_result"]
        )
        
        fallback_chain = LLMChain(llm=self.llm, prompt=fallback_prompt)
        
        try:
            fallback_result = await fallback_chain.arun(
                resume_text=resume_text[:2000],  # Truncate for token limits
                target_role=target_role or "General",
                raw_result=raw_result[:1000]
            )
            
            return {
                "success": True,
                "analysis_method": "AI Fallback Analysis",
                "comprehensive_analysis": raw_result,
                "structured_insights": fallback_result,
                "recommendations": "Review the comprehensive analysis for detailed insights",
                "target_role_analysis": f"Analysis focused on {target_role}" if target_role else "General analysis provided"
            }
        except Exception as fallback_error:
            logger.error(f"Fallback analysis error: {fallback_error}")
            return self._generate_error_response(f"Analysis failed: {fallback_error}")
    
    def _format_response(self, analysis: ResumeAnalysis, resume_text: str, target_role: Optional[str]) -> Dict[str, Any]:
        """Format the parsed analysis into the expected response structure"""
        
        word_count = len(resume_text.split())
        
        return {
            "success": True,
            "analysis_method": "AI-Powered LangChain Analysis",
            "resume_metadata": {
                "word_count": word_count,
                "validation_message": "Comprehensive AI analysis completed",
                "target_role": target_role or "General analysis"
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
                "improvement_areas_count": len(analysis.weaknesses_analysis)
            }
        }
    
    def _generate_error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate structured error response"""
        return {
            "success": False,
            "error": f"AI analysis failed: {error_message}",
            "message": "Resume analysis encountered an error. Please try again.",
            "analysis_method": "Error Response",
            "recommendations": "Please check your resume format and try again"
        }

# Initialize components
pdf_extractor = PDFExtractor()

# Initialize LangChain analyzer only if API key is available
langchain_analyzer = None
if openai_api_key:
    try:
        langchain_analyzer = LangChainResumeAnalyzer(openai_api_key)
        logger.info("LangChain analyzer initialized successfully")
    except Exception as init_error:
        logger.error(f"Failed to initialize LangChain analyzer: {init_error}")

@app.post("/analyze-resume")
async def analyze_resume(
    file: UploadFile = File(...),
    target_role: Optional[str] = None
):
    """
    Advanced AI-powered resume analysis using LangChain
    """
    try:
        # Check if analyzer is available
        if not langchain_analyzer:
            raise HTTPException(
                status_code=500, 
                detail="AI analyzer not initialized. Please configure OPENAI_API_KEY."
            )
        
        # Validate file type
        if not file.content_type or "pdf" not in file.content_type.lower():
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Extract text from PDF
        resume_text = await pdf_extractor.extract_text_from_pdf(file)
        
        if not resume_text:
            raise HTTPException(status_code=400, detail="Failed to extract text from PDF. Please ensure it's a valid PDF with text content.")
        
        # Validate minimum content length
        if len(resume_text.strip()) < 100:
            raise HTTPException(status_code=400, detail="Resume content too short for comprehensive analysis (minimum 100 characters required)")
        
        # Perform comprehensive AI analysis
        analysis_result = await langchain_analyzer.analyze_resume(resume_text, target_role)
        
        return analysis_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/quick-analyze")
async def quick_analyze_resume(
    file: UploadFile = File(...),
    target_role: Optional[str] = None
):
    """
    Quick AI analysis for faster response times
    """
    try:
        if not langchain_analyzer:
            raise HTTPException(status_code=500, detail="AI analyzer not available")
        
        if not file.content_type or "pdf" not in file.content_type.lower():
            raise HTTPException(status_code=400, detail="Only PDF files supported")
        
        resume_text = await pdf_extractor.extract_text_from_pdf(file)
        if not resume_text or len(resume_text.strip()) < 50:
            raise HTTPException(status_code=400, detail="Invalid or empty resume content")
        
        # Quick analysis with simplified prompt
        quick_prompt = f"""
        Provide a quick resume analysis for: {target_role or 'general position'}
        
        Resume: {resume_text[:2000]}  # Truncate for speed
        
        Give brief insights on:
        1. Overall quality (score out of 100)
        2. Top 3 strengths
        3. Top 3 improvements needed
        4. Job readiness level
        """
        
        quick_analysis = await langchain_analyzer.llm.apredict(quick_prompt)

        
        return {
            "success": True,
            "analysis_type": "Quick Analysis",
            "target_role": target_role,
            "insights": quick_analysis,
            "recommendation": "For detailed analysis, use the full /analyze-resume endpoint"
        }
        
    except Exception as e:
        logger.error(f"Quick analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Enhanced health check"""
    return {
        "status": "healthy",
        "service": "AI Resume Analyzer - LangChain Powered",
        "openai_configured": bool(openai_api_key),
        "langchain_analyzer": bool(langchain_analyzer),
        "endpoints": ["/analyze-resume", "/quick-analyze", "/health"],
        "features": ["Full AI Analysis", "Structured Output", "Quick Analysis", "PDF Processing"]
    }

@app.get("/")
async def root():
    """Enhanced root endpoint"""
    return {
        "service": "AI-Powered Resume Analyzer",
        "version": "2.0 - LangChain Enhanced",
        "description": "Comprehensive AI resume analysis using advanced language models",
        "endpoints": {
            "/analyze-resume": "POST - Comprehensive AI resume analysis",
            "/quick-analyze": "POST - Quick AI resume insights", 
            "/health": "GET - Service health check",
            "/docs": "GET - API documentation"
        },
        "requirements": {
            "api_key": "OPENAI_API_KEY environment variable required",
            "file_format": "PDF files only",
            "content": "Text-based resumes (not image-only PDFs)"
        },
        "features": [
            "100% AI-powered analysis",
            "Structured output format",
            "ATS optimization insights",
            "Job market analysis", 
            "Actionable recommendations"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )