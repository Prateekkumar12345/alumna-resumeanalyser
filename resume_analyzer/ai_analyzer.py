from openai import OpenAI
import logging
from .config import ATS_KEYWORDS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIResumeAnalyzer:
    """AI-powered comprehensive resume analysis using OpenAI GPT models"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.ats_keywords = ATS_KEYWORDS
        self.client = None
        if api_key:
            self.client = OpenAI(api_key=api_key)

    def get_targeted_role_analysis(self, resume_text, target_role):
        """Backward-compatible alias for role-specific comprehensive analysis."""
        analysis = self.get_comprehensive_ai_analysis(resume_text, target_role)
        print("\n=== AI Targeted Role Analysis ===\n", analysis, "\n=== END ===")
        return analysis


    def set_api_key(self, api_key):
        """Set OpenAI API key for AI analysis"""
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
    
    def get_comprehensive_ai_analysis(self, resume_text, target_role=None):
        """
        Get comprehensive AI analysis with detailed insights
        """
        if not self.client:
            return "AI analysis requires OpenAI API key. Please configure your API key to access detailed AI insights."
        
        try:
            prompt = self._create_comprehensive_analysis_prompt(resume_text, target_role)
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": self._get_role_specific_system_prompt(target_role)
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1200,
                temperature=0.4
            )
            
            analysis = response.choices[0].message.content.strip()
            
            # ✅ print + log so you see it in uvicorn logs
            print("\n=== AI Comprehensive Analysis ===\n", analysis, "\n=== END ===")
            logger.info(f"Generated targeted {target_role} analysis successfully")
            return analysis
            
        except Exception as e:
            error_msg = f"Role-specific AI analysis unavailable: {str(e)}"
            logger.error(error_msg)
            print(error_msg)
            return error_msg

    def get_improvement_recommendations(self, resume_text, weaknesses_analysis):
        """
        Get AI-powered improvement recommendations based on identified weaknesses
        """
        if not self.client:
            return "AI improvement recommendations require OpenAI API key."
        
        try:
            prompt = self._create_improvement_prompt(resume_text, weaknesses_analysis)
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert career coach specializing in resume optimization and professional development. Provide specific, actionable improvement recommendations."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            recommendations = response.choices[0].message.content.strip()
            logger.info("Generated AI improvement recommendations successfully")
            return recommendations
            
        except Exception as e:
            error_msg = f"AI improvement recommendations unavailable: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _create_comprehensive_analysis_prompt(self, resume_text: str, target_role: str = None) -> str:
        """
        Build a detailed prompt for the comprehensive resume analysis.
        Includes ATS keywords and optional target role.
        """
        ats_keywords_str = ", ".join(self.ats_keywords)
        role_part = (
            f"\nFocus the analysis on suitability for the '{target_role}' role." 
            if target_role else ""
        )
        prompt = (
            f"Analyze the following resume thoroughly.{role_part}\n"
            f"Identify strengths, weaknesses, achievements, and alignment with ATS keywords.\n"
            f"ATS Keywords to consider: {ats_keywords_str}\n\n"
            f"Resume text:\n{resume_text}\n\n"
            "Provide a detailed, structured analysis including:\n"
            "- Key strengths & accomplishments\n"
            "- Gaps or weaknesses\n"
            "- ATS keyword matches and missing keywords\n"
            "- Suggestions for improvement to better fit the role\n"
            "- Overall rating (1-10) for ATS compatibility\n"
        )
        return prompt

    def _get_role_specific_system_prompt(self, target_role: str = None) -> str:
        """
        Return the system message instructing the model on how to act.
        """
        if target_role:
            return (
                f"You are an experienced HR specialist and ATS expert evaluating resumes for {target_role} positions. "
                "Give objective, detailed, and actionable feedback."
            )
        return (
            "You are an experienced HR specialist and ATS expert evaluating resumes across multiple roles. "
            "Give objective, detailed, and actionable feedback."
        )

    def validate_api_connection(self):
        """Validate OpenAI API connection and key"""
        if not self.api_key:
            return False, "No API key provided"
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            return True, "API key validated successfully"
        except Exception as e:
            error_type = type(e).__name__
            if "AuthenticationError" in error_type:
                return False, "Invalid API key - please check your OpenAI API key"
            elif "RateLimitError" in error_type:
                return False, "API rate limit exceeded - please try again later"
            elif "APIError" in error_type:
                return False, f"API error: {str(e)}"
            else:
                return False, f"Connection error: {str(e)}"
    
    def get_analysis_cost_estimate(self, resume_text, target_role=None):
        """
        Estimate cost for AI analysis based on token usage
        """
        resume_tokens = len(resume_text) // 4
        prompt_tokens = 1000  # Average prompt size
        response_tokens = 1500  # Average response size
        
        total_tokens = resume_tokens + prompt_tokens + response_tokens
        
        cost_per_1k_tokens = 0.002
        estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens

        analysis_types = ["Comprehensive Analysis"]
        if target_role:
            analysis_types.append("Role-Specific Analysis")
            estimated_cost *= 1.5  # Additional cost for targeted analysis

        return {
            'estimated_tokens': total_tokens,
            'estimated_cost_usd': round(estimated_cost, 4),
            'analysis_types': analysis_types,
            'cost_breakdown': {
                'input_tokens': resume_tokens + prompt_tokens,
                'output_tokens': response_tokens,
                'total_tokens': total_tokens
            }
        }
