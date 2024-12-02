import os
from openai import OpenAI
from dotenv import load_dotenv
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class BotHandler:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.system_prompt = """You are an AI assistant specializing in methane analysis and interpretation. 
        You help users understand:
        
        1. EMIT Data Analysis:
           - Interpretation of methane plume detection
           - Understanding MAG1C algorithm results
           - Analysis of detection confidence levels
           - Geographic distribution of methane sources
        
        2. Time Series Analysis:
           - LSTM model predictions
           - Nixtla GPT forecasting results
           - Trend analysis and patterns
           - Seasonal variations in methane levels
        
        3. Environmental Context:
           - Impact assessment of detected methane levels
           - Comparison with global methane standards
           - Suggestions for mitigation strategies
           - Relevant environmental regulations
        
        4. Technical Assistance:
           - Model performance metrics
           - Data quality assessment
           - Methodology explanations
           - Best practices for analysis
        
        Base your responses on scientific principles and available data.
        When unsure, acknowledge limitations and suggest additional data needs."""
        
    def get_response(self, message, context=None):
        """
        Get response from ChatGPT API
        """
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            
            if context:
                messages.append({
                    "role": "system",
                    "content": f"Current analysis context: {context}"
                })
            
            messages.append({"role": "user", "content": message})
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                temperature=0.7,
                max_tokens=500,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            return {
                "status": "success",
                "response": response.choices[0].message.content
            }
            
        except Exception as e:
            logger.error(f"Error in chat response: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to generate response: {str(e)}"
            }
    
    def enhance_context(self, emit_data=None, time_series_data=None):
        """
        Enhance chat context with current analysis data
        """
        context = []
        
        if emit_data:
            if isinstance(emit_data, dict):
                if emit_data.get('has_plumes'):
                    context.append(
                        f"Methane plumes detected with max concentration of "
                        f"{emit_data.get('statistics', {}).get('max_concentration', 0)} ppm"
                    )
                else:
                    context.append("No significant methane plumes detected in the analyzed area")
                    
        if time_series_data:
            if isinstance(time_series_data, dict):
                lstm_mean = time_series_data.get('monthly_means', {}).get('lstm', 0)
                nixtla_mean = time_series_data.get('monthly_means', {}).get('nixtla', 0)
                context.append(
                    f"Recent forecasts show average methane levels of {lstm_mean:.2f} (LSTM) "
                    f"and {nixtla_mean:.2f} (Nixtla) ppb"
                )
                
        return " | ".join(context) if context else None