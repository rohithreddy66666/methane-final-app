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

    def generate_time_series_conclusion(self, time_series_results):
        try:
            prompt = f"""Based on the following methane time series analysis results, provide a clear, data-driven conclusion about the trends and predictions:

            Analysis Results:
            - LSTM Model Forecast:
            * Average Predicted Level: {time_series_results['monthly_means']['lstm']:.2f} ppb

            - Nixtla GPT Forecast:
            * Average Predicted Level: {time_series_results['monthly_means']['nixtla']:.2f} ppb

            Please provide:
            1. Analysis of the predicted trends
            2. Comparison between LSTM and Nixtla predictions
            3. Assessment of model performance
            4. Key insights about future methane levels"""

            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )
            
            return {
                "status": "success",
                "conclusion": response.choices[0].message.content
            }
            
        except Exception as e:
            logger.error(f"Error generating time series conclusion: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to generate conclusion: {str(e)}"
            }

    def generate_conclusion(self, analysis_results):
        try:
            # Format the analysis results into a clear prompt
            prompt = f"""Based on the following methane analysis results, provide a clear, concise conclusion about the methane levels and their implications:

            Location Analysis:
            - Maximum Concentration: {analysis_results['statistics']['max_concentration']:.2f} ppm x m
            - Average Concentration: {analysis_results['statistics']['avg_concentration']:.2f} ppm x m
            - Significant Methane Pixels: {analysis_results['statistics']['significant_methane']}
            - Total Pixels Analyzed: {analysis_results['statistics']['total_pixels']}
            - Percentage Area Affected: {analysis_results['statistics']['percentage_significant']:.2f}%
            - Detection Threshold: {analysis_results['statistics']['methane_threshold']} ppm x m
            
            Geographic Coverage:
            - Latitude Range: {analysis_results['coordinates']['min_lat']:.4f}° to {analysis_results['coordinates']['max_lat']:.4f}°
            - Longitude Range: {analysis_results['coordinates']['min_lon']:.4f}° to {analysis_results['coordinates']['max_lon']:.4f}°
            - talk which place can it be 
            
            Please provide:
            1. Assessment of methane concentration levels
            2. Potential environmental implications
            3. Whether these levels warrant attention
            4. Any notable patterns or hotspots"""

            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            return {
                "status": "success",
                "conclusion": response.choices[0].message.content
            }
            
        except Exception as e:
            logger.error(f"Error generating conclusion: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to generate conclusion: {str(e)}"
            }
