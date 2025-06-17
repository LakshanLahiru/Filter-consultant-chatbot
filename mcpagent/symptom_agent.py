import os
from dotenv import load_dotenv
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from textwrap import dedent
from agno.utils.log import logger

load_dotenv()

# Instructions for the Symptom Analysis Agent
SYMPTOM_ANALYZER_INSTRUCTIONS = dedent(
    """\
    You are a medical symptom analysis agent. Your job is to analyze symptoms and conditions 
    mentioned by users and identify which medical specialties would be most appropriate.

    When given symptoms or conditions, you should:
    1. Analyze the symptoms carefully
    2. Identify the most relevant medical specialties that typically handle these symptoms
    3. Provide a list of specialty names that are commonly used in medical databases

    For example:
    - "chest pain, shortness of breath, palpitations" → ["Cardiologist", "Cardiology"]
    - "headache, dizziness, memory problems" → ["Neurologist", "Neurology"] 
    - "stomach pain, nausea, diarrhea" → ["Gastroenterologist", "Gastroenterology"]
    - "joint pain, back pain, fracture" → ["Orthopedic Surgeon", "Orthopedics", "Orthopedist"]
    - "skin rash, acne, hair loss" → ["Dermatologist", "Dermatology"]

    IMPORTANT RULES:
    - Return ONLY a comma-separated list of specialty names
    - Use standard medical specialty names as they would appear in a hospital database
    - Include common variations (e.g., "Cardiologist, Cardiology, Heart Specialist")
    - Do not provide explanations, just the specialty list
    - If unsure, include the most likely specialties

    Example responses:
    "Cardiologist, Cardiology, Heart Specialist"
    "Neurologist, Neurology, Brain Specialist"
    "Gastroenterologist, Gastroenterology, Digestive Specialist"
    """
)


class SymptomAnalyzer:
    """Symptom Analysis Agent for identifying medical specialties"""

    def __init__(self):
        self.model_id = os.getenv('MODEL_ID')
        self.model_api_key = os.getenv('MODEL_API_KEY')

        if not self.model_id or not self.model_api_key:
            raise ValueError('MODEL_ID and MODEL_API_KEY must be set in environment variables')

        self.agent = self._create_agent()

    def _create_agent(self) -> Agent:
        """Create the symptom analysis agent"""
        try:
            # Try different parameter names for OpenAIChat
            try:
                openai_model = OpenAIChat(model=self.model_id, api_key=self.model_api_key)
            except TypeError:
                try:
                    openai_model = OpenAIChat(model_name=self.model_id, api_key=self.model_api_key)
                except TypeError:
                    openai_model = OpenAIChat(self.model_id, api_key=self.model_api_key)

            return Agent(
                model=openai_model,
                instructions=SYMPTOM_ANALYZER_INSTRUCTIONS,
                markdown=False,
                show_tool_calls=False,
            )
        except Exception as e:
            logger.error(f"Error creating symptom analyzer: {e}")
            raise

    async def analyze_symptoms(self, user_query: str) -> str:
        """
        Analyze symptoms and return relevant medical specialties

        Args:
            user_query: User's symptom description

        Returns:
            Comma-separated list of medical specialties
        """
        try:
            logger.info(f"Analyzing symptoms: {user_query}")

            # Create a specific prompt for the symptom analyzer
            symptom_analysis_prompt = f"""
            Analyze these symptoms/conditions and return the relevant medical specialties:

            User query: "{user_query}"

            Return only the specialty names, comma-separated.
            """

            response = await self.agent.arun(symptom_analysis_prompt)
            specialties = response.content.strip()

            logger.info(f"Identified specialties: {specialties}")
            return specialties

        except Exception as e:
            logger.error(f"Error analyzing symptoms: {e}")
            raise RuntimeError(f"Failed to analyze symptoms: {e}") from e


# Factory function for backward compatibility
def create_symptom_analyzer() -> SymptomAnalyzer:
    """Create a new SymptomAnalyzer instance"""
    return SymptomAnalyzer()


if __name__ == "__main__":
    import asyncio


    async def test_symptom_analyzer():
        """Test the symptom analyzer"""
        analyzer = SymptomAnalyzer()

        test_queries = [
            "chest pain and shortness of breath",
            "headache and dizziness",
            "unexplained weight loss, fatigue, lumps or thickening"
        ]

        for query in test_queries:
            print(f"\nQuery: {query}")
            try:
                specialties = await analyzer.analyze_symptoms(query)
                print(f"Specialties: {specialties}")
            except Exception as e:
                print(f"Error: {e}")


    asyncio.run(test_symptom_analyzer())