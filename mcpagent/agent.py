import asyncio
import os
from dotenv import load_dotenv
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.tools.mcp import MCPTools
from typing import Optional, List
from mcp import ClientSession, StdioServerParameters
from textwrap import dedent
from mcp.client.stdio import stdio_client
from agno.utils.log import logger
import signal
import sys

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

# Instructions for the Database Query Agent
DATABASE_QUERY_INSTRUCTIONS = dedent(
    """\
    You are an intelligent SQL assistant with access to a database through the MCP tool.

    Your job is to:
    1. Use the provided list of medical specialties to search the database
    2. Use the `get_schema` tool to understand the database structure if needed
    3. Generate SQL queries to find consultants who match ANY of the provided specialties
    4. Use flexible matching (LIKE operator) to find partial matches
    5. Present results in a clear, user-friendly format

    When searching for specialties:
    - Use OR conditions to match any of the provided specialties
    - Use LIKE operator with % wildcards for flexible matching
    - Search in relevant columns (specialty, specialization, department, etc.)

    Example SQL pattern:
    ```sql
    SELECT * FROM consultants 
    WHERE specialty LIKE '%Cardiologist%' 
       OR specialty LIKE '%Cardiology%' 
       OR specialty LIKE '%Heart%'
    ```

    Always provide:
    - Clear list of matching consultants
    - Their specialties
    - Any other relevant information available
    - If no matches found, suggest checking spelling or provide general advice

    Constraints:
    - Only use SELECT queries
    - Do not perform INSERT, UPDATE, DELETE operations
    - Focus on finding the best matches for the user's needs
    """
)

load_dotenv()
MODEL_ID = os.getenv('MODEL_ID')
MODEL_API_KEY = os.getenv('MODEL_API_KEY')

if not MODEL_ID or not MODEL_API_KEY:
    raise ValueError('MODEL_ID and MODEL_API_KEY must be set')


class DatabaseAgent:
    """A class to manage database agent lifecycle and cleanup"""

    def __init__(self):
        self.session = None
        self.agent = None

    async def cleanup(self):
        """Properly cleanup resources"""
        try:
            if self.session and hasattr(self.session, 'close'):
                await self.session.close()
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")


def create_symptom_analyzer() -> Agent:
    """Create the symptom analysis agent (no database connection needed)"""
    try:
        # Try different parameter names for OpenAIChat
        try:
            openai_model = OpenAIChat(model=MODEL_ID, api_key=MODEL_API_KEY)
        except TypeError:
            try:
                openai_model = OpenAIChat(model_name=MODEL_ID, api_key=MODEL_API_KEY)
            except TypeError:
                openai_model = OpenAIChat(MODEL_ID, api_key=MODEL_API_KEY)

        return Agent(
            model=openai_model,
            instructions=SYMPTOM_ANALYZER_INSTRUCTIONS,
            markdown=False,
            show_tool_calls=False,
        )
    except Exception as e:
        logger.error(f"Error creating symptom analyzer: {e}")
        raise


async def create_database_agent(session: ClientSession) -> Agent:
    """Create the database query agent with MCP tools"""
    try:
        mcp_tool = MCPTools(session=session)
        await mcp_tool.initialize()

        # Try different parameter names for OpenAIChat
        try:
            openai_model = OpenAIChat(model=MODEL_ID, api_key=MODEL_API_KEY)
        except TypeError:
            try:
                openai_model = OpenAIChat(model_name=MODEL_ID, api_key=MODEL_API_KEY)
            except TypeError:
                openai_model = OpenAIChat(MODEL_ID, api_key=MODEL_API_KEY)

        return Agent(
            model=openai_model,
            tools=[mcp_tool],
            instructions=DATABASE_QUERY_INSTRUCTIONS,
            markdown=False,
            show_tool_calls=True,
        )
    except Exception as e:
        logger.error(f"Error creating database agent: {e}")
        raise


async def analyze_symptoms_and_find_consultants(user_query: str) -> RunResponse:
    """
    Two-step process:
    1. Analyze symptoms to identify relevant specialties
    2. Query database for consultants in those specialties
    """
    required_variables = ["DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME"]
    missing_variables = [var for var in required_variables if not os.getenv(var)]
    if missing_variables:
        raise ValueError(f'Missing required environment variables: {", ".join(missing_variables)}')

    # Step 1: Analyze symptoms to get specialties
    logger.info("Step 1: Analyzing symptoms to identify specialties...")
    symptom_analyzer = create_symptom_analyzer()

    # Create a specific prompt for the symptom analyzer
    symptom_analysis_prompt = f"""
    Analyze these symptoms/conditions and return the relevant medical specialties:

    User query: "{user_query}"

    Return only the specialty names, comma-separated.
    """

    specialty_response = await symptom_analyzer.arun(symptom_analysis_prompt)
    specialties = specialty_response.content.strip()

    logger.info(f"Identified specialties: {specialties}")

    # Step 2: Query database using identified specialties
    logger.info("Step 2: Querying database for consultants...")

    server_parameters = StdioServerParameters(
        command='uvx',
        args=[
            'mcp-sql-server',
            "--db-host", os.getenv("DB_HOST"),
            "--db-user", os.getenv("DB_USER"),
            "--db-password", os.getenv("DB_PASSWORD"),
            "--db-database", os.getenv("DB_NAME"),
        ],
    )

    db_agent = DatabaseAgent()

    try:
        async with stdio_client(server_parameters) as (read, write):
            async with ClientSession(read, write) as session:
                db_agent.session = session
                await session.initialize()

                database_agent = await create_database_agent(session)

                # Create database query prompt
                db_query_prompt = f"""
                Find consultants who specialize in treating these conditions: {user_query}

                Based on the symptom analysis, search for consultants with these specialties: {specialties}

                Use flexible matching to find consultants whose specialties match any of: {specialties}
                """

                response = await database_agent.arun(db_query_prompt)
                return response

    except Exception as e:
        logger.error(f"Error in database query: {str(e)}")
        raise RuntimeError(f"Error connecting to database or running query: {e}") from e
    finally:
        await db_agent.cleanup()
        await asyncio.sleep(0.2)


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info("Received shutdown signal. Cleaning up...")
    sys.exit(0)


async def main():
    """Example usage of the two-agent system"""

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

    try:
        logger.info("Starting two-agent medical system...")

        # Test with symptom-based query
        symptom_query = "get a list of all consultants who treat   unexplained weight loss, fatigue, lumps or thickening"

        logger.info(f"Processing query: {symptom_query}")
        response = await analyze_symptoms_and_find_consultants(symptom_query)
        logger.info(f"Final Response: {response.content}")

    except ValueError as ve:
        logger.error(f"Configuration error: {ve}")
        return 1
    except RuntimeError as re:
        logger.error(f"Runtime error: {re}")
        return 1
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return 1

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Program interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)