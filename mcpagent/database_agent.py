import asyncio
import os
import traceback
import sys

# Fix for Windows asyncio subprocess issue - MUST be at the top
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.mcp import MCPTools
from typing import List, Dict, Any
from mcp import ClientSession, StdioServerParameters
from textwrap import dedent
from mcp.client.stdio import stdio_client
from agno.utils.log import logger
import signal

# Instructions for the Symptom Analysis Agent
SYMPTOM_ANALYZER_INSTRUCTIONS = dedent(
    """\
    You are a medical symptom analysis agent in a chat-based system. Your job is to analyze symptoms and conditions mentioned by users and identify appropriate medical specialties. You should respond conversationally, maintaining context from previous messages.

    For each user message:
    1. Analyze any symptoms or conditions provided
    2. Identify relevant medical specialties
    3. Return a comma-separated list of specialty names
    4. If no symptoms are provided, ask for clarification or more details
    5. Use the conversation history to refine your analysis if needed

    Example:
    - User: "I have chest pain and shortness of breath"
      Response: "Cardiologist, Cardiology, Heart Specialist"
    - User: "What about headaches?"
      Response: "Neurologist, Neurology, Brain Specialist"
    - User: "I'm not sure what's wrong"
      Response: "Could you describe any symptoms you're experiencing?"

    IMPORTANT RULES:
    - Return ONLY a comma-separated list of specialties if symptoms are clear
    - If clarification is needed, return a question as plain text
    - Use standard medical specialty names
    - Include common variations (e.g., "Cardiologist, Cardiology")
    - Do not provide explanations unless asked
    """
)

# Instructions for the Database Query Agent
DATABASE_QUERY_INSTRUCTIONS = dedent(
    """\
    You are an intelligent SQL assistant in a chat-based system, accessing a database via MCP tools. Your job is to find consultants based on provided medical specialties, maintaining conversational context.

    For each message:
    1. Use the provided specialties to search the database
    2. Use `get_schema` if needed to understand the database
    3. Generate SQL queries with flexible matching (LIKE operator)
    4. Return a formatted list of matching consultants
    5. If no specialties are provided, ask for clarification
    6. Use conversation history to refine searches

    Example SQL:
    ```sql
    SELECT * FROM consultants 
    WHERE specialty LIKE '%Cardiologist%' 
       OR specialty LIKE '%Cardiology%'
    ```

    Response format:
    - If found: "Found consultants: [Name1 - Specialty1, Name2 - Specialty2]"
    - If none: "No consultants found. Please provide more details or check spelling."
    - If unclear: "Please specify symptoms or specialties to search for."

    Constraints:
    - Use SELECT queries only
    - Focus on user-friendly responses
    """
)

load_dotenv()
MODEL_ID = os.getenv('MODEL_ID')
MODEL_API_KEY = os.getenv('MODEL_API_KEY')

if not MODEL_ID or not MODEL_API_KEY:
    raise ValueError('MODEL_ID and MODEL_API_KEY must be set')


class DatabaseAgent:
    """Manage database agent lifecycle and cleanup"""

    def __init__(self):
        self.session = None
        self.agent = None

    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.session and hasattr(self.session, 'close'):
                await self.session.close()
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")


def create_symptom_analyzer() -> Agent:
    """Create symptom analysis agent"""
    try:
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
    """Create database query agent"""
    try:
        mcp_tool = MCPTools(session=session)
        await mcp_tool.initialize()

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


async def process_chat_message(user_message: str, conversation_history: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Process a single chat message, maintaining conversation context.
    Returns a dictionary with response and status.
    """
    logger.info(f"Platform: {sys.platform}, Event loop policy: {type(asyncio.get_event_loop_policy()).__name__}")

    required_variables = ["DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME"]
    missing_variables = [var for var in required_variables if not os.getenv(var)]
    if missing_variables:
        return {"response": f"Error: Missing environment variables: {', '.join(missing_variables)}", "status": "error"}

    # Step 1: Analyze symptoms
    logger.info("Analyzing message for symptoms...")
    symptom_analyzer = create_symptom_analyzer()

    # Construct prompt with conversation history
    history_prompt = "\n".join([f"User: {msg['user']}\nAssistant: {msg['assistant']}" for msg in conversation_history])
    symptom_prompt = f"""
    Conversation history:
    {history_prompt}

    Current message: "{user_message}"

    Analyze the symptoms and return a comma-separated list of specialties or a question if clarification is needed.
    """

    specialty_response = await symptom_analyzer.arun(symptom_prompt)
    response_content = specialty_response.content.strip()

    # Check if clarification is needed
    if not response_content or "?" in response_content:
        return {"response": response_content or "Could you describe your symptoms?", "status": "clarification"}

    specialties = response_content
    logger.info(f"Identified specialties: {specialties}")

    # Step 2: Query database
    logger.info("Querying database for consultants...")
    logger.info(f"Using DB credentials: host={os.getenv('DB_HOST')}, user={os.getenv('DB_USER')}")

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

    logger.info(f"Server parameters: {server_parameters}")
    db_agent = DatabaseAgent()

    try:
        async with stdio_client(server_parameters) as (read, write):
            async with ClientSession(read, write) as session:
                db_agent.session = session
                await session.initialize()
                database_agent = await create_database_agent(session)

                db_prompt = f"""
                Conversation history:
                {history_prompt}

                Current message: "{user_message}"

                Search for consultants with these specialties: {specialties}
                Use flexible matching for any of: {specialties}
                """

                db_response = await database_agent.arun(db_prompt)
                return {"response": db_response.content.strip(), "status": "success"}

    except Exception as e:
        logger.error(f"Database query error: {e}\n{traceback.format_exc()}")
        return {"response": f"Error: Unable to fetch consultants: {str(e)}", "status": "error"}
    finally:
        await db_agent.cleanup()
        await asyncio.sleep(0.2)


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal. Cleaning up...")
    sys.exit(0)


async def main():
    """Test the chat functionality"""
    logger.info(f"Starting main function. Platform: {sys.platform}")
    logger.info(f"Event loop policy: {type(asyncio.get_event_loop_policy()).__name__}")

    conversation_history = []
    test_messages = [
        "I have chest pain and shortness of breath",
        "What about headaches?",
        "I'm not sure what's wrong"
    ]

    for msg in test_messages:
        logger.info(f"Processing message: {msg}")
        result = await process_chat_message(msg, conversation_history)
        logger.info(f"Response: {result['response']}")
        conversation_history.append({"user": msg, "assistant": result['response']})


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