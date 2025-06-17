import asyncio
from database_agent import process_chat_message  # Import from agent.py (assuming database_agent is agent.py)

async def call_function(message, conversation_history):
    result = await process_chat_message(message, conversation_history)
    print(result)
    return result

if __name__ == "__main__":
    message = "get a list of all consultants who treat unexplained weight loss, fatigue, lumps or thickening"
    conversation_history = []  # Empty history for initial message
    # Run the async function using asyncio
    result = asyncio.run(call_function(message, conversation_history))