import streamlit as st
import asyncio
import logging
import traceback
import os
import sys
import re
from dotenv import load_dotenv
from database_agent import process_chat_message  # Import from agent.py

# Fix for Windows asyncio subprocess issue
if sys.platform == 'win32':
    # Set the event loop policy to ProactorEventLoop for Windows
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)

# Set up logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Log environment variables and system info for debugging
logger.info(
    f"Environment variables: DB_HOST={os.getenv('DB_HOST')}, DB_USER={os.getenv('DB_USER')}, DB_NAME={os.getenv('DB_NAME')}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"PATH: {os.getenv('PATH')}")
logger.info(f"Platform: {sys.platform}")
logger.info(f"Event loop policy: {asyncio.get_event_loop_policy()}")

# Set page configuration
st.set_page_config(
    page_title="Medical Consultant Chat",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'input_key' not in st.session_state:
    st.session_state.input_key = 0


def format_assistant_response(response_text):
    """Format assistant response with enhanced styling for consultant lists"""

    # Check if response contains consultant information
    if "Found consultants:" in response_text:
        # Extract the consultant list
        consultant_match = re.search(r'Found consultants:\s*\[(.*?)\]', response_text)
        if consultant_match:
            consultant_data = consultant_match.group(1)

            # Parse individual consultants
            consultants = []
            consultant_entries = consultant_data.split(', Dr. ')

            for i, entry in enumerate(consultant_entries):
                if i == 0:
                    # First entry already has "Dr."
                    if entry.startswith('Dr. '):
                        entry = entry[4:]  # Remove "Dr. " prefix
                else:
                    # Add back "Dr." for subsequent entries
                    pass

                # Split name and specialty
                if ' - ' in entry:
                    name, specialty = entry.split(' - ', 1)
                    consultants.append({
                        'name': f"Dr. {name}" if not name.startswith('Dr.') else name,
                        'specialty': specialty
                    })

            # Display formatted consultant cards
            if consultants:
                st.markdown("### 🏥 **Found Medical Consultants**")

                # Create columns for better layout
                cols = st.columns(min(len(consultants), 2))

                for i, consultant in enumerate(consultants):
                    with cols[i % 2]:
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding: 20px;
                            border-radius: 15px;
                            margin: 10px 0;
                            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                            color: white;
                            border-left: 5px solid #4CAF50;
                        ">
                            <h4 style="margin: 0 0 10px 0; font-size: 1.2em;">
                                👨‍⚕️ {consultant['name']}
                            </h4>
                            <p style="margin: 0; font-size: 1em; opacity: 0.9;">
                                🏥 <strong>Specialty:</strong> {consultant['specialty']}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                # Add summary
                st.info(f"📋 **Total consultants found:** {len(consultants)}")

                # Add helpful note
                st.markdown("""
                ---
                💡 **Next Steps:**
                - Contact any of the specialists above for consultation
                - Prepare a list of your symptoms before the appointment
                - Bring any relevant medical records or test results
                """)
            else:
                st.warning("No consultants found in the response.")
        else:
            st.write(response_text)

    elif "No consultants found" in response_text:
        st.warning("🔍 " + response_text)
        st.markdown("""
        **Suggestions:**
        - Try describing your symptoms in different words
        - Be more specific about your condition
        - Ask about general practitioners or internal medicine doctors
        """)

    elif "?" in response_text:
        # This is a clarification question
        st.info("❓ " + response_text)
        st.markdown("*Please provide more details about your symptoms to get better recommendations.*")

    else:
        # Regular response
        st.write(response_text)


def run_async_task(coro):
    """
    Helper function to run async tasks in Streamlit
    Handles event loop creation properly for Windows
    """
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running (common in Streamlit),
            # we need to use asyncio.run in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            # If no loop is running, we can use the existing loop
            return loop.run_until_complete(coro)
    except RuntimeError:
        # If no event loop exists, create a new one
        return asyncio.run(coro)


async def process_user_message_async(user_message: str, conversation_history):
    """Async wrapper for processing user messages"""
    if not user_message.strip():
        return {"response": "Please enter a valid message.", "status": "error"}

    try:
        result = await asyncio.wait_for(
            process_chat_message(user_message, conversation_history),
            timeout=30.0
        )
        return result
    except asyncio.TimeoutError:
        return {
            "response": "Error: Database query timed out after 30 seconds. Please check if the database server is running.",
            "status": "error"
        }
    except Exception as e:
        return {
            "response": f"Error processing message: {str(e)}\nPlease check if 'uvx mcp-sql-server' is accessible and the database is running.",
            "status": "error"
        }


def main():
    """Main Streamlit chat application"""
    st.title("🏥 Medical Consultant Chat")
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, #4CAF50, #45a049);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
        text-align: center;
        font-size: 1.1em;
    ">
        💬 Chat with our AI assistant to find medical specialists based on your symptoms
    </div>
    """, unsafe_allow_html=True)

    # Quick examples section
    with st.expander("💡 **Example Questions You Can Ask**", expanded=False):
        st.markdown("""
        - *"I have chest pain and shortness of breath"*
        - *"Get a list of all consultants who treat unexplained weight loss, fatigue, lumps or thickening"*
        - *"I need help with headaches and dizziness"*
        - *"Show me oncologists and hematologists"*
        - *"I have digestive issues and stomach pain"*
        """)

    # Display system info for debugging
    with st.expander("🔧 System Information (for debugging)", expanded=False):
        st.write(f"**Platform:** {sys.platform}")
        st.write(f"**Event loop policy:** {type(asyncio.get_event_loop_policy()).__name__}")
        st.write(f"**Python version:** {sys.version}")
        st.write(f"**Database:** {os.getenv('DB_HOST', 'Not configured')}/{os.getenv('DB_NAME', 'Not configured')}")

    # Chat container
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.conversation_history:
            with st.chat_message("user"):
                st.write(msg["user"])
            with st.chat_message("assistant"):
                if msg["assistant"] is not None:
                    format_assistant_response(msg["assistant"])
                else:
                    # Show processing indicator for messages without responses yet
                    st.info("🔄 Processing your message...")

    # Input box
    user_input = st.chat_input(
        "Describe your symptoms or ask a question:",
        key=f"chat_input_{st.session_state.input_key}",
        disabled=st.session_state.processing
    )

    # Process input
    if user_input and not st.session_state.processing:
        # Immediately add user message to conversation history
        st.session_state.conversation_history.append({"user": user_input, "assistant": None})
        st.session_state.input_key += 1
        st.session_state.processing = True
        st.rerun()

    # Check if there's a message being processed
    if (st.session_state.processing and
            st.session_state.conversation_history and
            st.session_state.conversation_history[-1]["assistant"] is None):

        last_message = st.session_state.conversation_history[-1]
        try:
            result = run_async_task(process_user_message_async(
                last_message["user"],
                st.session_state.conversation_history[:-1]  # Exclude the current message being processed
            ))

            # Update the message with the response
            st.session_state.conversation_history[-1]["assistant"] = result["response"]
            st.session_state.processing = False
            st.rerun()

        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            st.session_state.conversation_history[-1]["assistant"] = error_msg
            st.session_state.processing = False
            logger.error(f"Error in main processing: {e}\n{traceback.format_exc()}")
            st.rerun()

    # Display processing status
    if st.session_state.processing:
        st.info("Processing your message...")

    # Clear conversation button
    if st.session_state.conversation_history:
        if st.button("Clear Conversation"):
            st.session_state.conversation_history = []
            st.session_state.input_key += 1
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="
        background: #313132;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #FF6B6B;
        margin-top: 20px;
    ">
        <strong>⚠️ Important Medical Disclaimer:</strong><br>
        This tool provides general guidance and is not a substitute for professional medical advice. 
        Always consult with qualified healthcare professionals for proper diagnosis and treatment.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()