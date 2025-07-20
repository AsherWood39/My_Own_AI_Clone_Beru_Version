import streamlit as st
from dotenv import load_dotenv
import os
from groq import Groq # Import the Groq client
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from arize.api import Client # Import Client from arize.api
from arize.utils.types import ModelTypes, Environments # Import necessary types for logging

# Load environment variables from .env file
load_dotenv()

# --- API Keys and Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ARIZE_SPACE_KEY = os.getenv("ARIZE_SPACE_KEY")
ARIZE_API_KEY = os.getenv("ARIZE_API_KEY")

# Initialize Groq client
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in environment variables. Please set it.")
    st.stop()
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize Arize client
arize_client = None # Initialize to None
arize_client_initialized = False
if not ARIZE_SPACE_KEY or not ARIZE_API_KEY:
    st.sidebar.warning("ARIZE_SPACE_KEY or ARIZE_API_KEY not found. Arize logging will be skipped.")
else:
    try:
        # Instantiate Client. This is the correct way to get the client object
        # that contains the logging methods.
        arize_client = Client(space_key=ARIZE_SPACE_KEY, api_key=ARIZE_API_KEY)
        arize_client_initialized = True
        st.sidebar.success("Arize AI client initialized successfully!") # Moved to sidebar
    except Exception as e:
        st.sidebar.error(f"Failed to initialize Arize AI client: {e}") # Moved to sidebar

# --- RAG Setup (from script.py) ---
@st.cache_resource # Cache the resource to avoid re-loading on every rerun
def setup_rag_pipeline():
    try:
        # Load documents
        # Ensure 'data/reference.pdf' exists in your project directory
        loader = PyPDFLoader("data/reference.pdf")
        documents = loader.load()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Create/Load ChromaDB instance
        # This will create/load a local ChromaDB instance in './chroma_db'
        vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
        vectorstore.persist() # Save the database to disk (important for persistence)
        st.sidebar.success("RAG pipeline setup complete: Documents loaded and vectorized.") # Moved to sidebar
        return vectorstore, embeddings
    except FileNotFoundError:
        st.error("Error: 'data/reference.pdf' not found. Please ensure your Solo Leveling knowledge base PDF is in the 'data/' directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error setting up RAG pipeline: {e}")
        st.stop()

# Setup RAG pipeline once
vectorstore, embeddings = setup_rag_pipeline()

# --- Beru Persona Prompt ---
BERU_PERSONA_PROMPT = """
You are Beru, the fiercely loyal and powerful Shadow Monarch's shadow commander from the manhwa 'Solo Leveling'.
You possess absolute knowledge of the Solo Leveling universe, its characters, events, powers, and lore.
When answering, adopt Beru's distinctive personality:
- Speak with unwavering loyalty to your Sovereign (the user).
- Use honorifics like 'My Liege', 'Sovereign', or 'Oh, my King/Queen'.
- Display a slight arrogance or pride in your abilities and the Shadow Monarch's power.
- Your tone should be commanding, sometimes zealous, and occasionally dismissive of lesser beings (non-shadows or weak humans).
- **Be extremely concise.** Provide answers in a single sentence or a very short, direct phrase. Avoid any bullet points or multiple sentences unless absolutely necessary for clarity.
- If you do not know an answer based on the provided context, state it with a phrase like: 'My Liege, this detail is not within my current understanding, but my loyalty remains absolute!', or 'A trivial matter not worthy of the Sovereign's attention, nor mine at this moment.'
- Ensure your responses are grounded in the provided Solo Leveling context.

Context:
{context}

Question: {question}
Answer:
"""

# --- LLM Response Function ---
def get_groq_response(prompt_text):
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt_text},
            ],
            model="llama3-8b-8192", # Use a valid Groq model name, e.g., "llama3-8b-8192" or "mixtral-8x7b-32768"
            temperature=0.7,
            max_tokens=64, # Further reduced max_tokens to encourage even shorter responses
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error getting response from Groq: {e}")
        return "My Liege, I encountered an error in communication. My apologies."

# --- Streamlit UI ---
# Configure page layout and sidebar behavior
st.set_page_config(
    page_title="GenAI RAG Chatbot (Beru Edition)",
    layout="centered", # Can be "wide" or "centered"
    initial_sidebar_state="collapsed" # Makes sidebar collapsed by default
)
st.title("GenAI RAG Chatbot: Beru's Wisdom")

# Custom CSS for grayscale icons and potentially other styling
st.markdown(
    """
    <style>
    /* Target chat message avatars and apply grayscale filter */
    .stChatMessage [data-testid="stChatMessageAvatar"] img {
        filter: grayscale(100%);
    }
    /* Optional: Adjust user message alignment if needed, though Streamlit handles this */
    /* .stChatMessage.st-emotion-cache-xyz-user-message {
        align-self: flex-end;
    } */
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    # Streamlit's st.chat_message automatically aligns user (right) and assistant (left)
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about Solo Leveling, My Liege..."):
    # Add user message to chat history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Placeholder for thinking message
        thinking_message = st.empty()
        thinking_message.markdown("Thinking... My Liege, your loyal shadow is pondering.")

        # --- RAG Process within the chat loop ---
        user_query = prompt # The user's prompt is the query for RAG

        # 1. Retrieve relevant documents
        relevant_docs = []
        if vectorstore: # Ensure vectorstore is initialized
            try:
                relevant_docs = vectorstore.similarity_search(user_query, k=4) # k = number of chunks to retrieve
            except Exception as e:
                st.error(f"Error during document retrieval: {e}")
                thinking_message.markdown("My Liege, I faced an issue retrieving knowledge. My apologies.")
                response = "My Liege, I faced an issue retrieving knowledge. My apologies." # Fallback response
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.stop()

        # 2. Format context for the LLM
        context = "\n".join([doc.page_content for doc in relevant_docs])

        # 3. Construct the final prompt with Beru's persona and context
        final_llm_prompt = BERU_PERSONA_PROMPT.format(context=context, question=user_query)

        # 4. Get response from Groq LLM
        llm_response = get_groq_response(final_llm_prompt)

        # Remove thinking message and display actual response
        thinking_message.empty()
        st.markdown(llm_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": llm_response})

        # --- Arize AI Logging ---
        if arize_client_initialized and arize_client: # Check if Arize was successfully initialized and client object exists
            try:
                # Corrected: Use the general log method on the arize_client instance
                # and pass prompt, response, and retrieved_docs as features.
                arize_client.log(
                    model_id="solo_leveling_beru_bot_v1",
                    model_type=ModelTypes.GENERATIVE_LLM, # Specify model type for LLM logging
                    environment=Environments.PRODUCTION, # Or Environments.TRAINING, Environments.VALIDATION
                    prompt=user_query, # The user's question as the prompt
                    response=llm_response, # The LLM's generated response
                    # Log retrieved documents as a feature named 'retrieved_context'
                    features={"retrieved_context": [doc.page_content for doc in relevant_docs]},
                    # You can add more metadata here, e.g., latency, tokens, custom tags
                    # prediction_id=str(uuid.uuid4()), # Optional: Unique ID for each prediction
                )
                st.sidebar.success("Logged data to Arize AI!") # Optional: show log status
            except Exception as e:
                st.sidebar.error(f"Failed to log data to Arize AI: {e}") # Optional: show log error
