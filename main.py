import streamlit as st
from dotenv import load_dotenv
import os
from groq import Groq # Import the Groq client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS # Changed from Chroma to FAISS
from arize.api import Client # Import Client from arize.api
from arize.utils.types import ModelTypes, Environments # Import necessary types for logging
from langchain.docstore.document import Document # Import Document for creating LangChain docs
from langchain_community.document_loaders import PyPDFLoader # Re-added for local PDF loading
# Removed: import requests # No longer directly used for PDF loading, but may be used elsewhere
# Removed: import io # No longer directly used for PDF loading from URL
# Removed: from pypdf import PdfReader # No longer directly used for PDF loading from URL

# --- FIX for ChromaDB sqlite3 issue is REMOVED as we are now using FAISS ---

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

# --- RAG Setup (using FAISS) ---
@st.cache_resource # Cache the resource to avoid re-loading on every rerun
def setup_rag_pipeline():
    vectorstore_path = "./faiss_index" # Path to save/load FAISS index

    # Check if FAISS index already exists to avoid re-downloading/re-processing
    if os.path.exists(vectorstore_path):
        st.info("Loading existing FAISS index...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # Embeddings needed for loading
        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        st.sidebar.success("RAG pipeline setup complete: FAISS index loaded.")
        return vectorstore, embeddings
    else:
        try:
            # --- Solo Leveling PDF from Local File ---
            # Ensure 'data/reference.pdf' exists in your project directory
            pdf_path = "data/reference.pdf"
            st.info(f"Attempting to load PDF from: {pdf_path}")

            # Use PyPDFLoader for local PDF files
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            st.info(f"Loaded {len(documents)} pages from the PDF.")
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            st.info(f"Split {len(chunks)} chunks from the PDF.")

            # Initialize embeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            # Create FAISS index from documents
            vectorstore = FAISS.from_documents(chunks, embeddings)
            vectorstore.save_local(vectorstore_path) # Save the FAISS index to disk
            st.sidebar.success("RAG pipeline setup complete: Documents loaded, vectorized, and FAISS index saved.")
            return vectorstore, embeddings
        except FileNotFoundError:
            st.error(f"Error: '{pdf_path}' not found. Please ensure your Solo Leveling knowledge base PDF is in the 'data/' directory.")
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
            model="llama3-8b-8192",
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

    /* Adjust user message alignment to the right */
    .stChatMessage[data-testid="stChatMessage"]:has([data-testid="stMarkdownContainer"]:not([data-testId="stUserChatMessage"])) {
        display: flex;
        justify-content: flex-end;
    }
    .stChatMessage[data-testid="stChatMessage"]:has([data-testid="stMarkdownContainer"]:not([data-testId="stUserChatMessage"])) [data-testid="stChatMessageContent"] {
        background-color: #e0f7fa; /* Light blue background for user messages */
        border-radius: 10px;
        padding: 10px;
    }
    .stChatMessage[data-testid="stChatMessage"]:has([data-testid="stMarkdownContainer"]:not([data-testId="stUserChatMessage"])) [data-testid="stChatMessageAvatar"] {
        order: 2; /* Move avatar to the right of the message */
        margin-left: 10px; /* Add space between message and avatar */
    }

    /* Adjust assistant message alignment to the left */
    .stChatMessage[data-testid="stChatMessage"]:has([data-testId="stUserChatMessage"]) {
        display: flex;
        justify-content: flex-start;
    }
    .stChatMessage[data-testid="stChatMessage"]:has([data-testId="stUserChatMessage"]) [data-testid="stChatMessageContent"] {
        background-color: #f0f0f0; /* Light gray background for assistant messages */
        border-radius: 10px;
        padding: 10px;
    }
    .stChatMessage[data-testid="stChatMessage"]:has([data-testId="stUserChatMessage"]) [data-testid="stChatMessageAvatar"] {
        order: 1; /* Keep avatar to the left of the message */
        margin-right: 10px; /* Add space between message and avatar */
    }
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
