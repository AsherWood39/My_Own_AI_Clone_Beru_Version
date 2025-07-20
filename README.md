# 🕶️ Ant King — Beru Clone
*"At your command, My Liege. This unit holds the knowledge of great battles, the Sovereign's will, and the world beyond."* — Beru

---

## 🔮 Features
- **🧠 Context-Aware Retrieval (RAG)**  
  Uses FAISS vector search on a local *Solo Leveling* PDF, chunked and embedded with `all-MiniLM-L6-v2`.

- **🗣️ Persona-Driven Prompt Engineering**  
  Custom prompts enforce Beru’s loyal, concise, and proud tone, speaking as the Shadow Monarch’s commander.

- **📚 Vector-Embedded Lorebase**  
  Indexed local knowledge base enabling fast, relevant context retrieval for accurate responses.

---

## ⚙️ System Overview
- **Document Processing:** Parses and chunks PDF using `PyPDFLoader`.  
- **Embeddings & Vector DB:** Creates embeddings and stores them in FAISS for efficient similarity search.  
- **LLM Interaction:** Sends combined prompt and retrieved context to a Groq-hosted LLM.  
- **Evaluation:** Logs interactions to Arize AI for quality and performance monitoring.  
- **Deployment:** Streamlit app with a chat interface styled to reflect Beru’s character.

---

## 🚀 Setup
1. Clone this repository.  
2. Create a `.env` file with the following keys:
    ```
    GROQ_API_KEY=your_groq_api_key
    ARIZE_SPACE_KEY=your_arize_space_key
    ARIZE_API_KEY=your_arize_api_key
    ```
3. Place the Solo Leveling PDF (`reference.pdf`) inside the `data/` folder.  
4. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
5. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
    
---

## 📌 Notes
- Responses are concise and persona-consistent.  
- FAISS index is cached locally for performance.  
- Arize AI integration is optional but recommended for monitoring.

---

## 🔗 Live Demo
Try it here: https://myownaicloneberuversion-eclqg6yzvrsrtfrkxtmbmn.streamlit.app/

---

## ⚔️ About
A GenAI chatbot combining Retrieval-Augmented Generation, prompt engineering, vector search, and evaluation to bring Beru’s character to life.
