"""
Resume Chatbot - RAG-based Q&A system for professional experience
Supports dual-mode: LOCAL (Ollama) or CLOUD (Groq)
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# Configuration
MODE = os.getenv("MODE", "CLOUD").upper()
RESUME_FILE = "info.txt"

# System prompt for synthesis-focused responses
SYSTEM_PROMPT = """You are a professional career assistant helping to answer questions about Amine Samoudi's professional background.

Use the following context to answer the question. IMPORTANT: Do not copy-paste text chunks directly. Instead:
1. Synthesize information from multiple parts of the context if relevant
2. Provide clear, conversational answers
3. Use specific details (dates, companies, technologies) when appropriate
4. If the information isn't in the context, politely say you don't have that information
5. **PRIORITY**: When asked about data science projects, machine learning projects, or "biggest project", prioritize information from the "FEATURED DATA SCIENCE PROJECT" section (Immo-Eliza) over other work experience

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer (synthesized, not copy-pasted):"""


@st.cache_resource
def initialize_llm():
    """Initialize LLM based on MODE environment variable"""
    if MODE == "LOCAL":
        try:
            from langchain_ollama import ChatOllama
            llm = ChatOllama(
                model="llama3.2",
                temperature=0.7,
            )
            st.sidebar.success("🟢 Using LOCAL mode (Ollama llama3.2)")
            return llm
        except Exception as e:
            st.sidebar.error(f"❌ Ollama connection failed: {e}")
            st.sidebar.info("💡 Make sure Ollama is running: `ollama serve`")
            st.stop()
    
    elif MODE == "CLOUD":
        try:
            from langchain_groq import ChatGroq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                st.sidebar.error("❌ GROQ_API_KEY not found in environment")
                st.sidebar.info("💡 Add your API key to .env file")
                st.stop()
            
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                groq_api_key=api_key
            )
            st.sidebar.success("🟢 Using CLOUD mode (Groq)")
            return llm
        except Exception as e:
            st.sidebar.error(f"❌ Groq initialization failed: {e}")
            st.stop()
    
    else:
        st.sidebar.error(f"❌ Invalid MODE: {MODE}. Use 'LOCAL' or 'CLOUD'")
        st.stop()


@st.cache_resource
def load_resume_data():
    """Load and process resume data into vector store"""
    try:
        # Try to load from Streamlit secrets (for cloud deployment)
        if hasattr(st, 'secrets') and 'RESUME_CONTENT' in st.secrets:
            resume_text = st.secrets['RESUME_CONTENT']
            st.sidebar.info("📄 Loaded resume from Streamlit secrets")
        # Fallback to local file (for development)
        elif os.path.exists(RESUME_FILE):
            with open(RESUME_FILE, "r", encoding="utf-8") as f:
                resume_text = f.read()
            st.sidebar.info(f"📄 Loaded resume from {RESUME_FILE}")
        else:
            st.error(
                "❌ Resume data not found!\n\n"
                "**For local development**: Create `info.txt` file\n\n"
                "**For Streamlit Cloud**: Add `RESUME_CONTENT` to secrets"
            )
            st.stop()
        
        # Split into chunks (larger chunks to keep project descriptions together)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(resume_text)
        
        # Create embeddings (same for both modes)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Create vector store
        vectorstore = FAISS.from_texts(chunks, embeddings)
        
        st.sidebar.success(f"✅ Processed {len(chunks)} chunks")
        return vectorstore
    
    except Exception as e:
        st.error(f"❌ Error loading resume: {e}")
        st.stop()


def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)


def format_chat_history(messages):
    """Format chat history for the prompt"""
    if not messages:
        return "No previous conversation."
    
    history = []
    for msg in messages[-6:]:  # Last 3 exchanges (6 messages)
        role = "Human" if msg["role"] == "user" else "Assistant"
        history.append(f"{role}: {msg['content']}")
    
    return "\n".join(history)


def create_rag_chain(llm, vectorstore):
    """Create a simple RAG chain using LCEL (LangChain Expression Language)"""
    
    # Create retriever (retrieve more chunks for better context)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    
    # Create chain using LCEL
    chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "chat_history": lambda x: x["chat_history"],
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


def main():
    st.set_page_config(
        page_title="Resume Chatbot - Amine Samoudi",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        /* Main title styling */
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        /* Subtitle styling */
        .subtitle {
            font-size: 1.1rem;
            color: #666;
            margin-bottom: 2rem;
        }
        
        /* Chat message styling */
        .stChatMessage {
            border-radius: 15px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
        }
        
        /* Button styling */
        .stButton>button {
            border-radius: 20px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title with custom styling
    st.markdown('<h1 class="main-title">🤖 AI Resume Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Ask me anything about <strong>Amine Samoudi\'s</strong> professional experience!</p>', unsafe_allow_html=True)
    
    # Initialize components
    llm = initialize_llm()
    vectorstore = load_resume_data()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "👋 Hello! I'm Amine's AI Resume Assistant. I can help you learn about his:\n\n"
                           "• 🎓 Educational background\n"
                           "• 💼 Professional experience\n"
                           "• 🛠️ Technical skills and certifications\n\n"
                           "What would you like to know?"
            }
        ]
    
    if "chain" not in st.session_state:
        st.session_state.chain = create_rag_chain(llm, vectorstore)
    
    # Display chat history
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("💬 Ask about experience, skills, projects..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    # Format chat history
                    chat_history = format_chat_history(st.session_state.messages[:-1])
                    
                    # Invoke chain
                    answer = st.session_state.chain.invoke({
                        "question": prompt,
                        "chat_history": chat_history
                    })
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_msg = f"❌ Error generating response: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Sidebar info
    with st.sidebar:
        # Mode indicator at the top
        mode_emoji = "🏠" if MODE == "LOCAL" else "☁️"
        mode_color = "#28a745" if MODE == "LOCAL" else "#007bff"
        st.markdown(
            f'<div style="background-color: {mode_color}; padding: 10px; border-radius: 10px; '
            f'text-align: center; color: white; font-weight: bold; margin-bottom: 1rem;">'
            f'{mode_emoji} {MODE} MODE</div>',
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        st.markdown("""
        - 🎓 What is Amine's educational background?
        - 🏗️ Tell me about the Immo-Eliza project
        - 💻 What programming languages does he know?
        - 🏢 What was his role at Unilin?
        - 📜 What certifications does he have?
        - 🚀 What is his biggest data science project?
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info(
            "This AI assistant uses **RAG (Retrieval-Augmented Generation)** "
            "to answer questions about Amine's professional background. "
            "Powered by LangChain and Groq/Ollama."
        )
        
        if st.button("🔄 Clear Chat History", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": "Chat cleared! What would you like to know?"}
            ]
            st.rerun()


if __name__ == "__main__":
    main()
