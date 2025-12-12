# 💼 Resume Chatbot

A RAG-based (Retrieval-Augmented Generation) chatbot that answers questions about Amine Samoudi's professional experience using natural language.

## ✨ Features

- **Dual-Mode Support**: Run locally with Ollama or in the cloud with Groq
- **Intelligent Retrieval**: Uses FAISS vector store with HuggingFaceEmbeddings
- **Synthesis-Focused**: Generates conversational answers, not copy-paste chunks
- **Chat Memory**: Maintains conversation context for follow-up questions
- **Clean UI**: Streamlit-based interface with chat history

## 🏗️ Architecture

```
┌─────────────────┐
│   Streamlit UI  │
└────────┬────────┘
         │
    ┌────▼─────┐
    │   LLM    │  ◄── MODE=LOCAL → Ollama (llama3.2)
    │ Provider │  ◄── MODE=CLOUD → Groq (llama-3.1-70b)
    └────┬─────┘
         │
┌────────▼────────────┐
│  LangChain RAG      │
│  - Retriever (FAISS)│
│  - Memory           │
│  - Custom Prompt    │
└────────┬────────────┘
         │
┌────────▼────────────┐
│ HuggingFace         │
│ Embeddings          │
│ (all-MiniLM-L6-v2)  │
└─────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- For LOCAL mode: [Ollama](https://ollama.ai) installed
- For CLOUD mode: [Groq API key](https://console.groq.com)

### Installation

1. **Clone and navigate to the repository**
```bash
cd Resume-Chatbot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
```bash
# Copy the example file
cp .env.example .env

# Edit .env with your preferred settings
```

### Running the Chatbot

#### Option 1: LOCAL Mode (Ollama)

1. **Install Ollama** from [ollama.ai](https://ollama.ai)

2. **Pull the llama3.2 model**
```bash
ollama pull llama3.2
```

3. **Start Ollama server** (if not running)
```bash
ollama serve
```

4. **Set MODE in .env**
```env
MODE=LOCAL
```

5. **Run the app**
```bash
streamlit run app.py
```

#### Option 2: CLOUD Mode (Groq)

1. **Get a free Groq API key** from [console.groq.com](https://console.groq.com)

2. **Set environment variables in .env**
```env
MODE=CLOUD
GROQ_API_KEY=your_actual_api_key_here
```

3. **Run the app**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🧪 Testing

Run the test script to validate your setup:

```bash
python test_bot.py
```

This will:
- ✅ Verify resume data loading
- ✅ Test embeddings generation
- ✅ Validate LLM connectivity
- ✅ Check the RAG chain logic

## 📁 Project Structure

```
Resume-Chatbot/
├── app.py              # Main Streamlit application
├── info.txt            # Resume data (source document)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .env               # Your local config (git-ignored)
├── test_bot.py        # Test suite
└── README.md          # This file
```

## 💡 Example Questions

Try asking:
- "What is Amine's educational background?"
- "Tell me about the Immo-Eliza project"
- "What technologies does he work with?"
- "What was his role at Unilin?"
- "What certifications does he have?"

## 🔧 Troubleshooting

### LOCAL Mode Issues

**Error: "Ollama connection failed"**
- Make sure Ollama is running: `ollama serve`
- Verify the model is pulled: `ollama pull llama3.2`
- Check Ollama is accessible at `http://localhost:11434`

### CLOUD Mode Issues

**Error: "GROQ_API_KEY not found"**
- Ensure `.env` file exists and contains your API key
- Verify the key is valid at [console.groq.com](https://console.groq.com)

### General Issues

**Error: "Resume file 'info.txt' not found"**
- Ensure `info.txt` is in the same directory as `app.py`

**Slow first response**
- First run downloads the embedding model (~90MB)
- Subsequent runs use cached embeddings

## 🌐 Deployment

### Streamlit Cloud (Recommended)

For detailed deployment instructions with **security best practices**, see:

📖 **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Complete guide for deploying to Streamlit Cloud

**Quick Summary**:
1. Add `info.txt` to `.gitignore` (keep your resume private)
2. Push code to public GitHub repository
3. Deploy on [share.streamlit.io](https://share.streamlit.io/)
4. Add resume content to **Streamlit Secrets** (encrypted storage)
5. App loads from secrets in cloud, from file locally

### Alternative: Render

For Render deployment:

1. **Create a `render.yaml`**
```yaml
services:
  - type: web
    name: resume-chatbot
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port $PORT
    envVars:
      - key: MODE
        value: CLOUD
      - key: GROQ_API_KEY
        sync: false  # Set in Render dashboard
```

2. **Set environment variables in Render dashboard**
   - `MODE=CLOUD`
   - `GROQ_API_KEY=your_key`

3. **Deploy** via Render's GitHub integration

## 🔒 Security Note

**Important**: `info.txt` contains personal information and should **NOT** be committed to GitHub.

- ✅ `info.txt` is in `.gitignore`
- ✅ Use `info.txt.example` as a template
- ✅ For deployment, use Streamlit Secrets or environment variables
- ✅ Never commit API keys or personal data

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for complete security setup.

## 📝 License

MIT License - feel free to use this for your own resume chatbot!

## 🤝 Contributing

This is a personal project, but suggestions are welcome via issues.

---

**Built with**: LangChain • Streamlit • HuggingFace • FAISS • Ollama/Groq
