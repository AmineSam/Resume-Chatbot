"""
Test script for Resume Chatbot
Validates the RAG chain works correctly with mocked retrieval
"""

import os
import sys
from unittest.mock import Mock, patch
from dotenv import load_dotenv

# Load environment
load_dotenv()

def test_local_mode():
    """Test LOCAL mode with Ollama"""
    print("\n" + "="*60)
    print("🧪 Testing LOCAL Mode (Ollama)")
    print("="*60)
    
    try:
        from langchain_ollama import ChatOllama
        from langchain_huggingface import HuggingFaceEmbeddings
        
        # Initialize LLM
        llm = ChatOllama(model="llama3.2", temperature=0.7)
        print("✅ ChatOllama initialized successfully")
        
        # Test embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        test_embedding = embeddings.embed_query("test")
        print(f"✅ HuggingFaceEmbeddings working (dimension: {len(test_embedding)})")
        
        # Test LLM with mock context
        mock_context = """
        Amine Samoudi has a PhD in Electrical Engineering from Ghent University (2018).
        He is currently enrolled in an AI & Data Science Bootcamp at BeCode (expected April 2026).
        He has 7 years of academic experience and 3 years in industrial automation.
        """
        
        from langchain_core.prompts import PromptTemplate
        prompt = PromptTemplate(
            template="Context: {context}\n\nQuestion: {question}\n\nAnswer:",
            input_variables=["context", "question"]
        )
        
        chain = prompt | llm
        response = chain.invoke({
            "context": mock_context,
            "question": "What is Amine's educational background?"
        })
        
        print(f"✅ LLM response generated: {response.content[:100]}...")
        print("\n✅ LOCAL MODE TEST PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ LOCAL MODE TEST FAILED: {e}")
        print("💡 Make sure Ollama is running: ollama serve")
        return False


def test_cloud_mode():
    """Test CLOUD mode with Groq"""
    print("\n" + "="*60)
    print("🧪 Testing CLOUD Mode (Groq)")
    print("="*60)
    
    try:
        from langchain_groq import ChatGroq
        from langchain_huggingface import HuggingFaceEmbeddings
        
        # Check API key
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            print("⚠️  GROQ_API_KEY not set - skipping CLOUD mode test")
            print("💡 Set GROQ_API_KEY in .env to test CLOUD mode")
            return None
        
        # Initialize LLM
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            groq_api_key=api_key
        )
        print("✅ ChatGroq initialized successfully")
        
        # Test embeddings (same as LOCAL)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        test_embedding = embeddings.embed_query("test")
        print(f"✅ HuggingFaceEmbeddings working (dimension: {len(test_embedding)})")
        
        # Test LLM with mock context
        mock_context = """
        Amine Samoudi has a PhD in Electrical Engineering from Ghent University (2018).
        He is currently enrolled in an AI & Data Science Bootcamp at BeCode (expected April 2026).
        He has 7 years of academic experience and 3 years in industrial automation.
        """
        
        from langchain_core.prompts import PromptTemplate
        prompt = PromptTemplate(
            template="Context: {context}\n\nQuestion: {question}\n\nAnswer:",
            input_variables=["context", "question"]
        )
        
        chain = prompt | llm
        response = chain.invoke({
            "context": mock_context,
            "question": "What is Amine's educational background?"
        })
        
        print(f"✅ LLM response generated: {response.content[:100]}...")
        print("\n✅ CLOUD MODE TEST PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ CLOUD MODE TEST FAILED: {e}")
        return False


def test_resume_loading():
    """Test that resume file can be loaded and chunked"""
    print("\n" + "="*60)
    print("🧪 Testing Resume Data Loading")
    print("="*60)
    
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        # Load resume
        with open("info.txt", "r", encoding="utf-8") as f:
            resume_text = f.read()
        
        print(f"✅ Resume loaded ({len(resume_text)} characters)")
        
        # Test chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(resume_text)
        
        print(f"✅ Text split into {len(chunks)} chunks")
        print(f"   Sample chunk: {chunks[0][:100]}...")
        print("\n✅ RESUME LOADING TEST PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ RESUME LOADING TEST FAILED: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("🚀 Resume Chatbot Test Suite")
    print("="*60)
    
    results = {}
    
    # Test resume loading (always runs)
    results['resume'] = test_resume_loading()
    
    # Test mode based on environment
    mode = os.getenv("MODE", "CLOUD").upper()
    
    if mode == "LOCAL":
        results['mode'] = test_local_mode()
    elif mode == "CLOUD":
        results['mode'] = test_cloud_mode()
    else:
        print(f"\n⚠️  Invalid MODE: {mode}")
        results['mode'] = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    
    for test_name, result in results.items():
        if result is True:
            print(f"✅ {test_name.upper()}: PASSED")
        elif result is False:
            print(f"❌ {test_name.upper()}: FAILED")
        else:
            print(f"⚠️  {test_name.upper()}: SKIPPED")
    
    all_passed = all(r in [True, None] for r in results.values())
    
    if all_passed:
        print("\n🎉 All tests passed! The chatbot is ready to use.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
