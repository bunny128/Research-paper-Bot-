import streamlit as st
import os
import time
from dotenv import load_dotenv

from modules.llm_setup import initialize_llm
from modules.file_handler import load_documents
from modules.vector_store import build_vectorstore
from modules.retriever_chain import build_conversational_rag_chain

# Load environment variables
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

st.title("🧠 HealthQ ChatBot")

api_key = os.getenv("GROQ_API_KEY")

if api_key:
    t0 = time.time()
    llm = initialize_llm()  # ✅ FIXED
    st.success(f"✅ LLM initialized in {time.time() - t0:.2f}s")

    uploaded_files = st.file_uploader(
        "📄 Upload PDF policy files",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        # Step 1: Load documents
        t1 = time.time()
        st.info("📚 Loading documents...")
        documents = load_documents(uploaded_files)
        st.success(f"✅ Loaded {len(documents)} documents in {time.time() - t1:.2f}s")

        # Step 2: Build vectorstore
        t2 = time.time()
        st.info("📦 Building vector store...")
        vectorstore = build_vectorstore(documents)
        st.success(f"✅ Vector store built in {time.time() - t2:.2f}s")

        # Step 3: Create RAG chain
        t3 = time.time()
        st.info("🔗 Creating RAG retrieval chain...")
        conversational_chain = build_conversational_rag_chain(
            llm,
            filter_metadata=None
        )
        st.success(f"✅ RAG chain initialized in {time.time() - t3:.2f}s")

        # Step 4: Question input
        user_input = st.text_input("💬 Ask a question about the policy")

        if user_input:
            t4 = time.time()
            st.info("🤖 Generating answer...")
            try:
                response = conversational_chain.invoke({"input": user_input})
                st.success(f"✅ Answered in {time.time() - t4:.2f}s")
                st.write("📝 Answer:", response["answer"])
            except Exception as e:
                st.error(f"❌ Error: {e}")
else:
    st.warning("❗ Missing GROQ API key or server error.")
