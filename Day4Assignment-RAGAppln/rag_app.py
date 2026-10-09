import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder

from dotenv import load_dotenv

load_dotenv()

# 1. Page Configuration
st.set_page_config(
    page_title="Multi-PDF RAG App",
    page_icon="📚",
    layout="wide"
)
st.title("📚 Multi-PDF RAG Explorer — BM25 + Reranking")


# 2. Check API Key
if not os.environ.get("OPENAI_API_KEY"):
    st.error(
        "OPENAI_API_KEY environment variable not found. "
        "Set it before running Streamlit."
    )
    st.stop()


# 3. Initialize Session State
if "bm25_retriever" not in st.session_state:
    st.session_state.bm25_retriever = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# 4. Load the Cross-Encoder Once
@st.cache_resource
def load_reranker():
    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


reranker = load_reranker()


# 5. Sidebar: Upload and Index PDFs
with st.sidebar:
    st.header("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload multiple PDFs at once",
        type=["pdf"],
        accept_multiple_files=True
    )

    process_button = st.button(
        "Index Documents",
        type="primary",
        disabled=not uploaded_files
    )

    if process_button and uploaded_files:
        all_docs = []

        with st.spinner("Loading and chunking PDF documents..."):
            for uploaded_file in uploaded_files:
                tmp_path = None

                try:
                    # Save uploaded PDF temporarily
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    # Load pages
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()

                    # Preserve original filename and 1-based page numbers
                    for doc in docs:
                        doc.metadata["source"] = uploaded_file.name
                        doc.metadata["page"] = (
                            doc.metadata.get("page", 0) + 1
                        )

                    all_docs.extend(docs)

                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            # Split pages into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            split_docs = text_splitter.split_documents(all_docs)

            if not split_docs:
                st.error("No readable text was found in the uploaded PDFs.")
                st.stop()

            # Build the BM25 keyword-search index
            bm25_retriever = BM25Retriever.from_documents(split_docs)

            # Retrieve up to 15 candidates before reranking
            bm25_retriever.k = min(15, len(split_docs))

            st.session_state.bm25_retriever = bm25_retriever

            st.success(
                f"Indexed {len(split_docs)} chunks "
                f"from {len(uploaded_files)} PDF files."
            )


# 6. Question and Answer Interface
st.subheader("Ask a Question Across Documents")

if st.session_state.bm25_retriever is None:
    st.info("Upload and index PDFs in the sidebar to get started.")

else:
    user_query = st.chat_input(
        "What would you like to know from your documents?"
    )

    system_prompt = (
        "You are an expert research assistant answering questions "
        "based strictly on the provided context.\n\n"
        "CRITICAL RULES:\n"
        "1. If different files disagree, explicitly identify the "
        "contradiction and explain what each source says. Mention "
        "the conflicting filenames.\n"
        "2. Do not invent facts or use knowledge outside the context.\n"
        "3. If the answer is not available in the context, say: "
        "'I cannot find the answer in the provided documents.'\n"
        "4. When possible, refer to the source filename and page number.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    question_answer_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    # 7. BM25 Retrieval + Cross-Encoder Reranking
    if user_query:
        with st.spinner(
            "Retrieving candidates, reranking, and generating answer..."
        ):
            # Stage 1: Retrieve keyword-relevant candidates
            candidates = st.session_state.bm25_retriever.invoke(
                user_query
            )

            if not candidates:
                st.warning(
                    "No relevant text was found in the indexed documents."
                )
                st.stop()

            # Stage 2: Score query-document pairs using the cross-encoder
            pairs = [
                (user_query, doc.page_content)
                for doc in candidates
            ]

            scores = reranker.predict(pairs)

            # Stage 3: Sort by relevance score, highest first
            ranked_results = sorted(
                zip(candidates, scores),
                key=lambda item: float(item[1]),
                reverse=True
            )

            # Stage 4: Keep the best four chunks
            top_documents = [
                doc for doc, score in ranked_results[:4]
            ]

            # Stage 5: Generate the grounded answer
            response = question_answer_chain.invoke({
                "input": user_query,
                "context": top_documents
            })

            answer = response

            st.markdown("### Answer")
            st.write(answer)

            # 8. Display Source Citations
            if top_documents:
                st.markdown("### 📌 Sources Cited")

                seen_citations = set()

                for doc in top_documents:
                    file_name = doc.metadata.get(
                        "source", "Unknown File"
                    )
                    page_num = doc.metadata.get(
                        "page", "Unknown Page"
                    )

                    citation = (file_name, page_num)

                    if citation not in seen_citations:
                        seen_citations.add(citation)
                        st.markdown(
                            f"- 📄 `{file_name}` — Page {page_num}"
                        )

                # Optional: inspect reranking scores
                with st.expander("View reranking results"):
                    for rank, (doc, score) in enumerate(
                        ranked_results[:4], start=1
                    ):
                        st.markdown(
                            f"**Rank {rank}** | "
                            f"Score: {float(score):.4f} | "
                            f"Source: {doc.metadata.get('source', 'Unknown')} "
                            f"| Page: {doc.metadata.get('page', 'Unknown')}"
                        )
                        st.caption(
                            doc.page_content[:400]
                        )