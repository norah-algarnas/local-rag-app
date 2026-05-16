import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

st.title("📚 Local RAG Assistant")
st.write("Upload PDF files and ask questions about them.")

uploaded_files = st.file_uploader(
    "Upload your PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    generator = pipeline("text-generation", model="distilgpt2")
    return embedding_model, generator

embedding_model, generator = load_models()

def read_pdfs(files):
    documents = []

    for file in files:
        reader = PdfReader(file)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        documents.append({
            "name": file.name,
            "text": text
        })

    return documents

def chunk_text(text, chunk_size=800, overlap=80):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

if uploaded_files:
    documents = read_pdfs(uploaded_files)

    chunks = []
    sources = []

    for doc in documents:
        doc_chunks = chunk_text(doc["text"])

        for i, chunk in enumerate(doc_chunks):
            chunks.append(chunk)
            sources.append({
                "file": doc["name"],
                "chunk_index": i
            })

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    st.success(f"Loaded {len(documents)} files and created {len(chunks)} chunks.")

    question = st.text_input("Ask a question:")

    if question:
        query_embedding = embedding_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores, indices = index.search(query_embedding, 3)

        retrieved_chunks = []
        retrieved_sources = []

        for idx in indices[0]:
            retrieved_chunks.append(chunks[idx])
            retrieved_sources.append(sources[idx])

        context = "\n\n".join(retrieved_chunks)

        prompt = f"""
Context:
{context}

Question:
{question}

Short Answer:
"""

        response = generator(
            prompt,
max_new_tokens=80,
do_sample=False,
repetition_penalty=1.5
        )[0]["generated_text"]

        answer = response.split("Short Answer:")[-1].strip()

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for source in retrieved_sources:
            st.write(f"- {source['file']} | chunk {source['chunk_index']}")

else:
    st.info("Please upload PDF files to start.")