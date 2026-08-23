import io
import os
from typing import List, Dict
import chromadb
from pypdf import PdfReader
from fastembed import TextEmbedding
from groq import Groq

class RAGEngine:
    def __init__(
        self, 
        groq_client: Groq,
        model_name: str = "openai/gpt-oss-20b",
        collection_name: str = "doc_knowledge_base"
    ):
        self.client = groq_client
        self.model_name = model_name
        
        # 1. Initialize FastEmbed model
        self.embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # 2. Local in-memory ChromaDB vector store
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract plain text from an uploaded PDF stream."""
        pdf_bytes = pdf_file.read()
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                text += f"\n--- Page {page_num + 1} ---\n" + page_text
        return text

    def chunk_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        """Split document into overlapping character chunks."""
        if not text or not text.strip():
            return []
            
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size - chunk_overlap
        return chunks

    def index_document(self, text: str, doc_name: str) -> int:
        """Generate embeddings and index chunks into ChromaDB."""
        chunks = self.chunk_text(text)
        if not chunks:
            return 0

        # Reset collection to avoid mixing documents
        try:
            self.chroma_client.delete_collection(name=self.collection.name)
        except Exception:
            pass
        self.collection = self.chroma_client.get_or_create_collection(name="doc_knowledge_base")

        # Compute embeddings with FastEmbed
        raw_embeddings = list(self.embedder.embed(chunks))
        embeddings = [e.tolist() for e in raw_embeddings]
        
        ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": doc_name, "chunk_id": i} for i in range(len(chunks))]

        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        return len(chunks)

    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict]:
        """Query ChromaDB for top-K semantically closest chunks."""
        count = self.collection.count()
        if count == 0:
            return []
            
        actual_k = min(top_k, count)
        raw_query_embed = list(self.embedder.embed([query]))[0]
        query_embedding = raw_query_embed.tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k
        )
        
        retrieved_docs = []
        if results and results.get("documents") and results["documents"][0]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                retrieved_docs.append({
                    "content": doc,
                    "source": meta["source"],
                    "chunk_id": meta["chunk_id"]
                })
        return retrieved_docs

    def answer_query(self, query: str, top_k: int = 3) -> Dict:
        """Retrieve relevant chunks and prompt Groq to generate a cited response."""
        retrieved_chunks = self.retrieve_context(query, top_k=top_k)
        
        if not retrieved_chunks:
            return {
                "answer": "No indexed content found. Please ensure the uploaded file contains selectable text and click 'Process & Index Document'.",
                "sources": []
            }

        context_str = "\n\n".join(
            [f"[Source: {c['source']} | Chunk #{c['chunk_id']}]:\n{c['content']}" for c in retrieved_chunks]
        )

        system_instruction = (
            "You are a strict, factual AI assistant. Answer the question using ONLY the provided context material. "
            "If the answer is not in the context, say 'I cannot find that in the provided document.' "
            "Always reference Chunk numbers when answering."
        )

        user_prompt = f"Context Material:\n{context_str}\n\nUser Question: {query}"

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": retrieved_chunks
        }