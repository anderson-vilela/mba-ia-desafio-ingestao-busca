import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "rag_documents")


def get_embeddings():
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError(
                "OPENAI_API_KEY não configurada. Defina-a no .env."
            )
        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        )

    if provider in {"gemini", "google"}:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        if not os.getenv("GOOGLE_API_KEY"):
            raise EnvironmentError(
                "GOOGLE_API_KEY não configurada. Defina-a no .env."
            )
        return GoogleGenerativeAIEmbeddings(
            model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001"),
        )

    raise ValueError(
        f"LLM_PROVIDER inválido: '{provider}'. Use 'openai' ou 'gemini'."
    )


def _resolve_pdf_path() -> Path:
    candidate = Path(PDF_PATH).expanduser()
    if not candidate.is_absolute():
        candidate = (Path(__file__).resolve().parent.parent / candidate).resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            f"Arquivo PDF não encontrado em: {candidate}. "
            "Ajuste a variável PDF_PATH no .env."
        )
    return candidate


def ingest_pdf() -> None:
    if not DATABASE_URL:
        raise EnvironmentError(
            "DATABASE_URL não configurada. Defina-a no .env."
        )

    pdf_path = _resolve_pdf_path()
    print(f"Carregando PDF: {pdf_path}")
    docs = PyPDFLoader(str(pdf_path)).load()
    if not docs:
        raise RuntimeError("Não foi possível extrair conteúdo do PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=False,
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise RuntimeError("Nenhum chunk foi gerado a partir do PDF.")

    enriched = [
        Document(
            page_content=chunk.page_content,
            metadata={k: v for k, v in chunk.metadata.items() if v not in ("", None)},
        )
        for chunk in chunks
    ]
    ids = [f"doc-{i}" for i in range(len(enriched))]

    print(f"Gerando embeddings e gravando {len(enriched)} chunks no PGVector...")
    store = PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )
    store.add_documents(documents=enriched, ids=ids)

    print(
        f"Ingestão concluída: {len(enriched)} chunks armazenados na coleção "
        f"'{COLLECTION_NAME}'."
    )


if __name__ == "__main__":
    ingest_pdf()
