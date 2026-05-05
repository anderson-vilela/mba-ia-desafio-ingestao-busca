import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_postgres import PGVector

from ingest import get_embeddings

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "rag_documents")
SEARCH_K = int(os.getenv("SEARCH_K", "10"))

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError(
                "OPENAI_API_KEY não configurada. Defina-a no .env."
            )
        model = os.getenv("OPENAI_LLM_MODEL", "gpt-5-nano")
        kwargs = {"model": model}
        if not model.startswith("gpt-5"):
            kwargs["temperature"] = 0
        return ChatOpenAI(**kwargs)

    if provider in {"gemini", "google"}:
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not os.getenv("GOOGLE_API_KEY"):
            raise EnvironmentError(
                "GOOGLE_API_KEY não configurada. Defina-a no .env."
            )
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash-lite"),
            temperature=0,
        )

    raise ValueError(
        f"LLM_PROVIDER inválido: '{provider}'. Use 'openai' ou 'gemini'."
    )


def _format_context(question: str, store: PGVector) -> str:
    results = store.similarity_search_with_score(question, k=SEARCH_K)
    if not results:
        return ""
    return "\n\n".join(doc.page_content.strip() for doc, _ in results)


def _build_chain():
    if not DATABASE_URL:
        raise EnvironmentError(
            "DATABASE_URL não configurada. Defina-a no .env."
        )

    store = PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = get_llm()

    return (
        {
            "contexto": RunnableLambda(lambda q: _format_context(q, store)),
            "pergunta": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def search_prompt(question=None):
    chain = _build_chain()
    if question is None:
        return chain
    return chain.invoke(question)
