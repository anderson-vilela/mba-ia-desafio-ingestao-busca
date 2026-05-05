# Desafio MBA Engenharia de Software com IA - Full Cycle

Ingestão de um PDF para um banco vetorial (PostgreSQL + pgVector) e chat de
perguntas e respostas via CLI usando LangChain. As respostas são geradas
**exclusivamente** com base no conteúdo do PDF — quando a informação não está
no contexto recuperado, o sistema responde:

> Não tenho informações necessárias para responder sua pergunta.

## Requisitos

- Python 3.10+
- Docker e Docker Compose
- Uma chave de API válida da OpenAI **ou** do Google Gemini

## Estrutura

```
.
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── document.pdf            # PDF a ser ingerido
└── src/
    ├── ingest.py           # Lê o PDF, gera chunks/embeddings e popula o pgVector
    ├── search.py           # Monta a chain RAG (retriever + prompt + LLM)
    └── chat.py             # CLI interativo de perguntas e respostas
```

## Passo a passo

### 1. Clonar o repositório e entrar no diretório

```bash
git clone <url-do-fork>
cd mba-ia-desafio-ingestao-busca
```

### 2. Criar e ativar o ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Configurar as variáveis de ambiente

Copie o arquivo de exemplo e preencha as variáveis:

```bash
cp .env.example .env
```

Edite o `.env`:

- `LLM_PROVIDER` — `openai` (padrão) ou `gemini`.
- `OPENAI_API_KEY` — obrigatória se `LLM_PROVIDER=openai`.
- `GOOGLE_API_KEY` — obrigatória se `LLM_PROVIDER=gemini`.
- `DATABASE_URL` — já vem apontando para o banco do `docker-compose.yml`
  (`postgresql+psycopg://postgres:postgres@localhost:5432/rag`).
- `PG_VECTOR_COLLECTION_NAME` — nome da coleção (padrão `rag_documents`).
- `PDF_PATH` — caminho do PDF (padrão `document.pdf` na raiz).

> Observação: ao trocar o provedor (e portanto o modelo de embeddings), use
> uma `PG_VECTOR_COLLECTION_NAME` diferente, pois as dimensões dos vetores
> de OpenAI e Gemini não são compatíveis.

### 5. Subir o PostgreSQL com pgVector

```bash
docker compose up -d
```

O serviço `bootstrap_vector_ext` cria automaticamente a extensão `vector` no
banco `rag`. Aguarde alguns segundos até o healthcheck do Postgres ficar OK.

### 6. Executar a ingestão do PDF

```bash
python src/ingest.py
```

Esse comando:

1. Lê o PDF em `PDF_PATH` com `PyPDFLoader`.
2. Divide o texto em chunks de **1000 caracteres** com **overlap de 150**
   usando `RecursiveCharacterTextSplitter`.
3. Gera embeddings com o provedor configurado.
4. Persiste os vetores no PostgreSQL via `langchain_postgres.PGVector`.

Saída esperada (resumida):

```
Carregando PDF: .../document.pdf
Gerando embeddings e gravando N chunks no PGVector...
Ingestão concluída: N chunks armazenados na coleção 'rag_documents'.
```

### 7. Conversar com o PDF

```bash
python src/chat.py
```

Exemplo de uso:

```
Faça sua pergunta:

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

Faça sua pergunta:

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

Para encerrar o chat, digite `sair` (ou pressione `Ctrl+C`).

## Como funciona a busca

`src/search.py` constrói uma chain LCEL que, para cada pergunta:

1. Vetoriza a pergunta com o mesmo modelo de embeddings da ingestão.
2. Recupera os `k=10` chunks mais similares do pgVector via
   `similarity_search_with_score`.
3. Concatena o conteúdo desses chunks no placeholder `{contexto}` do prompt.
4. Envia o prompt completo para a LLM, que deve responder somente com base
   no contexto — caso contrário, retorna a frase padrão de "fora do contexto".

## Resolução de problemas

- **`OPENAI_API_KEY não configurada`** / **`GOOGLE_API_KEY não configurada`**:
  preencha a chave correspondente no `.env`.
- **`DATABASE_URL não configurada`**: confira o `.env` — deve apontar para o
  banco `rag` exposto pelo Docker Compose.
- **Erro de conexão com o Postgres**: verifique se os contêineres estão de pé
  com `docker compose ps`.
- **Respostas incoerentes ou vazias**: confirme que a ingestão foi executada
  com sucesso e que `LLM_PROVIDER`, `*_EMBEDDING_MODEL` e
  `PG_VECTOR_COLLECTION_NAME` são consistentes entre `ingest.py` e `chat.py`.
