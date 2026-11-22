#!/usr/bin/env python3
"""
Mini-Lab RAG: Azure OpenAI + Azure AI Search
Tema: Cuidados Paliativos e Geriatria

Arquitetura:
- 1 recurso Azure OpenAI (East US 2): embeddings + geração de texto
- 1 recurso Azure AI Search (East US): busca vetorial (motivo: indisponibilidade de recurso free)
- Segurança: variáveis de ambiente via .env (nunca keys no código)
- Padrões: SOLID, try-except robusto, injeção de dependência
"""

import json
import os
import logging
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from openai import AzureOpenAI, APIError, APIConnectionError, RateLimitError
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError

# ============================================================================
# CONFIGURAÇÃO DE LOGS
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ============================================================================
# CARREGAMENTO DE VARIÁVEIS DE AMBIENTE
# ============================================================================

load_dotenv()

def _get_env(key: str) -> str:
    """
    Lê variável de ambiente obrigatória.
    Lança erro claro se não encontrada — fail fast pattern.
    """
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Variável de ambiente '{key}' não encontrada. "
            f"Verifique seu arquivo .env"
        )
    return value

# ============================================================================
# CONFIGURAÇÃO (injeção via .env)
# ============================================================================

@dataclass
class AzureConfig:
    """
    Centraliza toda configuração Azure.
    Injeção de dependência: o script não conhece os valores, só o .env conhece.
    """
    openai_endpoint: str
    openai_key: str
    deployment_gpt: str
    deployment_embedding: str
    search_endpoint: str
    search_key: str
    search_index: str

    @classmethod
    def from_env(cls) -> "AzureConfig":
        """Factory method: lê todas as configs do .env de uma vez."""
        return cls(
            openai_endpoint=_get_env("AZURE_OPENAI_ENDPOINT"),
            openai_key=_get_env("AZURE_OPENAI_KEY"),
            deployment_gpt=_get_env("AZURE_OPENAI_DEPLOYMENT"),
            deployment_embedding=_get_env("AZURE_EMBEDDING_DEPLOYMENT"),
            search_endpoint=_get_env("AZURE_SEARCH_ENDPOINT"),
            search_key=_get_env("AZURE_SEARCH_KEY"),
            search_index=_get_env("AZURE_SEARCH_INDEX"),
        )

# ============================================================================
# CLIENTE AZURE OPENAI (único para embeddings + GPT)
# ============================================================================

class AzureOpenAIClient:
    """
    Encapsula chamadas ao Azure OpenAI.
    Responsabilidade única: geração de embeddings e respostas.
    """

    def __init__(self, config: AzureConfig):
        self._config = config
        self._client = AzureOpenAI(
            azure_endpoint=config.openai_endpoint,
            api_key=config.openai_key,
            api_version="2024-02-01"
        )
        log.info("✅ Cliente Azure OpenAI inicializado: %s", config.openai_endpoint)

    def gerar_embedding(self, texto: str) -> Optional[list[float]]:
        """
        Gera embedding vetorial de um texto.
        Retorna None em caso de erro (permite continuar o pipeline).
        """
        try:
            response = self._client.embeddings.create(
                model=self._config.deployment_embedding,
                input=texto
            )
            return response.data[0].embedding

        except RateLimitError:
            log.error("❌ Rate limit atingido. Aguarde e tente novamente.")
            return None
        except APIConnectionError:
            log.error("❌ Erro de conexão com Azure OpenAI. Verifique o endpoint.")
            return None
        except APIError as e:
            log.error("❌ Erro na API OpenAI: %s", str(e))
            return None

    def gerar_resposta(self, query: str, contexto: str) -> Optional[str]:
        """
        Gera resposta usando GPT baseada no contexto recuperado (RAG).
        temperature=0.3: respostas mais factuais, menos criativas.
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Você é um assistente clínico especializado em cuidados paliativos e geriatria. "
                        "Seu único papel é responder perguntas com base nos documentos fornecidos pelo usuário. "
                        "\n\n"
                        "REGRAS OBRIGATÓRIAS:\n"
                        "1. Use EXCLUSIVAMENTE as informações presentes nos documentos fornecidos.\n"
                        "2. NUNCA invente, suponha ou complemente com conhecimento externo.\n"
                        "3. NUNCA crie exemplos, casos clínicos ou situações não descritas nos documentos.\n"
                        "4. Se a pergunta não puder ser respondida com os documentos, responda EXATAMENTE:\n"
                        "   'Não encontrei informações suficientes nos documentos disponíveis para responder a esta pergunta.'\n"
                        "5. Cite sempre o título do documento de onde a informação foi extraída.\n"
                        "6. Não interpole entre documentos para criar conclusões não explícitas.\n"
                        "\n"
                        "FORMATO OBRIGATÓRIO DA RESPOSTA:\n"
                        "- Resposta: <resposta direta baseada nos documentos>\n"
                        "- Fonte: <título do documento utilizado>\n"
                        "- Limitação: <o que os documentos NÃO cobrem sobre a pergunta, se houver>\n"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Documentos relevantes:\n\n{contexto}\n\n"
                        f"Pergunta: {query}"
                    )
                }
            ]

            response = self._client.chat.completions.create(
                model=self._config.deployment_gpt,
                messages=messages,
                temperature=0.3,
                max_tokens=800
            )
            return response.choices[0].message.content

        except RateLimitError:
            log.error("❌ Rate limit atingido no GPT. Aguarde e tente novamente.")
            return None
        except APIError as e:
            log.error("❌ Erro na API GPT: %s", str(e))
            return None

# ============================================================================
# CLIENTE AZURE AI SEARCH
# ============================================================================

class AzureSearchClient:
    """
    Encapsula operações no Azure AI Search.
    Responsabilidade única: indexação e busca vetorial.
    """

    def __init__(self, config: AzureConfig):
        self._config = config
        self._client = SearchClient(
            endpoint=config.search_endpoint,
            index_name=config.search_index,
            credential=AzureKeyCredential(config.search_key), 
            api_version="2021-07-01"
        )
        log.info("✅ Cliente Azure AI Search inicializado: %s", config.search_index)

    def indexar(self, documentos: list[dict]) -> bool:
        """
        Indexa lista de documentos no Azure AI Search.
        Retorna True se todos indexados com sucesso.
        """
        try:
            result = self._client.upload_documents(documents=documentos)
            sucesso = sum(1 for r in result if r.succeeded)
            falha = len(documentos) - sucesso

            if falha > 0:
                log.warning("⚠️ %d documentos falharam na indexação.", falha)

            log.info("✅ %d/%d documentos indexados.", sucesso, len(documentos))
            return falha == 0

        except AzureError as e:
            log.error("❌ Erro ao indexar documentos: %s", str(e))
            return False

    def buscar(self, embedding: list[float], top_k: int = 2) -> list[dict]:
        """
        Busca vetorial por similaridade de cosseno (HNSW).
        Retorna lista de documentos mais relevantes.
        """
        try:
            vector_query = VectorizedQuery(
                vector=embedding,
                k_nearest_neighbors=top_k,
                fields="embedding"
            )

            results = self._client.search(
                search_text=None,
                vector_queries=[vector_query],
                select=["id", "titulo", "conteudo", "categoria"]
            )

            documentos = []
            for i, result in enumerate(results):
                documentos.append({
                    "id": result["id"],
                    "titulo": result["titulo"],
                    "conteudo": result["conteudo"],
                    "categoria": result["categoria"]
                })
                if i + 1 >= top_k:
                    break

            log.info("🔍 %d documentos relevantes encontrados.", len(documentos))
            return documentos

        except AzureError as e:
            log.error("❌ Erro na busca vetorial: %s", str(e))
            return []

# ============================================================================
# PIPELINE RAG
# ============================================================================

class RAGPipeline:
    """
    Orquestra o pipeline RAG completo.
    Depende das abstrações (clientes), não das implementações.
    """

    def __init__(self, openai: AzureOpenAIClient, search: AzureSearchClient):
        self._openai = openai
        self._search = search

    def indexar_documentos(self, path: str) -> bool:
        """
        Lê JSON, gera embeddings e indexa no Azure AI Search.
        """
        if not os.path.exists(path):
            log.error("❌ Arquivo não encontrado: %s", path)
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            log.error("❌ Erro ao ler JSON: %s", str(e))
            return False

        documentos = data.get("value", [])
        log.info("📄 %d documentos carregados de %s", len(documentos), path)

        # Gera embeddings
        for i, doc in enumerate(documentos, 1):
            texto = f"{doc['titulo']}. {doc['conteudo']}"
            log.info("[%d/%d] Gerando embedding: %s...", i, len(documentos), doc["titulo"][:50])

            embedding = self._openai.gerar_embedding(texto)
            if embedding is None:
                log.error("❌ Falha no embedding do doc %s. Abortando.", doc["id"])
                return False

            doc["embedding"] = embedding

        # Indexa
        return self._search.indexar(documentos)

    def responder(self, query: str) -> Optional[str]:
        """
        Pipeline completo: embedding da query → busca → geração de resposta.
        """
        # 1. Embedding da query
        embedding = self._openai.gerar_embedding(query)
        if embedding is None:
            return None

        # 2. Busca vetorial
        docs = self._search.buscar(embedding, top_k=2)
        if not docs:
            log.warning("⚠️ Nenhum documento relevante encontrado.")
            return None

        # Log dos docs recuperados
        for i, doc in enumerate(docs, 1):
            log.info("  [%d] %s (Categoria: %s)", i, doc["titulo"], doc["categoria"])

        # 3. Monta contexto
        contexto = "\n\n".join([
            f"Documento {i+1} - {doc['titulo']}:\n{doc['conteudo']}"
            for i, doc in enumerate(docs)
        ])

        # 4. Gera resposta
        return self._openai.gerar_resposta(query, contexto)

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("🏥 MINI-LAB RAG: CUIDADOS PALIATIVOS E GERIATRIA")
    print("   Azure OpenAI (East US 2) + Azure AI Search (East US)")
    print("=" * 70)

    # Carrega configuração do .env (fail fast se faltar alguma variável)
    try:
        config = AzureConfig.from_env()
    except EnvironmentError as e:
        log.error(str(e))
        return

    # Inicializa clientes (injeção de dependência)
    openai_client = AzureOpenAIClient(config)
    search_client = AzureSearchClient(config)
    pipeline = RAGPipeline(openai_client, search_client)

    # Passo 1: Indexação
    print("\n" + "=" * 70)
    print("PASSO 1: INDEXAÇÃO DE DOCUMENTOS")
    print("=" * 70)

    sucesso = pipeline.indexar_documentos("documentos_cuidados_paliativos.json")
    if not sucesso:
        log.error("❌ Indexação falhou. Verifique os logs acima.")
        return

    # Passo 2: Query de teste
    print("\n" + "=" * 70)
    print("PASSO 2: QUERY DE TESTE (RAG)")
    print("=" * 70)

    query_teste = "Como avaliar o risco de lesões por pressão em paciente idoso acamado?"
    log.info("❓ Query: %s", query_teste)

    resposta = pipeline.responder(query_teste)
    if resposta:
        print(f"\n💡 RESPOSTA:\n{resposta}\n")
    
    # Passo 3: Modo interativo
    print("\n" + "=" * 70)
    print("MODO INTERATIVO (digite 'sair' para encerrar)")
    print("=" * 70 + "\n")

    while True:
        try:
            query = input("🩺 Sua pergunta: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if query.lower() in ["sair", "exit", "quit", ""]:
            break

        resposta = pipeline.responder(query)
        if resposta:
            print(f"\n💡 RESPOSTA:\n{resposta}\n")
            print("-" * 70 + "\n")
        else:
            print("❌ Não foi possível gerar resposta.\n")

    print("\n👋 Encerrando mini-lab. Até logo!")


if __name__ == "__main__":
    main()