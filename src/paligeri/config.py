"""
paligeri/config.py
Configuração centralizada e clientes Azure compartilhados.

Responsabilidades:
- Carregar variáveis de ambiente (.env)
- Inicializar clientes Azure OpenAI e Azure AI Search
- Expor interfaces para indexer.py e rag.py

Padrões aplicados:
- Fail fast: erro claro se variável de ambiente não encontrada
- Single Responsibility: cada classe tem uma responsabilidade
- Injeção de dependência: clientes recebem config, não leem .env diretamente
- frozen=True: configuração imutável após inicialização
"""

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
# LOGS
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# Candidatos vetoriais no modo híbrido (o RRF precisa de mais que top_k)
HYBRID_K_VIZINHOS = 50

# ============================================================================
# CARREGAMENTO DO .env
# ============================================================================

# Sobe 3 níveis a partir de src/paligeri/ para encontrar .env na raiz
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(ROOT_DIR, ".env"))


def _get_env(key: str) -> str:
    """
    Lê variável de ambiente obrigatória.
    Fail fast: lança EnvironmentError imediatamente se não encontrada.
    Nunca retorna None — garante que o sistema não inicia com config incompleta.
    """
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"[config.py] Variável '{key}' não encontrada no .env. "
            f"Verifique o arquivo .env na raiz do projeto."
        )
    return value


# ============================================================================
# DATACLASS DE CONFIGURAÇÃO
# ============================================================================

@dataclass(frozen=True)
class AzureConfig:
    """
    Configuração imutável dos recursos Azure.

    frozen=True: após criado, nenhum atributo pode ser alterado.
    Garante consistência durante toda a execução.
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
        """
        Factory method: lê todas as variáveis do .env de uma vez.
        Ponto único de entrada para configuração.
        """
        log.info("🔧 Carregando configuração do .env...")
        config = cls(
            openai_endpoint=_get_env("AZURE_OPENAI_ENDPOINT"),
            openai_key=_get_env("AZURE_OPENAI_KEY"),
            deployment_gpt=_get_env("AZURE_OPENAI_DEPLOYMENT"),
            deployment_embedding=_get_env("AZURE_EMBEDDING_DEPLOYMENT"),
            search_endpoint=_get_env("AZURE_SEARCH_ENDPOINT"),
            search_key=_get_env("AZURE_SEARCH_KEY"),
            search_index=_get_env("AZURE_SEARCH_INDEX"),
        )
        log.info("✅ Configuração carregada com sucesso.")
        return config


# ============================================================================
# CLIENTE AZURE OPENAI
# ============================================================================

class AzureOpenAIClient:
    """
    Encapsula chamadas ao Azure OpenAI.

    Responsabilidade única:
    - gerar_embedding(): vetoriza textos para indexação e busca
    - gerar_resposta(): gera texto baseado em contexto (RAG)

    Um único recurso Azure OpenAI serve ambos os deployments
    (embeddings + GPT), por isso um único cliente é suficiente.
    """

    def __init__(self, config: AzureConfig):
        self._config = config
        self._client = AzureOpenAI(
            azure_endpoint=config.openai_endpoint,
            api_key=config.openai_key,
            api_version="2024-02-01"
        )
        log.info("✅ AzureOpenAIClient inicializado: %s", config.openai_endpoint)

    def gerar_embedding(self, texto: str) -> Optional[list[float]]:
        """
        Gera embedding vetorial (1536 dimensões) de um texto.

        Usado por:
        - indexer.py: vetoriza documentos antes de indexar
        - rag.py: vetoriza a query do usuário antes de buscar
        """
        try:
            response = self._client.embeddings.create(
                model=self._config.deployment_embedding,
                input=texto
            )
            return response.data[0].embedding

        except RateLimitError:
            log.error("❌ Rate limit atingido (embeddings). Aguarde e tente novamente.")
            return None
        except APIConnectionError:
            log.error("❌ Falha de conexão com Azure OpenAI. Verifique o endpoint.")
            return None
        except APIError as e:
            log.error("❌ Erro na API OpenAI (embeddings): %s", str(e))
            return None

    def gerar_resposta(self, query: str, contexto: str) -> Optional[str]:
        """
        Gera resposta fundamentada nos documentos recuperados (RAG).

        System prompt com regras de groundedness:
        - Responde APENAS com base nos documentos fornecidos
        - Recusa claramente quando informação não está disponível
        - Cita a fonte de cada afirmação
        - Não inventa, não itera, não completa com conhecimento externo

        temperature=0.0: máximo determinismo para contexto clínico.
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Você é um assistente clínico especializado em cuidados paliativos e geriatria. "
                        "Seu único papel é responder perguntas com base nos documentos fornecidos pelo usuário.\n\n"
                        "REGRAS OBRIGATÓRIAS:\n"
                        "1. Use EXCLUSIVAMENTE as informações presentes nos documentos fornecidos.\n"
                        "2. NUNCA invente, suponha ou complemente com conhecimento externo.\n"
                        "3. NUNCA crie exemplos, casos clínicos ou situações não descritas nos documentos.\n"
                        "4. Se a pergunta não puder ser respondida com os documentos, responda EXATAMENTE:\n"
                        "   'Não encontrei informações suficientes nos documentos disponíveis para responder a esta pergunta.'\n"
                        "5. Cite sempre o título do documento de onde a informação foi extraída.\n"
                        "6. Não interpole entre documentos para criar conclusões não explícitas.\n\n"
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
                temperature=0.0,
                max_tokens=800
            )
            return response.choices[0].message.content

        except RateLimitError:
            log.error("❌ Rate limit atingido (GPT). Aguarde e tente novamente.")
            return None
        except APIError as e:
            log.error("❌ Erro na API OpenAI (GPT): %s", str(e))
            return None

    def chat_json(self, system: str, user: str, max_tokens: int = 300) -> Optional[str]:
        """
        Chamada genérica que força saída JSON (response_format).
        Usada pelo eval.py como "LLM-as-judge". Separada de gerar_resposta()
        para não misturar o prompt clínico com o prompt do avaliador.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._config.deployment_gpt,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except (RateLimitError, APIError) as e:
            log.error("❌ Erro na API OpenAI (juiz): %s", str(e))
            return None


# ============================================================================
# CLIENTE AZURE AI SEARCH
# ============================================================================

class AzureSearchClient:
    """
    Encapsula operações no Azure AI Search.

    Responsabilidade única:
    - indexar(): recebe documentos vetorizados e sobe no índice
    - buscar(): recebe embedding da query e retorna docs relevantes

    api_version="2024-07-01": versão estável fixada explicitamente.
    """

    def __init__(self, config: AzureConfig):
        self._config = config
        self._client = SearchClient(
            endpoint=config.search_endpoint,
            index_name=config.search_index,
            credential=AzureKeyCredential(config.search_key),
            api_version="2024-07-01"
        )
        log.info("✅ AzureSearchClient inicializado: %s | índice: %s",
                 config.search_endpoint, config.search_index)

    def indexar(self, documentos: list[dict]) -> bool:
        """
        Indexa documentos vetorizados no Azure AI Search.
        Idempotente: documentos com mesmo 'id' são atualizados, não duplicados.
        """
        try:
            result = self._client.upload_documents(documents=documentos)
            sucesso = sum(1 for r in result if r.succeeded)
            falha = len(documentos) - sucesso

            if falha > 0:
                log.warning("⚠️ %d/%d documentos falharam.", falha, len(documentos))

            log.info("✅ %d/%d documentos indexados.", sucesso, len(documentos))
            return falha == 0

        except AzureError as e:
            log.error("❌ Erro ao indexar: %s", str(e))
            return False

    def buscar(
        self,
        embedding: list[float],
        top_k: int = 3,
        texto: Optional[str] = None,
    ) -> list[dict]:
        """
        Busca no índice. Dois modos:

        - Vetorial pura (texto=None): similaridade de cosseno (HNSW).
        - Híbrida (texto informado): BM25 (palavras-chave) + vetorial,
          combinadas pelo Azure via RRF (Reciprocal Rank Fusion).

        Por que híbrida? A vetorial acerta paráfrases, mas erra termos
        exatos (siglas, nomes de escalas). A BM25 é o oposto. E só a parte
        textual aplica synonym maps (ex.: "demência" → "déficit cognitivo").

        No modo híbrido, a parte vetorial busca HYBRID_K_VIZINHOS candidatos
        para que o RRF tenha material para fundir; `top` corta no final.
        """
        try:
            k_vetorial = top_k if texto is None else max(top_k, HYBRID_K_VIZINHOS)
            vector_query = VectorizedQuery(
                vector=embedding,
                k_nearest_neighbors=k_vetorial,
                fields="embedding"
            )

            results = self._client.search(
                search_text=texto,
                vector_queries=[vector_query],
                select=["id", "titulo", "conteudo", "categoria"],
                top=top_k,
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

            log.info("🔍 %d documentos encontrados.", len(documentos))
            return documentos

        except AzureError as e:
            log.error("❌ Erro na busca vetorial: %s", str(e))
            return []