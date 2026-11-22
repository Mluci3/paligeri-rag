"""
paligeri/rag.py
Pipeline de inferência RAG (Retrieval-Augmented Generation).

Responsabilidade única:
- Receber query do usuário
- Gerar embedding da query
- Buscar documentos relevantes no Azure AI Search
- Gerar resposta fundamentada com GPT

Quando rodar:
- Sempre que o usuário quiser consultar o assistente
- NUNCA é responsável por indexação (use indexer.py para isso)

Uso (a partir da raiz do projeto):
    PYTHONPATH=src python -m paligeri.rag
"""

import logging
from typing import Optional

from paligeri.config import AzureConfig, AzureOpenAIClient, AzureSearchClient

log = logging.getLogger(__name__)


# ============================================================================
# RAG PIPELINE
# ============================================================================

class RAGPipeline:
    """
    Orquestra o pipeline de inferência RAG.

    Fluxo:
    1. Recebe query do usuário
    2. Gera embedding da query (mesmo modelo usado na indexação)
    3. Busca documentos relevantes por similaridade vetorial
    4. Monta contexto com os documentos recuperados
    5. Gera resposta fundamentada com GPT

    Não conhece nada sobre indexação ou corpus.
    Depende exclusivamente das abstrações de config.py.
    """

    def __init__(self, openai: AzureOpenAIClient, search: AzureSearchClient):
        self._openai = openai
        self._search = search

    def _montar_contexto(self, documentos: list[dict]) -> str:
        """
        Monta o contexto textual a partir dos documentos recuperados.
        Formato estruturado para maximizar groundedness do GPT.
        """
        return "\n\n".join([
            f"Documento {i+1} - {doc['titulo']}:\n{doc['conteudo']}"
            for i, doc in enumerate(documentos)
        ])

    def responder(self, query: str, top_k: int = 2) -> Optional[str]:
        """
        Pipeline completo: query → embedding → busca → resposta.

        Args:
            query: Pergunta do usuário
            top_k: Número de documentos a recuperar (padrão: 2)

        Returns:
            Resposta fundamentada nos documentos ou None em caso de erro.
        """
        # Passo 1: Embedding da query
        log.info("🔄 Gerando embedding da query...")
        embedding = self._openai.gerar_embedding(query)
        if embedding is None:
            log.error("❌ Falha ao gerar embedding da query.")
            return None

        # Passo 2: Busca vetorial
        log.info("🔍 Buscando documentos relevantes...")
        documentos = self._search.buscar(embedding, top_k=top_k)

        if not documentos:
            log.warning("⚠️ Nenhum documento relevante encontrado.")
            return "Não encontrei informações suficientes nos documentos disponíveis para responder a esta pergunta."

        # Log dos documentos recuperados
        for i, doc in enumerate(documentos, 1):
            log.info("  [%d] %s (Categoria: %s)", i, doc["titulo"], doc["categoria"])

        # Passo 3: Monta contexto
        contexto = self._montar_contexto(documentos)

        # Passo 4: Gera resposta
        log.info("🤖 Gerando resposta com GPT...")
        return self._openai.gerar_resposta(query, contexto)


# ============================================================================
# INTERFACE INTERATIVA
# ============================================================================

def executar_modo_interativo(pipeline: RAGPipeline):
    """
    Interface de linha de comando para consultas ao assistente.
    Loop até o usuário digitar 'sair'.
    """
    print("\n" + "=" * 70)
    print("🏥 PALIGERI — Assistente RAG: Cuidados Paliativos e Geriatria")
    print("   Azure OpenAI (East US 2) + Azure AI Search (East US)")
    print("=" * 70)
    print("\nDigite sua pergunta ou 'sair' para encerrar.\n")

    while True:
        try:
            query = input("🩺 Pergunta: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not query:
            continue

        if query.lower() in ["sair", "exit", "quit"]:
            break

        resposta = pipeline.responder(query)

        if resposta:
            print(f"\n💡 RESPOSTA:\n{resposta}\n")
            print("-" * 70 + "\n")
        else:
            print("❌ Não foi possível gerar resposta. Verifique os logs.\n")

    print("\n👋 Encerrando assistente. Até logo!")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    # Carrega configuração do .env (fail fast)
    try:
        config = AzureConfig.from_env()
    except EnvironmentError as e:
        log.error(str(e))
        return

    # Inicializa clientes (injeção de dependência)
    openai_client = AzureOpenAIClient(config)
    search_client = AzureSearchClient(config)

    # Inicializa pipeline
    pipeline = RAGPipeline(openai_client, search_client)

    # Executa modo interativo
    executar_modo_interativo(pipeline)


if __name__ == "__main__":
    main()