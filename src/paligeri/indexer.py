"""
indexer.py
Pipeline de indexação de documentos no Azure AI Search.

Responsabilidade única:
- Ler documentos do JSON
- Gerar embeddings via Azure OpenAI
- Indexar no Azure AI Search

Quando rodar:
- Na primeira vez (setup inicial)
- Quando o corpus de documentos mudar
- NUNCA durante inferência (use rag.py para isso)

Uso:
    python indexer.py
    python indexer.py --file meu_corpus.json
"""

import json
import logging
import argparse
import os
from typing import Optional

from paligeri.config import AzureConfig, AzureOpenAIClient, AzureSearchClient

log = logging.getLogger(__name__)

# ============================================================================
# CONSTANTES
# ============================================================================

DEFAULT_DOCS_PATH = "documentos_cuidados_paliativos.json"


# ============================================================================
# INDEXER
# ============================================================================

class Indexer:
    """
    Orquestra o pipeline de indexação.

    Fluxo:
    1. Carrega documentos do JSON
    2. Para cada documento, gera embedding (titulo + conteudo)
    3. Indexa no Azure AI Search (idempotente via campo 'id')

    Não conhece nada sobre busca vetorial ou geração de respostas.
    """

    def __init__(self, openai: AzureOpenAIClient, search: AzureSearchClient):
        self._openai = openai
        self._search = search

    def carregar_documentos(self, path: str) -> Optional[list[dict]]:
        """
        Lê e valida o arquivo JSON de documentos.
        Retorna None se arquivo inválido ou não encontrado.
        """
        if not os.path.exists(path):
            log.error("❌ Arquivo não encontrado: %s", path)
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            log.error("❌ JSON inválido em %s: %s", path, str(e))
            return None

        documentos = data.get("value", [])

        if not documentos:
            log.error("❌ Nenhum documento encontrado no arquivo.")
            return None

        log.info("📄 %d documentos carregados de: %s", len(documentos), path)
        return documentos

    def _gerar_texto_embedding(self, doc: dict) -> str:
        """
        Define o texto usado para gerar o embedding de cada documento.

        Estratégia: titulo + conteudo
        - Título ancora o tema principal do chunk
        - Conteúdo carrega a semântica detalhada

        Nota para AI-102: chunking strategy impacta diretamente
        a qualidade da busca vetorial. Documentos muito longos
        devem ser divididos antes de embedar.
        """
        return f"{doc['titulo']}. {doc['conteudo']}"

    def _validar_documento(self, doc: dict) -> bool:
        """
        Valida campos obrigatórios antes de processar.
        Evita erros silenciosos na indexação.
        """
        campos_obrigatorios = ["id", "titulo", "conteudo", "categoria"]
        for campo in campos_obrigatorios:
            if campo not in doc or not doc[campo]:
                log.warning("⚠️ Documento com campo '%s' ausente ou vazio. Ignorando.", campo)
                return False
        return True

    def executar(self, path: str) -> bool:
        """
        Executa o pipeline completo de indexação.

        Retorna True se todos os documentos foram indexados com sucesso.
        """
        print("=" * 70)
        print("📥 INDEXER: PIPELINE DE INDEXAÇÃO")
        print("   Azure OpenAI (embeddings) → Azure AI Search")
        print("=" * 70)

        # Passo 1: Carregar documentos
        documentos = self.carregar_documentos(path)
        if documentos is None:
            return False

        # Passo 2: Gerar embeddings
        print(f"\n🔄 Gerando embeddings para {len(documentos)} documentos...")
        documentos_validos = []

        for i, doc in enumerate(documentos, 1):

            # Valida campos obrigatórios
            if not self._validar_documento(doc):
                continue

            # Gera texto para embedding
            texto = self._gerar_texto_embedding(doc)

            log.info("[%d/%d] Embedando: %s...", i, len(documentos), doc["titulo"][:55])

            # Gera embedding
            embedding = self._openai.gerar_embedding(texto)
            if embedding is None:
                log.error("❌ Falha ao gerar embedding para doc '%s'. Abortando.", doc["id"])
                return False

            doc["embedding"] = embedding
            documentos_validos.append(doc)

        if not documentos_validos:
            log.error("❌ Nenhum documento válido para indexar.")
            return False

        log.info("✅ %d embeddings gerados com sucesso.", len(documentos_validos))

        # Passo 3: Indexar no Azure AI Search
        print(f"\n📤 Indexando {len(documentos_validos)} documentos no Azure AI Search...")
        sucesso = self._search.indexar(documentos_validos)

        if sucesso:
            print("\n" + "=" * 70)
            print("✅ INDEXAÇÃO CONCLUÍDA COM SUCESSO")
            print(f"   {len(documentos_validos)} documentos prontos para busca vetorial.")
            print("   Execute rag.py para iniciar o assistente.")
            print("=" * 70)
        else:
            print("\n❌ INDEXAÇÃO FALHOU. Verifique os logs acima.")

        return sucesso


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    # Argumento opcional para path do arquivo de documentos
    parser = argparse.ArgumentParser(
        description="Indexa documentos no Azure AI Search com embeddings."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=DEFAULT_DOCS_PATH,
        help=f"Caminho para o JSON de documentos (padrão: {DEFAULT_DOCS_PATH})"
    )
    args = parser.parse_args()

    # Carrega configuração do .env (fail fast)
    try:
        config = AzureConfig.from_env()
    except EnvironmentError as e:
        log.error(str(e))
        return

    # Inicializa clientes (injeção de dependência)
    openai_client = AzureOpenAIClient(config)
    search_client = AzureSearchClient(config)

    # Executa indexação
    indexer = Indexer(openai_client, search_client)
    indexer.executar(args.file)


if __name__ == "__main__":
    main()