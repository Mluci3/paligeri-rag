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

Uso (a partir da raiz do projeto):
    PYTHONPATH=src python -m paligeri.indexer
    PYTHONPATH=src python -m paligeri.indexer --file data/meu_corpus.json
"""

import json
import logging
import argparse
import os
import time
from typing import Optional

from paligeri.config import ROOT_DIR, AzureConfig, AzureOpenAIClient, AzureSearchClient

log = logging.getLogger(__name__)

# ============================================================================
# CONSTANTES
# ============================================================================

# Caminho absoluto, ancorado na raiz do projeto: funciona de qualquer diretório.
DEFAULT_DOCS_PATH = os.path.join(ROOT_DIR, "data", "documentos_cuidados_paliativos.json")

# Textos por requisição de embedding: poucas requisições sem estourar
# o limite de tokens por minuto (16 chunks × ~400 tokens ≈ 6,4k tokens).
LOTE_EMBEDDING = 16
# Azure AI Search aceita no máximo 1000 documentos (e 16 MB) por envio;
# 500 vetores de 1536 floats ficam folgados abaixo de 16 MB.
LOTE_UPLOAD = 500
# Retentativas por lote (limite de tokens/minuto renova a cada 60 s)
TENTATIVAS = 5
ESPERA_S = 30


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

    def _embedar_com_retentativa(self, textos: list[str]) -> Optional[list[list[float]]]:
        """
        Tenta o lote até TENTATIVAS vezes, esperando ESPERA_S entre elas.

        Por quê: em corpus grande o limite de tokens por minuto do
        deployment é atingido com frequência (erro 429). Esperar e repetir
        é o comportamento correto; abortar perderia todo o progresso.
        """
        for tentativa in range(1, TENTATIVAS + 1):
            vetores = self._openai.gerar_embeddings(textos)
            if vetores is not None:
                return vetores
            if tentativa < TENTATIVAS:
                log.warning("⏳ Tentativa %d/%d falhou; aguardando %ds...",
                            tentativa, TENTATIVAS, ESPERA_S)
                time.sleep(ESPERA_S)
        return None

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

        # Passo 2: Validar e gerar embeddings em lotes
        documentos_validos = [d for d in documentos if self._validar_documento(d)]
        if not documentos_validos:
            log.error("❌ Nenhum documento válido para indexar.")
            return False

        total = len(documentos_validos)
        print(f"\n🔄 Gerando embeddings para {total} documentos "
              f"(lotes de {LOTE_EMBEDDING})...")

        for inicio in range(0, total, LOTE_EMBEDDING):
            lote = documentos_validos[inicio:inicio + LOTE_EMBEDDING]
            log.info("[%d-%d/%d] Embedando lote...", inicio + 1, inicio + len(lote), total)
            vetores = self._embedar_com_retentativa(
                [self._gerar_texto_embedding(d) for d in lote]
            )
            if vetores is None or len(vetores) != len(lote):
                log.error("❌ Falha no lote iniciado em '%s'. Abortando.", lote[0]["id"])
                return False
            for doc, vetor in zip(lote, vetores):
                doc["embedding"] = vetor

        log.info("✅ %d embeddings gerados com sucesso.", total)

        # Passo 3: Indexar no Azure AI Search em lotes (limite: 1000 docs/envio)
        print(f"\n📤 Indexando {total} documentos no Azure AI Search "
              f"(lotes de {LOTE_UPLOAD})...")
        sucesso = all(
            self._search.indexar(documentos_validos[i:i + LOTE_UPLOAD])
            for i in range(0, total, LOTE_UPLOAD)
        )

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