"""
paligeri/setup_search.py
Infraestrutura do Azure AI Search como código (índice + synonym map).

Por que existe:
- Hoje o índice foi criado à mão no portal. Isso não é reproduzível:
  ninguém (nem você daqui a 6 meses) sabe qual era o schema exato.
- Este script documenta o schema e permite recriá-lo em outro ambiente.

Comandos (a partir da raiz do projeto):
    PYTHONPATH=src python -m paligeri.setup_search verificar
        Só leitura. Mostra campos do índice e nº de documentos. Custo zero.

    PYTHONPATH=src python -m paligeri.setup_search sinonimos
        Cria/atualiza o synonym map e o associa aos campos de texto
        do índice EXISTENTE. Não apaga nada, não reindexa.

    PYTHONPATH=src python -m paligeri.setup_search criar-indice
        Cria o índice do zero (só se ele ainda NÃO existir).
"""

import argparse
import logging
import os

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError, ResourceNotFoundError
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SimpleField,
    SynonymMap,
    VectorSearch,
    VectorSearchProfile,
)

from paligeri.config import ROOT_DIR, AzureConfig

log = logging.getLogger(__name__)

SYNONYM_MAP_NAME = "sinonimos-clinicos"
SINONIMOS_PATH = os.path.join(ROOT_DIR, "data", "sinonimos.txt")
CAMPOS_TEXTO = ("titulo", "conteudo")
EMBEDDING_DIMS = 1536  # text-embedding-ada-002


def carregar_regras_sinonimos(path: str = SINONIMOS_PATH) -> list[str]:
    """Lê o arquivo de sinônimos ignorando comentários e linhas vazias."""
    with open(path, encoding="utf-8") as f:
        return [
            linha.strip()
            for linha in f
            if linha.strip() and not linha.lstrip().startswith("#")
        ]


def definir_indice(nome: str) -> SearchIndex:
    """
    Schema do índice. Espelha os campos usados por indexer.py e rag.py.

    - analyzer pt-Br.microsoft: lematização em português
      ("dores" encontra "dor"), melhor que o analisador padrão (inglês).
    - Vetorial HNSW com 1536 dimensões (ada-002).
    """
    return SearchIndex(
        name=nome,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(
                name="titulo",
                analyzer_name="pt-Br.microsoft",
                synonym_map_names=[SYNONYM_MAP_NAME],
            ),
            SearchableField(
                name="conteudo",
                analyzer_name="pt-Br.microsoft",
                synonym_map_names=[SYNONYM_MAP_NAME],
            ),
            SimpleField(
                name="categoria",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
            ),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=EMBEDDING_DIMS,
                vector_search_profile_name="perfil-hnsw",
            ),
        ],
        vector_search=VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
            profiles=[
                VectorSearchProfile(
                    name="perfil-hnsw", algorithm_configuration_name="hnsw"
                )
            ],
        ),
    )


def verificar(client: SearchIndexClient, nome: str, config: AzureConfig) -> None:
    """Só leitura: mostra o schema atual e a contagem de documentos."""
    from azure.search.documents import SearchClient

    indice = client.get_index(nome)
    print(f"\n📋 Índice '{indice.name}':")
    for campo in indice.fields:
        extras = []
        if campo.key:
            extras.append("key")
        if campo.searchable:
            extras.append("searchable")
        if getattr(campo, "analyzer_name", None):
            extras.append(f"analyzer={campo.analyzer_name}")
        if getattr(campo, "synonym_map_names", None):
            extras.append(f"sinônimos={campo.synonym_map_names}")
        if getattr(campo, "vector_search_dimensions", None):
            extras.append(f"dims={campo.vector_search_dimensions}")
        print(f"   - {campo.name}: {', '.join(extras) or '-'}")

    total = SearchClient(
        config.search_endpoint, nome, AzureKeyCredential(config.search_key)
    ).get_document_count()
    print(f"\n📄 Documentos no índice: {total}")


def aplicar_sinonimos(client: SearchIndexClient, nome: str) -> None:
    """Cria/atualiza o synonym map e o associa aos campos de texto."""
    regras = carregar_regras_sinonimos()
    client.create_or_update_synonym_map(
        SynonymMap(name=SYNONYM_MAP_NAME, synonyms=regras)
    )
    log.info("✅ Synonym map '%s' com %d regras.", SYNONYM_MAP_NAME, len(regras))

    indice = client.get_index(nome)
    alterados = []
    for campo in indice.fields:
        if campo.name in CAMPOS_TEXTO:
            if not campo.searchable:
                log.warning("⚠️ Campo '%s' não é searchable; ignorado.", campo.name)
                continue
            campo.synonym_map_names = [SYNONYM_MAP_NAME]
            alterados.append(campo.name)

    client.create_or_update_index(indice)
    log.info("✅ Sinônimos associados aos campos: %s", ", ".join(alterados))


def criar_indice(client: SearchIndexClient, nome: str) -> None:
    """Cria o índice do zero. Recusa se já existir (evita sobrescrever)."""
    try:
        client.get_index(nome)
        log.error("❌ Índice '%s' já existe. Nada foi alterado.", nome)
        return
    except ResourceNotFoundError:
        pass

    # O synonym map precisa existir antes do índice que o referencia
    client.create_or_update_synonym_map(
        SynonymMap(name=SYNONYM_MAP_NAME, synonyms=carregar_regras_sinonimos())
    )
    client.create_index(definir_indice(nome))
    log.info("✅ Índice '%s' criado. Rode o indexer para popular.", nome)


def main():
    parser = argparse.ArgumentParser(description="Setup do Azure AI Search.")
    parser.add_argument("comando", choices=["verificar", "sinonimos", "criar-indice"])
    args = parser.parse_args()

    try:
        config = AzureConfig.from_env()
    except EnvironmentError as e:
        log.error(str(e))
        return

    client = SearchIndexClient(
        config.search_endpoint, AzureKeyCredential(config.search_key)
    )

    try:
        if args.comando == "verificar":
            verificar(client, config.search_index, config)
        elif args.comando == "sinonimos":
            aplicar_sinonimos(client, config.search_index)
        else:
            criar_indice(client, config.search_index)
    except AzureError as e:
        log.error("❌ Erro no Azure AI Search: %s", str(e))


if __name__ == "__main__":
    main()
