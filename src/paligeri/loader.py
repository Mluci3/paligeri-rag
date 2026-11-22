"""
paligeri/loader.py
Ingestão de documentos (.md e .pdf) → corpus JSON pronto para o indexer.

Responsabilidade única:
- Ler arquivos-fonte
- Dividir em chunks (com overlap)
- Gerar JSON no MESMO formato que indexer.py consome

NÃO gera embeddings nem fala com o Azure. Isso permite revisar o corpus
gerado antes de pagar pela indexação.

Uso (a partir da raiz do projeto):
    PYTHONPATH=src python -m paligeri.loader data/raw/manual_ancp.md \\
        --categoria "Manual ANCP" --saida data/corpus_ancp.json

    # depois, revisado o JSON:
    PYTHONPATH=src python -m paligeri.indexer --file data/corpus_ancp.json

Dependência opcional para PDF:  pip install pypdf
"""

import argparse
import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass

log = logging.getLogger(__name__)

# ============================================================================
# PARÂMETROS DE CHUNKING
# ============================================================================
# ~1500 caracteres ≈ 350-400 tokens em português: cabe folgado no limite
# do ada-002 (8191 tokens) e mantém o embedding focado em um assunto.
# Mesma ordem de grandeza dos 5 documentos atuais (~1000-1200 chars),
# o que mantém o corpus homogêneo.
MAX_CHARS = 1500
# Overlap: repete o fim de um chunk no início do próximo para que uma
# frase que cruze a fronteira não perca o contexto.
OVERLAP_CHARS = 200
# Trechos menores que isso (ex.: só um título, número de página) são ruído.
MIN_CHARS = 80

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


@dataclass
class Secao:
    titulos: list[str]  # caminho hierárquico, ex.: ["Dor", "Avaliação"]
    texto: str


# ============================================================================
# CHUNKING
# ============================================================================

def dividir_texto(texto: str, max_chars: int = MAX_CHARS,
                  overlap: int = OVERLAP_CHARS) -> list[str]:
    """
    Divide texto em pedaços de até max_chars, preferindo quebrar entre
    parágrafos, depois entre frases (nunca no meio de uma frase, salvo
    frase gigante). Aplica overlap entre pedaços consecutivos.
    """
    texto = texto.strip()
    if len(texto) <= max_chars:
        return [texto] if texto else []

    # Unidades: parágrafos; parágrafos gigantes viram frases
    unidades: list[str] = []
    for paragrafo in re.split(r"\n\s*\n", texto):
        paragrafo = paragrafo.strip()
        if not paragrafo:
            continue
        if len(paragrafo) <= max_chars:
            unidades.append(paragrafo)
        else:
            unidades.extend(_dividir_em_frases(paragrafo, max_chars))

    chunks: list[str] = []
    atual = ""
    for unidade in unidades:
        candidato = f"{atual}\n\n{unidade}" if atual else unidade
        if len(candidato) <= max_chars:
            atual = candidato
            continue
        chunks.append(atual)
        cauda = _cauda_overlap(atual, overlap)
        atual = f"{cauda}\n\n{unidade}" if cauda else unidade
        if len(atual) > max_chars:  # overlap não pode estourar o limite
            atual = unidade
    if atual:
        chunks.append(atual)
    return chunks


def _dividir_em_frases(paragrafo: str, max_chars: int) -> list[str]:
    frases = re.split(r"(?<=[.!?;])\s+", paragrafo)
    blocos, atual = [], ""
    for frase in frases:
        # Frase isolada maior que o limite: corte duro (raro em texto clínico)
        while len(frase) > max_chars:
            if atual:
                blocos.append(atual)
                atual = ""
            blocos.append(frase[:max_chars])
            frase = frase[max_chars:]
        candidato = f"{atual} {frase}".strip()
        if len(candidato) <= max_chars:
            atual = candidato
        else:
            blocos.append(atual)
            atual = frase
    if atual:
        blocos.append(atual)
    return blocos


def _cauda_overlap(texto: str, overlap: int) -> str:
    """Últimos ~overlap caracteres, começando em início de palavra."""
    if overlap <= 0 or len(texto) <= overlap:
        return ""
    cauda = texto[-overlap:]
    espaco = cauda.find(" ")
    return cauda[espaco + 1:] if espaco != -1 else cauda


# ============================================================================
# LEITORES
# ============================================================================

def ler_markdown(path: str) -> list[Secao]:
    """
    Divide o Markdown em seções pelos títulos (#, ##, ...).
    Por quê: o autor do manual já agrupou o conteúdo por assunto; respeitar
    essa estrutura dá chunks semanticamente coesos "de graça".
    """
    with open(path, encoding="utf-8") as f:
        linhas = f.read().splitlines()

    secoes: list[Secao] = []
    pilha: list[tuple[int, str]] = []  # (nível, título)
    buffer: list[str] = []

    def fechar_secao():
        texto = "\n".join(buffer).strip()
        if texto:
            secoes.append(Secao([t for _, t in pilha], texto))
        buffer.clear()

    for linha in linhas:
        m = _HEADING_RE.match(linha)
        if m:
            fechar_secao()
            nivel = len(m.group(1))
            while pilha and pilha[-1][0] >= nivel:
                pilha.pop()
            pilha.append((nivel, m.group(2)))
        else:
            buffer.append(linha)
    fechar_secao()
    return secoes


def ler_pdf(path: str) -> list[Secao]:
    """
    Uma seção por página. Por quê: PDF não guarda títulos de forma
    estruturada; a página é a única fronteira confiável. Para melhor
    qualidade, converta o PDF para .md antes (preserva os títulos).
    """
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError("Para ler PDF instale: pip install pypdf") from e

    secoes = []
    for i, pagina in enumerate(PdfReader(path).pages, 1):
        texto = (pagina.extract_text() or "").strip()
        if texto:
            secoes.append(Secao([f"p. {i}"], texto))
    return secoes


# ============================================================================
# MONTAGEM DO CORPUS
# ============================================================================

def slug(texto: str) -> str:
    """
    Gera id compatível com Azure AI Search (só letras, números, _ - =).
    Remove acentos: 'Manual ANCP – Dor' → 'manual-ancp-dor'.
    """
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", sem_acento.lower()).strip("-") or "doc"


def gerar_documentos(path: str, categoria: str) -> list[dict]:
    """
    Lê um arquivo e devolve documentos no formato do corpus.

    Ids determinísticos (arquivo + posição): reprocessar o mesmo arquivo
    ATUALIZA os docs no índice em vez de duplicar (upload é idempotente
    por id).
    O título carrega o caminho de seções ("Manual > Dor > Avaliação"):
    melhora o embedding (indexer usa titulo + conteudo) e a citação da fonte.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".md", ".markdown"):
        secoes = ler_markdown(path)
    elif ext == ".pdf":
        secoes = ler_pdf(path)
    else:
        raise ValueError(f"Formato não suportado: {ext} (use .md ou .pdf)")

    nome_base = os.path.splitext(os.path.basename(path))[0]
    prefixo = slug(nome_base)
    documentos = []

    for secao in secoes:
        caminho = " > ".join(secao.titulos) or nome_base
        for parte in dividir_texto(secao.texto):
            if len(parte) < MIN_CHARS:
                continue
            documentos.append({
                "@search.action": "upload",
                "id": f"{prefixo}-{len(documentos) + 1:04d}",
                "titulo": f"{nome_base} — {caminho}",
                "conteudo": parte,
                "categoria": categoria,
            })

    log.info("📄 %s → %d seções → %d chunks", path, len(secoes), len(documentos))
    return documentos


def main():
    parser = argparse.ArgumentParser(description="Gera corpus JSON a partir de .md/.pdf.")
    parser.add_argument("arquivos", nargs="+", help="Arquivos .md ou .pdf")
    parser.add_argument("--categoria", default="Geral")
    parser.add_argument("--saida", required=True, help="Caminho do JSON de saída")
    args = parser.parse_args()

    documentos = []
    for path in args.arquivos:
        documentos.extend(gerar_documentos(path, args.categoria))

    with open(args.saida, "w", encoding="utf-8") as f:
        json.dump({"value": documentos}, f, ensure_ascii=False, indent=2)

    tamanhos = [len(d["conteudo"]) for d in documentos] or [0]
    print(f"✅ {len(documentos)} chunks salvos em {args.saida}")
    print(f"   tamanho (chars): mín {min(tamanhos)} | médio "
          f"{sum(tamanhos) // len(tamanhos)} | máx {max(tamanhos)}")
    print("   Revise o JSON antes de rodar o indexer.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
