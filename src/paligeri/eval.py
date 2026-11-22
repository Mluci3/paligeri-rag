"""
paligeri/eval.py
Avaliação quantitativa do RAG em duas camadas.

1. Retrieval (padrão, BARATO — só embeddings):
   Compara busca vetorial x híbrida no golden set.
   Métricas: Hit@1, Hit@k e MRR.

2. Geração (opcional, --julgar — usa GPT, custa mais):
   LLM-as-judge dá notas de 1 a 5 para:
   - faithfulness: a resposta só afirma o que está no contexto?
   - relevancy: a resposta responde à pergunta?
   E verifica se perguntas fora do escopo são RECUSADAS.

Uso (a partir da raiz do projeto):
    PYTHONPATH=src python -m paligeri.eval              # só retrieval
    PYTHONPATH=src python -m paligeri.eval --julgar     # + geração
"""

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Optional

from paligeri.config import ROOT_DIR, AzureConfig, AzureOpenAIClient, AzureSearchClient
from paligeri.rag import RAGPipeline

log = logging.getLogger(__name__)

GOLDEN_PATH = os.path.join(ROOT_DIR, "data", "avaliacao.json")
RESULTADOS_DIR = os.path.join(ROOT_DIR, "resultados")
FRASE_RECUSA = "Não encontrei informações suficientes"

PROMPT_JUIZ = (
    "Você é um avaliador rigoroso de sistemas RAG clínicos. "
    "Receberá CONTEXTO, PERGUNTA e RESPOSTA. Avalie:\n"
    "- faithfulness (1-5): 5 = toda afirmação da resposta está no contexto; "
    "1 = há afirmações inventadas ou não suportadas.\n"
    "- relevancy (1-5): 5 = responde diretamente à pergunta; 1 = não responde.\n"
    "Responda APENAS em JSON: "
    '{"faithfulness": int, "relevancy": int, "justificativa": "<1 frase>"}'
)


# ============================================================================
# MÉTRICAS (funções puras — testáveis sem Azure)
# ============================================================================

def hit_at_k(recuperados: list[str], esperados: list[str], k: int) -> float:
    """1.0 se algum id esperado está entre os k primeiros, senão 0.0."""
    return 1.0 if set(recuperados[:k]) & set(esperados) else 0.0


def reciprocal_rank(recuperados: list[str], esperados: list[str]) -> float:
    """1/posição do primeiro id esperado (1º → 1.0, 2º → 0.5...). 0 se ausente."""
    for posicao, doc_id in enumerate(recuperados, 1):
        if doc_id in esperados:
            return 1.0 / posicao
    return 0.0


def resumir(linhas: list[dict], modo: str, k: int) -> dict:
    """Média das métricas de um modo de busca."""
    n = len(linhas) or 1
    return {
        "hit@1": sum(l[modo]["hit@1"] for l in linhas) / n,
        f"hit@{k}": sum(l[modo][f"hit@{k}"] for l in linhas) / n,
        "mrr": sum(l[modo]["rr"] for l in linhas) / n,
    }


def carregar_golden(path: str = GOLDEN_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["casos"]


# ============================================================================
# CAMADA 1 — RETRIEVAL
# ============================================================================

def avaliar_retrieval(openai: AzureOpenAIClient, search: AzureSearchClient,
                      casos: list[dict], k: int) -> Optional[list[dict]]:
    """
    Para cada pergunta: gera o embedding UMA vez e roda as duas buscas.
    Assim a comparação é justa (mesmo vetor) e o custo cai pela metade.
    Perguntas fora do escopo (esperados vazio) não entram aqui.
    """
    linhas = []
    for caso in casos:
        if not caso["esperados"]:
            continue
        embedding = openai.gerar_embedding(caso["pergunta"])
        if embedding is None:
            return None

        linha = {"id": caso["id"], "tipo": caso["tipo"], "pergunta": caso["pergunta"]}
        for modo, texto in (("vetorial", None), ("hibrida", caso["pergunta"])):
            ids = [d["id"] for d in search.buscar(embedding, top_k=k, texto=texto)]
            linha[modo] = {
                "recuperados": ids,
                "hit@1": hit_at_k(ids, caso["esperados"], 1),
                f"hit@{k}": hit_at_k(ids, caso["esperados"], k),
                "rr": reciprocal_rank(ids, caso["esperados"]),
            }
        linhas.append(linha)
    return linhas


def imprimir_retrieval(linhas: list[dict], k: int) -> None:
    print("\n📊 RETRIEVAL — posição do documento esperado (– = não encontrado)\n")
    print(f"{'id':<5}{'tipo':<16}{'vetorial':>10}{'híbrida':>10}")
    for l in linhas:
        pos = lambda m: f"{int(1 / l[m]['rr'])}º" if l[m]["rr"] else "–"
        print(f"{l['id']:<5}{l['tipo']:<16}{pos('vetorial'):>10}{pos('hibrida'):>10}")

    print(f"\n{'métrica':<10}{'vetorial':>10}{'híbrida':>10}")
    rv, rh = resumir(linhas, "vetorial", k), resumir(linhas, "hibrida", k)
    for metrica in rv:
        print(f"{metrica:<10}{rv[metrica]:>10.2f}{rh[metrica]:>10.2f}")


# ============================================================================
# CAMADA 2 — GERAÇÃO (LLM-as-judge)
# ============================================================================

def avaliar_geracao(openai: AzureOpenAIClient, pipeline: RAGPipeline,
                    casos: list[dict], k: int) -> list[dict]:
    resultados = []
    for caso in casos:
        docs = pipeline.recuperar(caso["pergunta"], top_k=k) or []
        contexto = pipeline.montar_contexto(docs)
        resposta = openai.gerar_resposta(caso["pergunta"], contexto) or ""
        recusou = FRASE_RECUSA in resposta
        linha = {"id": caso["id"], "tipo": caso["tipo"], "resposta": resposta, "recusou": recusou}

        if not caso["esperados"]:
            # Fora do escopo: o correto é recusar. Não faz sentido julgar fidelidade.
            linha["correto"] = recusou
        else:
            bruto = openai.chat_json(
                PROMPT_JUIZ,
                f"CONTEXTO:\n{contexto}\n\nPERGUNTA: {caso['pergunta']}\n\nRESPOSTA:\n{resposta}",
            )
            try:
                linha.update(json.loads(bruto or "{}"))
            except json.JSONDecodeError:
                log.warning("⚠️ Juiz devolveu JSON inválido em %s", caso["id"])
        resultados.append(linha)
    return resultados


def imprimir_geracao(resultados: list[dict]) -> None:
    print("\n🧑‍⚖️ GERAÇÃO — LLM-as-judge (1 a 5)\n")
    print(f"{'id':<5}{'tipo':<16}{'faith':>6}{'relev':>6}  observação")
    for r in resultados:
        if "correto" in r:
            obs = "✅ recusou corretamente" if r["correto"] else "❌ deveria ter recusado"
            print(f"{r['id']:<5}{r['tipo']:<16}{'-':>6}{'-':>6}  {obs}")
        else:
            print(f"{r['id']:<5}{r['tipo']:<16}{r.get('faithfulness', '?'):>6}"
                  f"{r.get('relevancy', '?'):>6}  {r.get('justificativa', '')[:60]}")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Avaliação do RAG Paligeri.")
    parser.add_argument("--k", type=int, default=3, help="Top-k da busca (padrão 3)")
    parser.add_argument("--julgar", action="store_true",
                        help="Também avalia a geração com LLM-as-judge (usa GPT)")
    args = parser.parse_args()

    try:
        config = AzureConfig.from_env()
    except EnvironmentError as e:
        log.error(str(e))
        return

    openai = AzureOpenAIClient(config)
    search = AzureSearchClient(config)
    casos = carregar_golden()

    saida = {"data": datetime.now().isoformat(timespec="seconds"), "k": args.k}

    linhas = avaliar_retrieval(openai, search, casos, args.k)
    if linhas is None:
        log.error("❌ Falha nos embeddings; avaliação interrompida.")
        return
    imprimir_retrieval(linhas, args.k)
    saida["retrieval"] = linhas

    if args.julgar:
        resultados = avaliar_geracao(openai, RAGPipeline(openai, search), casos, args.k)
        imprimir_geracao(resultados)
        saida["geracao"] = resultados

    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    arquivo = os.path.join(RESULTADOS_DIR, f"avaliacao_{datetime.now():%Y%m%d-%H%M}.json")
    with open(arquivo, "w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Resultado salvo em {arquivo}")


if __name__ == "__main__":
    main()
