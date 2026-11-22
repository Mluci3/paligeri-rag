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
    PYTHONPATH=src python -m paligeri.eval --julgar     # + geração (juiz = gerador)

    # Juiz de outro provedor (reduz viés de autoavaliação):
    PYTHONPATH=src python -m paligeri.eval --julgar --juiz gemini --pausa 13

    # Calibração: MESMAS respostas de uma execução anterior, só o juiz muda:
    PYTHONPATH=src python -m paligeri.eval --rejulgar resultados/<arquivo>.json --juiz gemini
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from typing import Optional

from paligeri.config import (
    PROVEDORES_JUIZ, ROOT_DIR, AzureConfig, AzureOpenAIClient,
    AzureSearchClient, JuizExterno,
)
from paligeri.rag import RAGPipeline

log = logging.getLogger(__name__)

GOLDEN_PATH = os.path.join(ROOT_DIR, "data", "avaliacao.json")
RESULTADOS_DIR = os.path.join(ROOT_DIR, "resultados")
FRASE_RECUSA = "Não encontrei informações suficientes"
MAX_FALHAS_JUIZ = 2

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

def julgar(juiz, pergunta: str, contexto: str, resposta: str) -> dict:
    """
    Pede notas ao juiz. `juiz` é qualquer objeto com chat_json(system, user):
    AzureOpenAIClient (mesmo modelo do gerador) ou JuizExterno (outro provedor).
    """
    bruto = juiz.chat_json(
        PROMPT_JUIZ,
        f"CONTEXTO:\n{contexto}\n\nPERGUNTA: {pergunta}\n\nRESPOSTA:\n{resposta}",
    )
    try:
        return json.loads(bruto or "{}")
    except json.JSONDecodeError:
        log.warning("⚠️ Juiz devolveu JSON inválido.")
        return {}


def avaliar_geracao(gerador: AzureOpenAIClient, pipeline: RAGPipeline,
                    casos: list[dict], k: int, juiz=None, pausa: float = 0.0,
                    respostas_salvas: Optional[dict] = None) -> list[dict]:
    """
    Gera (ou reaproveita) respostas e as julga.

    respostas_salvas: {id_caso: resposta} de uma execução anterior.
        Permite trocar SÓ o juiz mantendo as MESMAS respostas, isolando
        uma variável (o gerador não é determinístico).
    pausa: segundos entre julgamentos (limites de requisição em planos free).
    """
    juiz = juiz or gerador
    resultados = []
    falhas_seguidas = 0
    for i, caso in enumerate(casos):
        docs = pipeline.recuperar(caso["pergunta"], top_k=k) or []
        contexto = pipeline.montar_contexto(docs)
        if respostas_salvas and caso["id"] in respostas_salvas:
            resposta = respostas_salvas[caso["id"]]
        else:
            resposta = gerador.gerar_resposta(caso["pergunta"], contexto) or ""
        recusou = FRASE_RECUSA in resposta
        linha = {
            "id": caso["id"], "tipo": caso["tipo"], "resposta": resposta,
            "recusou": recusou, "docs_contexto": [d["id"] for d in docs],
        }

        if not caso["esperados"]:
            # Fora do escopo: o correto é recusar. Não faz sentido julgar fidelidade.
            linha["correto"] = recusou
        else:
            if pausa and i > 0:
                time.sleep(pausa)
            notas = julgar(juiz, caso["pergunta"], contexto, resposta)
            # Fail fast: juiz falhando seguidamente (cota zerada, modelo
            # inexistente) produziria um resultado inválido cheio de "?".
            falhas_seguidas = 0 if notas else falhas_seguidas + 1
            if falhas_seguidas >= MAX_FALHAS_JUIZ:
                raise RuntimeError(
                    f"Juiz falhou {falhas_seguidas}x seguidas; avaliação abortada "
                    "sem salvar. Veja o erro acima."
                )
            linha.update(notas)
        resultados.append(linha)
    return resultados


def carregar_respostas(path: str) -> dict:
    """Lê as respostas de um resultado anterior: {id_caso: resposta}."""
    with open(path, encoding="utf-8") as f:
        dados = json.load(f)
    if "geracao" not in dados:
        raise ValueError(f"{path} não tem respostas geradas (rode com --julgar).")
    return {g["id"]: g["resposta"] for g in dados["geracao"]}


def imprimir_geracao(resultados: list[dict], nome_juiz: str = "") -> None:
    print(f"\n🧑‍⚖️ GERAÇÃO — LLM-as-judge (1 a 5) | juiz: {nome_juiz}\n")
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
    parser.add_argument("--juiz", choices=["azure", *PROVEDORES_JUIZ], default="azure",
                        help="Quem julga: azure (mesmo modelo do gerador), gemini ou openai")
    parser.add_argument("--rejulgar", metavar="ARQUIVO",
                        help="Reaproveita as respostas de um resultado anterior "
                             "(troca só o juiz). Implica --julgar.")
    parser.add_argument("--pausa", type=float, default=0.0,
                        help="Segundos entre julgamentos (ex.: 13 no plano free do Gemini Pro)")
    args = parser.parse_args()
    if args.rejulgar:
        args.julgar = True

    try:
        config = AzureConfig.from_env()
        juiz = None if args.juiz == "azure" else JuizExterno.from_env(args.juiz)
        respostas = carregar_respostas(args.rejulgar) if args.rejulgar else None
    except (EnvironmentError, ValueError, OSError) as e:
        log.error(str(e))
        return
    nome_juiz = (f"{juiz.provedor}/{juiz.modelo}" if juiz
                 else f"azure/{config.deployment_gpt}")

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
        try:
            resultados = avaliar_geracao(
                openai, RAGPipeline(openai, search), casos, args.k,
                juiz=juiz, pausa=args.pausa, respostas_salvas=respostas,
            )
        except RuntimeError as e:
            log.error("❌ %s", e)
            return
        imprimir_geracao(resultados, nome_juiz)
        saida["juiz"] = nome_juiz
        saida["respostas_de"] = args.rejulgar
        saida["geracao"] = resultados

    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    arquivo = os.path.join(RESULTADOS_DIR, f"avaliacao_{datetime.now():%Y%m%d-%H%M}.json")
    with open(arquivo, "w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Resultado salvo em {arquivo}")


if __name__ == "__main__":
    main()
