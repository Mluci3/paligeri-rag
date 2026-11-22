# Paligeri RAG — Queries Validadas (Mini-Lab v1)

**Data:** 11/05/2026  
**Ambiente:** Azure OpenAI (East US 2) + Azure AI Search (East US)  
**Índice:** `docs-geriatric-test`  
**Modelo GPT:** `gpt-4o-study` (gpt-4o)  
**Modelo Embedding:** `text-embedding-ada-002`  
**Corpus:** 5 documentos (Manual ANCP — base)

---

## Queries Validadas

| # | Pergunta | Categoria | Status |
|---|----------|-----------|--------|
| 1 | Como controlar a dor em pacientes em cuidados paliativos? | Controle de Sintomas | ✅ |
| 2 | Quais são as etapas do protocolo SPIKES? | Comunicação | ✅ |
| 3 | Como prevenir lesões por pressão em pacientes acamados? | Cuidados de Enfermagem | ✅ |
| 4 | Como controlar a dispneia em pacientes terminais? | Fim de Vida | ✅ |
| 5 | O que é deprescrição e quando aplicar em idosos? | Farmacoterapia | ✅ |

---

## Queries com Limitação Conhecida

| # | Pergunta | Motivo |
|---|----------|--------|
| 1 | Quais escalas avaliam dor em pacientes com demência? | Termo "demência" não está no corpus — documento usa "déficit cognitivo" |

---

## Corpus Atual (5 documentos)

| ID | Título | Categoria |
|----|--------|-----------|
| doc1 | Controle de sintomas: dor em cuidados paliativos | Controle de Sintomas |
| doc2 | Comunicação em cuidados paliativos: protocolo SPIKES | Comunicação |
| doc3 | Prevenção de lesões por pressão: Escala de Braden | Cuidados de Enfermagem |
| doc4 | Cuidados ao final da vida: controle de sintomas e sedação paliativa | Fim de Vida |
| doc5 | Deprescrição e farmacoterapia em cuidados paliativos | Farmacoterapia |

---

## Próximos Passos (backlog)

- [x] Sinônimos clínicos via synonym map (`data/sinonimos.txt` + `setup_search.py`) — *aplicar no Azure*
- [x] Busca híbrida (BM25 + vetorial/RRF) em `rag.py`
- [ ] Adicionar documentos do Manual ANCP completo (formato `.md`)
- [x] Implementar `loader.py` para ingestão de `.md` e `.pdf`
- [ ] Migrar para Azure Blob Storage + AI Search Indexer nativo (AI-102)
- [x] Avaliação com métricas (`eval.py`: Hit@k, MRR, faithfulness, relevancy) — *rodar no Azure*