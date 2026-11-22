# Paligeri RAG: busca clínica fundamentada em Cuidados Paliativos e Geriatria

Pipeline de **Retrieval-Augmented Generation (RAG)** em Azure para responder perguntas clínicas **somente** com base em material de referência (Manual ANCP), citando a fonte e recusando quando a informação não está no corpus.

> ⚠️ Projeto acadêmico de engenharia. **Não é ferramenta de apoio à decisão clínica** e não substitui avaliação profissional.

---

## 1. Problema

Equipes de geriatria e cuidados paliativos consultam manuais longos para dúvidas pontuais, como controle de dor, comunicação de más notícias ou prevenção de lesões por pressão. Um LLM genérico responde rápido, mas **pode inventar doses e condutas**. Em contexto clínico, esse é o risco principal.

**Objetivo:** respostas rápidas que sejam **rastreáveis até um documento** e que **recusem** em vez de inventar.

## 2. Arquitetura

```mermaid
flowchart LR
    subgraph Ingestão [Ingestão — offline]
        A[.md / .pdf] --> B[loader.py<br/>chunking]
        B --> C[corpus.json<br/>revisão humana]
        C --> D[indexer.py<br/>embeddings]
    end
    D --> E[(Azure AI Search<br/>BM25 + HNSW<br/>synonym map)]
    subgraph Inferência
        Q[Pergunta] --> R[rag.py]
        R -->|embedding + texto| E
        E -->|top-k via RRF| R
        R -->|contexto + regras| G[Azure OpenAI<br/>gpt-4o]
        G --> S[Resposta + Fonte + Limitação]
    end
    E -.-> V[eval.py<br/>Hit@k · MRR · LLM-as-judge]
```

| Módulo | Responsabilidade |
|---|---|
| [`loader.py`](src/paligeri/loader.py) | Lê `.md`/`.pdf` e divide em chunks respeitando seções, com overlap |
| [`indexer.py`](src/paligeri/indexer.py) | Gera embeddings e envia para o índice (idempotente por `id`) |
| [`rag.py`](src/paligeri/rag.py) | Busca híbrida, montagem do contexto e geração |
| [`config.py`](src/paligeri/config.py) | Configuração imutável e clientes Azure (injeção de dependência) |
| [`setup_search.py`](src/paligeri/setup_search.py) | Schema do índice e synonym map **como código** |
| [`eval.py`](src/paligeri/eval.py) | Avaliação de retrieval e de geração |

## 3. Decisões de engenharia e trade-offs

**Busca híbrida (BM25 + vetorial, fundidas por RRF).** A busca vetorial acerta paráfrases ("notícia ruim" → SPIKES), mas é fraca com termos exatos (OncPal, Braden). A BM25 é o contrário. O RRF combina as duas listas pela posição de cada documento. *Trade-off:* uma chamada de busca um pouco mais pesada, sem custo extra de LLM.

**Synonym map clínico ([`data/sinonimos.txt`](data/sinonimos.txt)).** O corpus usa "déficit cognitivo" e o usuário pergunta "demência". Nem o embedding nem a BM25 garantem essa ponte. Os sinônimos são expandidos na consulta, **sem reindexar**, e ficam num arquivo de texto que uma pessoa da área clínica pode revisar. *Limitação:* só valem para a parte textual da busca.

**Chunking por seção com overlap (1500 / 200 caracteres).** Respeita a estrutura do manual, e cada chunk leva o caminho de seções no título, o que melhora o embedding e a citação. O loader gera um JSON **revisável antes de pagar** por embeddings.

**Groundedness no prompt.** `temperature=0`, uso exclusivo do contexto, formato fixo (Resposta / Fonte / Limitação) e uma frase de recusa padronizada, que torna a recusa **verificável automaticamente** na avaliação.

**Testabilidade por injeção de dependência.** O pipeline recebe os clientes prontos, então os 19 testes rodam com fakes, sem custo e sem rede.

## 4. Avaliação

O golden set ([`data/avaliacao.json`](data/avaliacao.json)) tem 10 casos escolhidos para testar cada decisão acima: perguntas validadas, sinônimo, termo exato, paráfrase e uma pergunta **fora do escopo**, que deve ser recusada.

| Camada | Métricas | Custo |
|---|---|---|
| Retrieval (vetorial × híbrida) | Hit@1, Hit@3, MRR | só embeddings |
| Geração (`--julgar`) | faithfulness, relevancy (LLM-as-judge), taxa de recusa correta | chamadas GPT |

**Resultados:** *pendente da primeira execução no ambiente Azure.*

> Com só 5 documentos, Hit@3 é pouco informativo (até o acaso acerta com frequência). **Hit@1 e MRR** são os números que importam até o corpus crescer.

## 5. Como rodar

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # preencha endpoints e chaves

export PYTHONPATH=src
python -m unittest discover -s tests -v       # testes offline (sem Azure)

python -m paligeri.setup_search verificar     # schema + nº de docs (leitura)
python -m paligeri.setup_search sinonimos     # aplica synonym map
python -m paligeri.indexer                    # indexa o corpus padrão
python -m paligeri.rag                        # modo interativo
python -m paligeri.eval [--julgar]            # avaliação
```

Para ingerir novos documentos, coloque o arquivo em `data/raw/` (fora do Git por direitos autorais) e rode:

```bash
python -m paligeri.loader data/raw/manual.md --categoria "Manual ANCP" --saida data/corpus_ancp.json
python -m paligeri.indexer --file data/corpus_ancp.json
```

## 6. Limitações conhecidas

- Corpus pequeno (5 documentos): as métricas ainda têm baixo poder estatístico.
- Usa `text-embedding-ada-002`. Modelos mais novos (`text-embedding-3-*`) tendem a ser melhores, mas exigem reindexar.
- Autenticação por chave. Em produção, o ideal é Managed Identity + Key Vault.
- O LLM-as-judge usa o mesmo modelo que gera as respostas, o que pode introduzir viés de autoavaliação.
- A extração de PDF é feita por página. Converter para Markdown antes preserva melhor a estrutura.

## 7. Próximos passos

- [ ] Ingerir o Manual ANCP completo e ampliar o golden set
- [ ] Semantic ranker do Azure AI Search como reranqueamento
- [ ] Comparar o loader próprio com Blob Storage + Indexer nativo + skillset (Text Split / Embedding)
- [ ] Managed Identity no lugar de chaves
- [ ] API (FastAPI) e observabilidade (latência, tokens, custo por consulta)

## Stack

Python 3.11 · Azure OpenAI (gpt-4o, text-embedding-ada-002) · Azure AI Search (BM25, HNSW, RRF, synonym maps) · unittest
