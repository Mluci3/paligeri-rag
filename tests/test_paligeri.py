"""
Testes offline do Paligeri — NENHUMA chamada ao Azure.

Os clientes Azure são substituídos por fakes (injeção de dependência),
então os testes são gratuitos, rápidos e determinísticos.

Rodar (a partir da raiz do projeto):
    PYTHONPATH=src python -m unittest discover -s tests -v
"""

import json
import os
import tempfile
import unittest

from paligeri.eval import avaliar_retrieval, hit_at_k, reciprocal_rank, resumir
from paligeri.indexer import DEFAULT_DOCS_PATH, Indexer
from paligeri.loader import MAX_CHARS, dividir_texto, gerar_documentos, ler_markdown, slug
from paligeri.rag import RAGPipeline
from paligeri.setup_search import carregar_regras_sinonimos


# ============================================================================
# FAKES
# ============================================================================

class FakeOpenAI:
    def __init__(self, embedding=(0.1, 0.2), resposta="Resposta: ok"):
        self.embedding = list(embedding) if embedding is not None else None
        self.resposta = resposta
        self.contextos = []

    def gerar_embedding(self, texto):
        return self.embedding

    def gerar_resposta(self, query, contexto):
        self.contextos.append(contexto)
        return self.resposta


class FakeSearch:
    """Devolve ids diferentes para busca vetorial e híbrida."""

    def __init__(self, vetorial_ids=("doc2", "doc1"), hibrida_ids=("doc1", "doc2")):
        self.vetorial_ids = vetorial_ids
        self.hibrida_ids = hibrida_ids
        self.chamadas = []

    def buscar(self, embedding, top_k=3, texto=None):
        self.chamadas.append({"top_k": top_k, "texto": texto})
        ids = self.hibrida_ids if texto else self.vetorial_ids
        return [
            {"id": i, "titulo": f"T-{i}", "conteudo": f"C-{i}", "categoria": "X"}
            for i in ids[:top_k]
        ]


# ============================================================================
# LOADER
# ============================================================================

class TestLoader(unittest.TestCase):

    def test_texto_curto_vira_um_chunk(self):
        self.assertEqual(dividir_texto("Texto curto."), ["Texto curto."])

    def test_texto_longo_respeita_limite_e_tem_overlap(self):
        paragrafos = [f"Parágrafo {i}. " + ("palavra " * 120) for i in range(6)]
        chunks = dividir_texto("\n\n".join(paragrafos))
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= MAX_CHARS for c in chunks))
        # overlap: o início do 2º chunk aparece no fim do 1º
        self.assertIn(chunks[1][:40], chunks[0])

    def test_frase_gigante_e_cortada(self):
        chunks = dividir_texto("x" * (MAX_CHARS * 2 + 10))
        self.assertTrue(all(len(c) <= MAX_CHARS for c in chunks))

    def test_slug_remove_acentos_e_simbolos(self):
        self.assertEqual(slug("Manual ANCP – Dor & Sedação"), "manual-ancp-dor-sedacao")

    def test_markdown_preserva_hierarquia_de_titulos(self):
        md = "# Manual\n\n## Dor\n\nTexto dor.\n\n### Avaliação\n\nTexto aval.\n\n## SPIKES\n\nTexto s."
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(md)
        try:
            secoes = ler_markdown(f.name)
        finally:
            os.unlink(f.name)
        self.assertEqual(
            [s.titulos for s in secoes],
            [["Manual", "Dor"], ["Manual", "Dor", "Avaliação"], ["Manual", "SPIKES"]],
        )

    def test_documentos_gerados_passam_na_validacao_do_indexer(self):
        texto = "Conteúdo clínico relevante. " * 10
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(f"# Seção A\n\n{texto}\n\n# Seção B\n\n{texto}")
        try:
            docs = gerar_documentos(f.name, "Teste")
        finally:
            os.unlink(f.name)
        self.assertEqual(len(docs), 2)
        self.assertEqual(len({d["id"] for d in docs}), 2)  # ids únicos
        indexer = Indexer(None, None)
        self.assertTrue(all(indexer._validar_documento(d) for d in docs))

    def test_formato_nao_suportado(self):
        with self.assertRaises(ValueError):
            gerar_documentos("arquivo.docx", "X")


# ============================================================================
# INDEXER
# ============================================================================

class TestIndexer(unittest.TestCase):

    def test_corpus_padrao_e_encontrado_de_qualquer_diretorio(self):
        cwd = os.getcwd()
        os.chdir(tempfile.gettempdir())
        try:
            docs = Indexer(None, None).carregar_documentos(DEFAULT_DOCS_PATH)
        finally:
            os.chdir(cwd)
        self.assertEqual(len(docs), 5)

    def test_documento_sem_campo_obrigatorio_e_rejeitado(self):
        self.assertFalse(Indexer(None, None)._validar_documento({"id": "x", "titulo": "t"}))


# ============================================================================
# RAG
# ============================================================================

class TestRAGPipeline(unittest.TestCase):

    def test_hibrida_envia_texto_da_query(self):
        search = FakeSearch()
        RAGPipeline(FakeOpenAI(), search, hibrida=True).responder("dor?")
        self.assertEqual(search.chamadas[0]["texto"], "dor?")

    def test_vetorial_nao_envia_texto(self):
        search = FakeSearch()
        RAGPipeline(FakeOpenAI(), search, hibrida=False).responder("dor?")
        self.assertIsNone(search.chamadas[0]["texto"])

    def test_contexto_inclui_titulo_e_conteudo(self):
        openai = FakeOpenAI()
        RAGPipeline(openai, FakeSearch()).responder("dor?")
        self.assertIn("Documento 1 - T-doc1:\nC-doc1", openai.contextos[0])

    def test_falha_no_embedding_retorna_none(self):
        self.assertIsNone(RAGPipeline(FakeOpenAI(embedding=None), FakeSearch()).responder("x"))

    def test_sem_documentos_retorna_recusa(self):
        resposta = RAGPipeline(FakeOpenAI(), FakeSearch((), ())).responder("x")
        self.assertIn("Não encontrei informações suficientes", resposta)


# ============================================================================
# AVALIAÇÃO
# ============================================================================

class TestMetricas(unittest.TestCase):

    def test_hit_at_k(self):
        self.assertEqual(hit_at_k(["a", "b", "c"], ["b"], 1), 0.0)
        self.assertEqual(hit_at_k(["a", "b", "c"], ["b"], 2), 1.0)

    def test_reciprocal_rank(self):
        self.assertEqual(reciprocal_rank(["a", "b"], ["a"]), 1.0)
        self.assertEqual(reciprocal_rank(["a", "b"], ["b"]), 0.5)
        self.assertEqual(reciprocal_rank(["a", "b"], ["z"]), 0.0)

    def test_avaliar_retrieval_compara_modos_e_ignora_fora_do_escopo(self):
        casos = [
            {"id": "q1", "tipo": "t", "pergunta": "p", "esperados": ["doc1"]},
            {"id": "q2", "tipo": "fora", "pergunta": "p2", "esperados": []},
        ]
        linhas = avaliar_retrieval(FakeOpenAI(), FakeSearch(), casos, k=2)
        self.assertEqual(len(linhas), 1)
        self.assertEqual(resumir(linhas, "vetorial", 2)["mrr"], 0.5)  # doc1 em 2º
        self.assertEqual(resumir(linhas, "hibrida", 2)["mrr"], 1.0)   # doc1 em 1º

    def test_golden_set_valido(self):
        from paligeri.eval import carregar_golden
        casos = carregar_golden()
        self.assertEqual(len({c["id"] for c in casos}), len(casos))
        with open(DEFAULT_DOCS_PATH, encoding="utf-8") as f:
            ids_corpus = {d["id"] for d in json.load(f)["value"]}
        for caso in casos:  # todo id esperado precisa existir no corpus
            self.assertTrue(set(caso["esperados"]) <= ids_corpus, caso["id"])


class FakeJuiz:
    def __init__(self, notas='{"faithfulness": 4, "relevancy": 5, "justificativa": "x"}'):
        self.notas = notas
        self.chamadas = 0

    def chat_json(self, system, user, max_tokens=300):
        self.chamadas += 1
        return self.notas


class TestJuiz(unittest.TestCase):

    CASOS = [
        {"id": "q1", "tipo": "t", "pergunta": "p", "esperados": ["doc1"]},
        {"id": "q2", "tipo": "fora", "pergunta": "p2", "esperados": []},
    ]

    def test_juiz_externo_e_usado_no_lugar_do_gerador(self):
        from paligeri.eval import avaliar_geracao
        gerador, juiz = FakeOpenAI(), FakeJuiz()
        res = avaliar_geracao(gerador, RAGPipeline(gerador, FakeSearch()),
                              self.CASOS, k=2, juiz=juiz)
        self.assertEqual(juiz.chamadas, 1)  # só o caso dentro do escopo é julgado
        self.assertEqual(res[0]["faithfulness"], 4)
        self.assertEqual(res[0]["docs_contexto"], ["doc1", "doc2"])

    def test_rejulgar_reaproveita_respostas_sem_chamar_o_gerador(self):
        from paligeri.eval import avaliar_geracao
        gerador = FakeOpenAI()
        salvas = {"q1": "resposta antiga",
                  "q2": "Não encontrei informações suficientes nos documentos."}
        res = avaliar_geracao(gerador, RAGPipeline(gerador, FakeSearch()), self.CASOS,
                              k=2, juiz=FakeJuiz(), respostas_salvas=salvas)
        self.assertEqual(gerador.contextos, [])  # gerar_resposta nunca foi chamado
        self.assertEqual(res[0]["resposta"], "resposta antiga")
        self.assertTrue(res[1]["correto"])

    def test_juiz_falhando_seguidamente_aborta(self):
        from paligeri.eval import avaliar_geracao
        casos = [{"id": f"q{i}", "tipo": "t", "pergunta": "p", "esperados": ["doc1"]}
                 for i in range(5)]
        gerador, juiz = FakeOpenAI(), FakeJuiz(notas=None)  # juiz sempre falha
        with self.assertRaises(RuntimeError):
            avaliar_geracao(gerador, RAGPipeline(gerador, FakeSearch()), casos, k=2, juiz=juiz)
        self.assertEqual(juiz.chamadas, 2)  # parou na 2ª falha, não tentou os 5

    def test_json_invalido_do_juiz_nao_quebra(self):
        from paligeri.eval import julgar
        self.assertEqual(julgar(FakeJuiz("isto não é json"), "p", "c", "r"), {})

    def test_carregar_respostas_de_resultado_anterior(self):
        from paligeri.eval import carregar_respostas
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"geracao": [{"id": "q1", "resposta": "R1"}]}, f)
        try:
            self.assertEqual(carregar_respostas(f.name), {"q1": "R1"})
        finally:
            os.unlink(f.name)

    def test_resultado_sem_geracao_e_rejeitado(self):
        from paligeri.eval import carregar_respostas
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"retrieval": []}, f)
        try:
            with self.assertRaises(ValueError):
                carregar_respostas(f.name)
        finally:
            os.unlink(f.name)

    def test_provedor_desconhecido(self):
        from paligeri.config import JuizExterno
        with self.assertRaises(ValueError):
            JuizExterno.from_env("inexistente")

    def test_gemini_usa_endpoint_compativel_openai(self):
        from paligeri.config import JuizExterno
        j = JuizExterno("gemini", "gemini-3.1-pro-preview", "chave-falsa")
        self.assertIn("generativelanguage.googleapis.com", str(j._client.base_url))


class TestParametrosModelo(unittest.TestCase):
    """Parâmetros de geração por tipo de modelo (sem chamar a API)."""

    def _cliente(self, raciocinio: bool):
        from paligeri.config import AzureConfig, AzureOpenAIClient
        cfg = AzureConfig("https://x.openai.azure.com/", "k", "gpt", "emb",
                          "https://s", "k", "idx", modelo_raciocinio=raciocinio)
        return AzureOpenAIClient(cfg)  # só instancia; não faz requisição

    def test_modelo_classico_usa_temperature_e_max_tokens(self):
        p = self._cliente(False)._params_geracao(800)
        self.assertEqual(p, {"temperature": 0.0, "max_tokens": 800})

    def test_modelo_raciocinio_nao_envia_temperature(self):
        p = self._cliente(True)._params_geracao(800)
        self.assertNotIn("temperature", p)
        self.assertNotIn("max_tokens", p)
        self.assertGreater(p["max_completion_tokens"], 800)  # margem p/ raciocínio
        self.assertEqual(p["reasoning_effort"], "low")

    def test_resposta_vazia_vira_none(self):
        from types import SimpleNamespace as NS
        from paligeri.config import AzureOpenAIClient
        vazia = NS(choices=[NS(message=NS(content=""), finish_reason="length")])
        cheia = NS(choices=[NS(message=NS(content="ok"), finish_reason="stop")])
        self.assertIsNone(AzureOpenAIClient._extrair_texto(vazia))
        self.assertEqual(AzureOpenAIClient._extrair_texto(cheia), "ok")


class TestArmadilhas(unittest.TestCase):

    def test_detecta_dose_importada_de_outro_documento(self):
        from paligeri.eval import verificar_proibidos
        resposta = "Para dor intensa, use morfina 2,5-5mg a cada 4 horas."
        self.assertIn("2,5", verificar_proibidos(resposta, ["2.5", "2,5", "5mg"]))

    def test_regex_nao_gera_falso_positivo_ao_citar_unidade(self):
        from paligeri.eval import verificar_proibidos
        correta = "Os documentos não informam a dose de tramadol em mg."
        inventada = "Tramadol 50 mg a cada 8 horas."
        self.assertEqual(verificar_proibidos(correta, [r"re:\d+\s*mg"]), [])
        self.assertEqual(verificar_proibidos(inventada, [r"re:\d+\s*mg"]), [r"re:\d+\s*mg"])

    def test_violacoes_registradas_na_avaliacao(self):
        from paligeri.eval import avaliar_geracao
        casos = [{"id": "a1", "tipo": "armadilha", "pergunta": "dose?",
                  "esperados": ["doc1"], "proibidos": ["2,5"]}]
        gerador = FakeOpenAI(resposta="Morfina 2,5 mg.")
        res = avaliar_geracao(gerador, RAGPipeline(gerador, FakeSearch()), casos,
                              k=2, juiz=FakeJuiz())
        self.assertEqual(res[0]["violacoes"], ["2,5"])

    def test_casos_sem_proibidos_nao_tem_campo_violacoes(self):
        from paligeri.eval import avaliar_geracao
        casos = [{"id": "q1", "tipo": "t", "pergunta": "p", "esperados": ["doc1"]}]
        gerador = FakeOpenAI()
        res = avaliar_geracao(gerador, RAGPipeline(gerador, FakeSearch()), casos,
                              k=2, juiz=FakeJuiz())
        self.assertNotIn("violacoes", res[0])

    def test_respostas_validadas_de_1814_nao_violam_armadilhas_vizinhas(self):
        """Sanidade: as regras não acusam respostas corretas já revisadas."""
        from paligeri.eval import carregar_golden, verificar_proibidos
        caminho = os.path.join(os.path.dirname(DEFAULT_DOCS_PATH), "..",
                               "resultados", "avaliacao_20261122-1814.json")
        if not os.path.exists(caminho):
            self.skipTest("resultado de referência ausente")
        with open(caminho, encoding="utf-8") as f:
            q06 = next(g for g in json.load(f)["geracao"] if g["id"] == "q06")
        a02 = next(c for c in carregar_golden() if c["id"] == "a02")
        self.assertEqual(verificar_proibidos(q06["resposta"], a02["proibidos"]), [])


class TestSinonimos(unittest.TestCase):

    def test_regras_ignoram_comentarios(self):
        regras = carregar_regras_sinonimos()
        self.assertTrue(regras)
        self.assertFalse(any(r.startswith("#") for r in regras))
        self.assertTrue(any("demência" in r and "déficit cognitivo" in r for r in regras))


if __name__ == "__main__":
    unittest.main()
