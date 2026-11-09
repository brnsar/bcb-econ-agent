from econ_agent.embeddings import HashingEmbedder
from econ_agent.text import chunk_text, html_to_text
from econ_agent.vector_store import VectorStore


def test_html_to_text_keeps_paragraphs_and_strips_tags():
    html = '<div><p>​Primeiro <em>parágrafo</em>.</p><p>Segundo&nbsp;parágrafo.</p><script>x()</script></div>'
    assert html_to_text(html) == "Primeiro parágrafo.\n\nSegundo parágrafo."


def test_chunk_text_respects_limit_and_overlaps():
    text = "\n\n".join(f"Parágrafo {i} " + "palavra " * 20 for i in range(10))
    chunks = chunk_text(text, max_chars=400, overlap_paragraphs=1)
    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)
    # consecutive chunks share a paragraph
    assert chunks[0].split("\n\n")[-1] == chunks[1].split("\n\n")[0]


def test_chunk_text_splits_long_paragraph_on_sentences():
    para = " ".join(f"Frase número {i}." for i in range(100))
    chunks = chunk_text(para, max_chars=200)
    assert all(len(c) <= 200 for c in chunks)
    assert "Frase número 99." in chunks[-1]


def test_hashing_embedder_is_normalized_and_deterministic():
    emb = HashingEmbedder(dim=64)
    a = emb.embed(["inflação de serviços", "inflação de serviços"])
    assert abs(float((a[0] ** 2).sum()) - 1.0) < 1e-5
    assert (a[0] == a[1]).all()


def test_vector_store_search_filter_and_roundtrip(tmp_path, index, embedder):
    hits = index.search(embedder.embed(["conflitos no Oriente Médio e petróleo"])[0], top_k=1)
    assert hits[0].record["meeting"] == 282

    only_277 = index.search(embedder.embed(["petróleo"])[0], top_k=3, where=lambda r: r["meeting"] == 277)
    assert [h.record["meeting"] for h in only_277] == [277]

    index.save(tmp_path / "idx")
    loaded = VectorStore.load(tmp_path / "idx")
    assert len(loaded) == len(index) and loaded.embedder_name == index.embedder_name
