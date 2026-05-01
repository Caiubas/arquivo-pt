"""
Pipeline NLP — BERTopic-like (Embeddings + UMAP + HDBSCAN + c-TF-IDF)
=====================================================================

Melhorias:
- Clustering com HDBSCAN (detecta k automaticamente + outliers)
- Redução com UMAP (preserva estrutura semântica)
- Interpretação com c-TF-IDF (melhor que TF-IDF clássico)
- Pronto para integração com Streamlit

Executar:
    python nlp_pipeline_bertopic.py --input arquivo_sns_data.json
"""

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

# externos
import umap
import hdbscan

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
MIN_TEXT_LEN = 80
MAX_TEXT_LEN = 2000
TOP_TERMS = 10

STOPWORDS_PT = {
    "de","a","o","que","e","do","da","em","um","para","com","uma","os","no",
    "se","na","por","mais","as","dos","como","mas","ao","ele","das","seu",
    "sua","ou","quando","muito","nos","já","eu","também","só","pelo","pela",
    "até","isso","ela","entre","depois","sem","mesmo",
    "serviço","nacional","saúde","sns","portugal"
}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────

def carregar_textos(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    if isinstance(data, dict) and "records" in data:
        data = data["records"]

    df = pd.DataFrame(data)

    df = df.rename(columns={
        "content": "texto",
        "title": "titulo",
        "date": "data"
    })

    if "texto" not in df:
        raise ValueError("Campo 'texto' não encontrado")

    df["texto"] = df["texto"].astype(str)
    df["id"] = range(len(df))

    return df

# ─────────────────────────────────────────────────────────────
# 2. CLEAN
# ─────────────────────────────────────────────────────────────

def limpar_texto(t):
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:MAX_TEXT_LEN]

def preparar(df):
    df = df.copy()
    df["texto_limpo"] = df["texto"].apply(limpar_texto)

    df = df[df["texto_limpo"].str.len() >= MIN_TEXT_LEN]
    df = df.reset_index(drop=True)

    log.info(f"Corpus: {len(df)} docs")
    return df

# ─────────────────────────────────────────────────────────────
# 3. EMBEDDINGS
# ─────────────────────────────────────────────────────────────

def gerar_embeddings(textos, modelo):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(modelo)

    emb = model.encode(
        textos,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return emb

# ─────────────────────────────────────────────────────────────
# 4. UMAP
# ─────────────────────────────────────────────────────────────

def reduzir_umap(embeddings):
    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )

    return reducer.fit_transform(embeddings)

# ─────────────────────────────────────────────────────────────
# 5. HDBSCAN
# ─────────────────────────────────────────────────────────────

def clusterizar(embeddings):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10,
        metric="euclidean",
        cluster_selection_method="eom"
    )

    labels = clusterer.fit_predict(embeddings)

    log.info(f"Clusters encontrados: {len(set(labels)) - (1 if -1 in labels else 0)}")
    log.info(f"Outliers: {(labels == -1).sum()}")

    return labels

# ─────────────────────────────────────────────────────────────
# 6. c-TF-IDF
# ─────────────────────────────────────────────────────────────

def c_tf_idf(docs_per_cluster):
    vectorizer = CountVectorizer(
        stop_words=list(STOPWORDS_PT),
        ngram_range=(1,2)
    )

    X = vectorizer.fit_transform(docs_per_cluster)

    # frequência por cluster
    tf = X.toarray()

    # normalização (c-TF-IDF)
    tf = normalize(tf, norm="l1", axis=1)

    idf = np.log((1 + len(docs_per_cluster)) / (1 + (tf > 0).sum(axis=0))) + 1

    ctfidf = tf * idf

    return ctfidf, vectorizer

# ─────────────────────────────────────────────────────────────
# 7. INTERPRETAÇÃO
# ─────────────────────────────────────────────────────────────

def interpretar(df, labels):
    df = df.copy()
    df["cluster"] = labels

    clusters = {}

    docs_por_cluster = []
    ids = []

    for cid in sorted(set(labels)):
        if cid == -1:
            continue

        sub = df[df["cluster"] == cid]

        docs_por_cluster.append(" ".join(sub["texto_limpo"]))
        ids.append(cid)

    ctfidf, vectorizer = c_tf_idf(docs_por_cluster)
    termos = vectorizer.get_feature_names_out()

    for i, cid in enumerate(ids):
        scores = ctfidf[i]
        top_idx = scores.argsort()[::-1][:TOP_TERMS]

        top_terms = [termos[j] for j in top_idx]

        clusters[int(cid)] = {
            "top_termos": top_terms,
            "n_documentos": int((labels == cid).sum()),
            "rotulo": " · ".join(top_terms[:3])
        }

    return clusters

# ─────────────────────────────────────────────────────────────
# 8. EXPORT
# ─────────────────────────────────────────────────────────────

def exportar(df, labels, clusters, coords_2d, output):
    from datetime import datetime

    df_out = df.copy()
    df_out["cluster_id"] = labels
    df_out["cluster_rotulo"] = [
        clusters.get(int(l), {}).get("rotulo", "Outlier")
        if l != -1 else "Outlier"
        for l in labels
    ]

    # coordenadas UMAP (IMPORTANTE pro teu mapa)
    if coords_2d is not None:
        df_out["umap_x"] = coords_2d[:, 0]
        df_out["umap_y"] = coords_2d[:, 1]

    result = {
        "metadata": {
            "total_documentos": len(df_out),
            "k_clusters": len(clusters),
            "gerado_em": datetime.utcnow().isoformat()
        },
        "clusters": clusters,
        "documentos": df_out[[
            "id", "titulo", "data", "categoria", "url",
            "cluster_id", "cluster_rotulo",
            "umap_x", "umap_y"
        ]].to_dict("records")
    }

    Path(output).write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run(input_path, output_path, modelo):
    log.info("Carregando dados...")
    df = carregar_textos(input_path)

    log.info("Limpando...")
    df = preparar(df)

    log.info("Embeddings...")
    emb = gerar_embeddings(df["texto_limpo"].tolist(), modelo)

    log.info("UMAP...")
    emb_umap = reduzir_umap(emb)

    log.info("Clustering...")
    labels = clusterizar(emb_umap)

    log.info("Interpretando...")

    clusters = interpretar(df, labels)
    coords_2d = reduzir_umap(emb)[:, :2]  # garante 2D
    exportar(df, labels, clusters, coords_2d, output_path)

# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run("arquivo_sns_data.json", "nlp_resultados.json", DEFAULT_MODEL)

if __name__ == "__main_":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="nlp_resultados.json")
    parser.add_argument("--modelo", default=DEFAULT_MODEL)

    args = parser.parse_args()

    run(args.input, args.output, args.modelo)