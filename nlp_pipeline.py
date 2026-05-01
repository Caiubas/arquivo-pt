"""
Pipeline NLP — Embeddings + Clustering para textos do SNS
==========================================================
Fluxo:
  1. Carrega textos do JSON produzido pelo arquivo_pt_pipeline.py
  2. Gera embeddings semânticos com sentence-transformers
  3. Reduz dimensionalidade (PCA → UMAP opcional)
  4. Determina k ótimo via Silhouette Score
  5. Aplica KMeans e interpreta temas por TF-IDF
  6. Exporta resultados em JSON e relatório legível

Instalar dependências:
    pip install sentence-transformers scikit-learn umap-learn numpy pandas

Executar:
    # Com ficheiro do pipeline:
    python nlp_pipeline.py --input arquivo_sns_data.json

    # Com ficheiro de eventos de exemplo:
    python nlp_pipeline.py --input eventos_sns.json

    # Ajustar número de clusters manualmente:
    python nlp_pipeline.py --input arquivo_sns_data.json --clusters 6
"""

import argparse
import json
import logging
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
# Modelo multilingue compacto — bom para português sem precisar de fine-tuning
DEFAULT_MODEL    = "paraphrase-multilingual-MiniLM-L12-v2"
MIN_TEXT_LEN     = 80       # descarta textos demasiado curtos
MAX_TEXT_LEN     = 2_000    # trunca textos longos para economizar memória
K_MIN, K_MAX     = 2, 10    # intervalo de k avaliado pelo Silhouette
PCA_COMPONENTS   = 50       # dimensões após PCA (antes do KMeans)
TOP_TERMS        = 10       # termos por cluster no relatório
STOPWORDS_PT = {            # stopwords básicas — evita instalar nltk
    "de","a","o","que","e","do","da","em","um","para","com","uma","os","no",
    "se","na","por","mais","as","dos","como","mas","ao","ele","das","seu",
    "sua","ou","quando","muito","nos","já","eu","também","só","pelo","pela",
    "até","isso","ela","entre","depois","sem","mesmo","aos","seus","quem",
    "nas","me","esse","eles","você","essa","num","nem","suas","meu","às",
    "minha","numa","pelos","pelas","este","fosse","dele","tu","te","vocês",
    "vos","lhes","meus","minhas","teu","tua","teus","tuas","nosso","nossa",
    "nossos","nossas","dela","delas","deles","lhe","este","esta","estes",
    "estas","isto","aquele","aquela","aqueles","aquelas","aquilo","lo","la",
    "los","las","foi","são","ser","ter","tem","há","esta","está","pelo",
    "serviço","nacional","saúde","sns","portugal","português","portuguesa",
    "hospital","hospitais","paciente","pacientes",
}


# ---------------------------------------------------------------------------
# 1. CARREGAMENTO E NORMALIZAÇÃO DO JSON
# ---------------------------------------------------------------------------

def _extrair_registos(dados) -> list[dict]:
    """Aceita lista simples ou envelope {metadata, records}."""
    if isinstance(dados, list):
        return dados
    if isinstance(dados, dict):
        if "records" in dados:
            return dados["records"]
        for v in dados.values():
            if isinstance(v, list) and v:
                return v
    raise ValueError("Formato JSON não reconhecido. Esperado lista ou {'records': [...]}")


def carregar_textos(caminho: str) -> pd.DataFrame:
    """
    Lê o JSON e devolve DataFrame com colunas padronizadas:
      id, texto, titulo, data, categoria, url
    """
    with open(caminho, encoding="utf-8") as f:
        dados = json.load(f)

    registos = _extrair_registos(dados)
    df = pd.DataFrame(registos)

    # Normaliza nomes de campos entre os dois formatos (pipeline / eventos_sns.json)
    renomear = {
        "title":    "titulo",
        "content":  "texto",
        "date":     "data",
        "keyword":  "categoria",
        "originalURL": "url",
    }
    df = df.rename(columns={k: v for k, v in renomear.items() if k in df.columns})

    # Campo de texto principal: "texto" > "descricao" > "snippet"
    if "texto" not in df.columns:
        for fallback in ("descricao", "snippet"):
            if fallback in df.columns:
                df["texto"] = df[fallback]
                break

    if "texto" not in df.columns:
        raise ValueError("Nenhum campo de texto encontrado (esperado: content, descricao, ou snippet).")

    for campo in ("titulo", "categoria", "url", "data"):
        if campo not in df.columns:
            df[campo] = ""

    df["id"] = range(len(df))
    df = df[["id", "titulo", "data", "categoria", "url", "texto"]].copy()
    return df


# ---------------------------------------------------------------------------
# 2. PRÉ-PROCESSAMENTO DE TEXTO
# ---------------------------------------------------------------------------

def limpar_texto(texto: str) -> str:
    """Remove artefactos HTML residuais, URLs e normaliza espaços."""
    if not isinstance(texto, str):
        return ""
    texto = re.sub(r"<[^>]+>", " ", texto)             # tags HTML
    texto = re.sub(r"https?://\S+", " ", texto)         # URLs
    texto = re.sub(r"[^\w\s\-áéíóúâêîôûãõàèìòùçÁÉÍÓÚÂÊÎÔÛÃÕÀÈÌÒÙÇ]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto[:MAX_TEXT_LEN]


def preparar_corpus(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Filtra registos inválidos e devolve DataFrame limpo + lista de textos.
    """
    df = df.copy()
    df["texto_limpo"] = df["texto"].apply(limpar_texto)

    # Descarta textos demasiado curtos para produzir embeddings úteis
    mascara = df["texto_limpo"].str.len() >= MIN_TEXT_LEN
    n_descartados = (~mascara).sum()
    if n_descartados:
        log.warning("  %d registos descartados (texto < %d chars)", n_descartados, MIN_TEXT_LEN)

    df = df[mascara].reset_index(drop=True)
    df["id"] = range(len(df))

    if df.empty:
        raise ValueError("Corpus vazio após filtragem. Verifique o ficheiro de entrada.")

    log.info("  Corpus final: %d documentos", len(df))
    return df, df["texto_limpo"].tolist()


# ---------------------------------------------------------------------------
# 3. EMBEDDINGS COM SENTENCE-TRANSFORMERS
# ---------------------------------------------------------------------------

def gerar_embeddings(textos: list[str], modelo_nome: str = DEFAULT_MODEL) -> np.ndarray:
    """
    Gera embeddings semânticos com sentence-transformers.

    Na primeira execução descarrega o modelo (~120 MB para MiniLM).
    Execuções seguintes usam a cache local em ~/.cache/huggingface/
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.error("sentence-transformers não instalado.")
        log.error("  Execute: pip install sentence-transformers")
        sys.exit(1)

    log.info("  Modelo: %s", modelo_nome)
    log.info("  (primeira execução descarrega o modelo — aguarde)")

    modelo = SentenceTransformer(modelo_nome)
    embeddings = modelo.encode(
        textos,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine similarity equivale a dot product
    )
    log.info("  Embeddings: shape=%s  dtype=%s", embeddings.shape, embeddings.dtype)
    return embeddings


# ---------------------------------------------------------------------------
# 4. REDUÇÃO DE DIMENSIONALIDADE
# ---------------------------------------------------------------------------

def reduzir_pca(embeddings: np.ndarray, n_components: int = PCA_COMPONENTS) -> np.ndarray:
    """
    PCA para reduzir dimensões antes do KMeans.

    KMeans sofre de "curse of dimensionality" em espaços de 384+ dims.
    PCA→50 dims retém >90% da variância na maioria dos corpora.
    """
    n_components = min(n_components, embeddings.shape[0] - 1, embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    reduzido = pca.fit_transform(embeddings)
    variancia = pca.explained_variance_ratio_.sum()
    log.info("  PCA: %d → %d dims  (variância retida: %.1f%%)",
             embeddings.shape[1], n_components, variancia * 100)
    return reduzido


def reduzir_umap(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray | None:
    """
    UMAP para redução a 2D (apenas visualização).
    Retorna None se umap-learn não estiver instalado.
    """
    try:
        import umap  # noqa: F401
        reducer = umap.UMAP(n_components=n_components, random_state=42, verbose=False)
        coords = reducer.fit_transform(embeddings)
        log.info("  UMAP 2D calculado para visualização")
        return coords
    except ImportError:
        log.info("  umap-learn não instalado — visualização 2D ignorada")
        return None


# ---------------------------------------------------------------------------
# 5. DETERMINAÇÃO DO K ÓTIMO
# ---------------------------------------------------------------------------

def escolher_k(embeddings: np.ndarray, k_min: int = K_MIN, k_max: int = K_MAX) -> tuple[int, dict]:
    """
    Avalia KMeans para k em [k_min, k_max] e devolve o k com maior
    Silhouette Score — métrica que não precisa de rótulos externos.

    Também devolve o dicionário de scores para diagnóstico.
    """
    n_amostras = embeddings.shape[0]
    k_max = min(k_max, n_amostras - 1)
    k_min = min(k_min, k_max)

    scores: dict[int, float] = {}
    log.info("  Avaliando k de %d a %d ...", k_min, k_max)

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, sample_size=min(1000, n_amostras))
        scores[k] = round(float(score), 4)
        log.info("    k=%d  silhouette=%.4f", k, score)

    k_otimo = max(scores, key=scores.__getitem__)
    log.info("  → k ótimo: %d  (silhouette=%.4f)", k_otimo, scores[k_otimo])
    return k_otimo, scores


# ---------------------------------------------------------------------------
# 6. CLUSTERING KMEANS
# ---------------------------------------------------------------------------

def aplicar_kmeans(embeddings: np.ndarray, k: int) -> tuple[KMeans, np.ndarray]:
    """Aplica KMeans com k fixo e devolve modelo + labels."""
    km = KMeans(n_clusters=k, random_state=42, n_init="auto", max_iter=500)
    labels = km.fit_predict(embeddings)
    log.info("  KMeans k=%d  inertia=%.2f", k, km.inertia_)
    return km, labels


# ---------------------------------------------------------------------------
# 7. INTERPRETAÇÃO DE TEMAS POR TF-IDF
# ---------------------------------------------------------------------------

def interpretar_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    top_n: int = TOP_TERMS,
) -> dict[int, dict]:
    """
    Para cada cluster:
      - Concatena todos os textos do cluster
      - Aplica TF-IDF para extrair os termos mais representativos
      - Recolhe exemplos de títulos (até 3)
      - Gera um rótulo automático com os 3 termos principais

    Retorna dicionário cluster_id → metadados.
    """
    df = df.copy()
    df["cluster"] = labels

    # TF-IDF sobre o corpus completo para penalizar termos globalmente frequentes
    vectorizer = TfidfVectorizer(
        min_df=1,
        max_df=0.95,
        ngram_range=(1, 2),       # uni e bigramas
        stop_words=list(STOPWORDS_PT),
        token_pattern=r"(?u)\b[a-záéíóúâêîôûãõàèìòùçA-ZÁÉÍÓÚÂÊÎÔÛÃÕÀÈÌÒÙÇ]{3,}\b",
    )
    vectorizer.fit(df["texto_limpo"].tolist())
    termos = vectorizer.get_feature_names_out()

    clusters: dict[int, dict] = {}

    for cluster_id in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cluster_id]
        corpus_cluster = " ".join(sub["texto_limpo"].tolist())

        # Score TF-IDF do cluster agregado
        vec = vectorizer.transform([corpus_cluster]).toarray()[0]
        top_idx = vec.argsort()[::-1][:top_n]
        top_termos = [termos[i] for i in top_idx if vec[i] > 0]

        # Exemplos de títulos
        titulos = (
            sub["titulo"]
            .replace("", pd.NA)
            .dropna()
            .head(3)
            .tolist()
        )

        # Rótulo automático: 3 primeiros termos em maiúsculas
        rotulo = " · ".join(t.title() for t in top_termos[:3]) or f"Cluster {cluster_id}"

        clusters[int(cluster_id)] = {
            "rotulo":         rotulo,
            "n_documentos":   int(len(sub)),
            "top_termos":     top_termos,
            "exemplos_titulos": titulos,
            "ids_documentos": sub["id"].tolist(),
        }

    return clusters


# ---------------------------------------------------------------------------
# 8. EXPORTAÇÃO DE RESULTADOS
# ---------------------------------------------------------------------------

def exportar_resultados(
    df: pd.DataFrame,
    labels: np.ndarray,
    clusters: dict,
    silhouette_scores: dict,
    k_otimo: int,
    coords_2d: np.ndarray | None,
    caminho_saida: str,
) -> None:
    """Guarda o JSON estruturado com todos os resultados."""
    from datetime import datetime, timezone

    # Documentos com cluster atribuído
    df_out = df[["id", "titulo", "data", "categoria", "url"]].copy()
    df_out["cluster_id"]    = labels.tolist()
    df_out["cluster_rotulo"] = [clusters[int(l)]["rotulo"] for l in labels]
    if coords_2d is not None:
        df_out["umap_x"] = coords_2d[:, 0].round(4).tolist()
        df_out["umap_y"] = coords_2d[:, 1].round(4).tolist()

    output = {
        "metadata": {
            "total_documentos": len(df),
            "k_clusters":       k_otimo,
            "silhouette_scores": {str(k): v for k, v in silhouette_scores.items()},
            "silhouette_otimo": silhouette_scores[k_otimo],
            "gerado_em":        datetime.now(timezone.utc).isoformat(),
        },
        "clusters": clusters,
        "documentos": df_out.to_dict(orient="records"),
    }

    Path(caminho_saida).write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("  Resultados guardados: %s", caminho_saida)


# ---------------------------------------------------------------------------
# 9. RELATÓRIO LEGÍVEL NO TERMINAL
# ---------------------------------------------------------------------------

SEPARADOR = "─" * 60

def imprimir_relatorio(clusters: dict, silhouette_scores: dict, k_otimo: int) -> None:
    """Imprime relatório formatado no terminal."""
    print(f"\n{'═' * 60}")
    print(f"  ANÁLISE TEMÁTICA — {k_otimo} CLUSTERS")
    print(f"  Silhouette Score: {silhouette_scores[k_otimo]:.4f}  "
          f"(1.0 = clusters perfeitos)")
    print(f"{'═' * 60}")

    for cid, info in clusters.items():
        print(f"\n  CLUSTER {cid}  ·  {info['n_documentos']} documentos")
        print(f"  Tema: {info['rotulo']}")
        print(f"  {SEPARADOR}")
        print(f"  Termos:   {', '.join(info['top_termos'][:8])}")
        if info["exemplos_titulos"]:
            print("  Exemplos:")
            for t in info["exemplos_titulos"]:
                print(f"    → {t}")

    print(f"\n{'═' * 60}\n")


# ---------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------

def run(
    caminho_input: str,
    caminho_output: str,
    k_manual: int | None = None,
    modelo: str = DEFAULT_MODEL,
) -> dict:
    """
    Executa o pipeline NLP completo.

    Parâmetros
    ----------
    caminho_input  : ficheiro JSON do arquivo_pt_pipeline.py ou eventos_sns.json
    caminho_output : caminho para o JSON de resultados
    k_manual       : força um número de clusters (None = automático)
    modelo         : nome do modelo sentence-transformers

    Retorna
    -------
    Dicionário com clusters interpretados
    """
    log.info("━━━━  PASSO 1/6 · Carregamento  ━━━━")
    df_raw = carregar_textos(caminho_input)
    log.info("  Registos carregados: %d", len(df_raw))

    log.info("━━━━  PASSO 2/6 · Pré-processamento  ━━━━")
    df, textos = preparar_corpus(df_raw)

    log.info("━━━━  PASSO 3/6 · Embeddings  ━━━━")
    embeddings = gerar_embeddings(textos, modelo_nome=modelo)

    log.info("━━━━  PASSO 4/6 · Redução dimensional  ━━━━")
    emb_pca  = reduzir_pca(embeddings)
    emb_umap = reduzir_umap(embeddings)   # None se umap-learn não instalado

    log.info("━━━━  PASSO 5/6 · Clustering  ━━━━")
    if k_manual:
        k_otimo = k_manual
        # Calcula silhouette só para o k escolhido
        km_temp = KMeans(n_clusters=k_otimo, random_state=42, n_init="auto")
        lbl_temp = km_temp.fit_predict(emb_pca)
        score = silhouette_score(emb_pca, lbl_temp)
        silhouette_scores = {k_otimo: round(float(score), 4)}
        log.info("  k manual=%d  silhouette=%.4f", k_otimo, score)
    else:
        k_otimo, silhouette_scores = escolher_k(emb_pca)

    _, labels = aplicar_kmeans(emb_pca, k_otimo)

    log.info("━━━━  PASSO 6/6 · Interpretação de temas  ━━━━")
    clusters = interpretar_clusters(df, labels)

    exportar_resultados(
        df, labels, clusters, silhouette_scores,
        k_otimo, emb_umap, caminho_output,
    )

    imprimir_relatorio(clusters, silhouette_scores, k_otimo)
    return clusters


# ---------------------------------------------------------------------------
# ENTRADA DE EXECUÇÃO
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pipeline NLP: embeddings + clustering para textos do SNS"
    )
    p.add_argument(
        "--input", "-i",
        default="arquivo_sns_data.json",
        help="JSON de entrada (arquivo_pt_pipeline.py ou eventos_sns.json)",
    )
    p.add_argument(
        "--output", "-o",
        default="nlp_resultados.json",
        help="Ficheiro JSON de resultados",
    )
    p.add_argument(
        "--clusters", "-k",
        type=int,
        default=None,
        help="Número de clusters (omitir = automático por Silhouette)",
    )
    p.add_argument(
        "--modelo", "-m",
        default=DEFAULT_MODEL,
        help=f"Modelo sentence-transformers (default: {DEFAULT_MODEL})",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not Path(args.input).exists():
        log.error("Ficheiro não encontrado: %s", args.input)
        log.error("Gere o ficheiro com: python arquivo_pt_pipeline.py")
        sys.exit(1)

    run(
        caminho_input=args.input,
        caminho_output=args.output,
        k_manual=args.clusters,
        modelo=args.modelo,
    )
