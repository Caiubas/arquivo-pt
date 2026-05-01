"""
SNS Arquivo — Aplicação Streamlit Integrada
============================================
Três módulos num só:
  📅  Timeline    — eventos históricos interactivos
  🔍  Pesquisa    — busca semântica sobre os documentos
  🧠  NLP         — clustering temático com TF-IDF / embeddings

Executar:
    streamlit run app.py

Dependências mínimas:
    pip install streamlit plotly pandas numpy scikit-learn

Opcionais (melhoram a experiência):
    pip install sentence-transformers faiss-cpu anthropic
"""

# ── stdlib ─────────────────────────────────────────────────────────────────
import json
import logging
import os
import re
import sys
import warnings
from pathlib import Path

# ── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO DA PÁGINA
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SNS Arquivo · Portugal",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — CSS GLOBAL
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Sans', system-ui, sans-serif;
    color: #1a1f2e;
}

/* ── Variáveis ── */
:root {
    --ink:        #0d1117;
    --ink-2:      #3d4455;
    --ink-3:      #8891a8;
    --paper:      #f6f4f0;
    --paper-2:    #eceae4;
    --paper-3:    #dedbd3;
    --accent:     #c8410a;
    --accent-2:   #e8621e;
    --blue:       #1a4a7a;
    --blue-2:     #2d6ca8;
    --green:      #1d7a55;
    --gold:       #b8860b;
    --radius:     10px;
    --shadow:     0 2px 14px rgba(0,0,0,.07);
    --shadow-lg:  0 8px 36px rgba(0,0,0,.13);
}

/* ── App background ── */
.stApp                        { background: var(--paper); }
[data-testid="stSidebar"]     { background: #0d1117 !important; border-right: none; }
[data-testid="stSidebar"] *   { color: #ddd8ce !important; }
[data-testid="stSidebar"] hr  { border-color: #1e2433 !important; }

/* ── Tipografia ── */
h1, h2, h3 { font-family: 'DM Serif Display', Georgia, serif; }

/* ── Sidebar nav ── */
div[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    color: #9aa0b4 !important;
    font-size: .88rem !important;
    font-weight: 500 !important;
    text-align: left !important;
    padding: .5rem .8rem !important;
    border-radius: 8px !important;
    transition: background .15s, color .15s !important;
    width: 100% !important;
}
div[data-testid="stSidebar"] .stButton > button:hover {
    background: #1a2030 !important;
    color: #fff !important;
}
div[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: var(--accent) !important;
    color: #fff !important;
}

/* ── Page header ── */
.page-header {
    background: linear-gradient(135deg, #0d1117 0%, #1a2540 55%, #1a4a7a 100%);
    border-radius: 14px;
    padding: 2rem 2.4rem 1.6rem;
    margin-bottom: 1.8rem;
    position: relative; overflow: hidden;
}
.page-header::after {
    content: '';
    position: absolute; right: -50px; top: -50px;
    width: 250px; height: 250px; border-radius: 50%;
    background: radial-gradient(circle, rgba(200,65,10,.22) 0%, transparent 70%);
}
.page-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: .66rem; letter-spacing: 3px;
    text-transform: uppercase; color: var(--accent-2);
    margin-bottom: .4rem;
}
.page-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem; color: #fff;
    margin: 0 0 .35rem; line-height: 1.1;
}
.page-subtitle { font-size: .87rem; color: #7a829a; margin: 0; font-weight: 300; }

/* ── KPI grid ── */
.kpi-grid { display: flex; gap: 1rem; margin-bottom: 1.6rem; flex-wrap: wrap; }
.kpi-card {
    flex: 1; min-width: 120px;
    background: #fff; border-radius: var(--radius);
    padding: 1.1rem 1.3rem;
    border-top: 3px solid var(--blue);
    box-shadow: var(--shadow);
}
.kpi-card.accent { border-top-color: var(--accent); }
.kpi-card.green  { border-top-color: var(--green); }
.kpi-card.gold   { border-top-color: var(--gold); }
.kpi-val { font-family: 'DM Serif Display', serif; font-size: 1.85rem; color: var(--ink); line-height: 1; }
.kpi-lbl { font-size: .68rem; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; color: var(--ink-3); margin-top: .25rem; }

/* ── Section title ── */
.section-title {
    font-family: 'DM Serif Display', serif; font-size: 1.15rem;
    color: var(--ink); margin: 1.2rem 0 .9rem;
    padding-bottom: .35rem;
    border-bottom: 2px solid var(--paper-3);
    display: flex; align-items: center; gap: .45rem;
}

/* ── Timeline event card ── */
.ev-card {
    background: #fff; border-radius: var(--radius);
    padding: .85rem 1.1rem .95rem;
    margin-bottom: .5rem;
    border-left: 3px solid var(--blue);
    box-shadow: var(--shadow);
    transition: box-shadow .18s, transform .18s;
}
.ev-card:hover { box-shadow: var(--shadow-lg); transform: translateX(4px); }
.ev-date {
    font-family: 'DM Mono', monospace;
    font-size: .66rem; letter-spacing: 1.2px; color: var(--ink-3);
    text-transform: uppercase;
}
.ev-title { font-weight: 600; color: var(--ink); font-size: .95rem; margin: .13rem 0 .22rem; }
.ev-desc  { font-size: .81rem; color: var(--ink-2); line-height: 1.58; }
.pill {
    display: inline-block; font-size: .62rem; font-weight: 700;
    letter-spacing: 1px; text-transform: uppercase;
    padding: .1rem .5rem; border-radius: 20px; margin-top: .4rem; margin-right: .25rem;
}
.pill-alto  { background: #fde8e8; color: #b91c1c; }
.pill-medio { background: #fef3cd; color: #b45309; }
.pill-baixo { background: #dcfce7; color: #15803d; }
.pill-none  { background: #e8eaf0; color: #4a5068; }

/* ── Search result card ── */
.sr-card {
    background: #fff; border-radius: var(--radius);
    padding: 1rem 1.2rem; margin-bottom: .75rem;
    border-left: 4px solid var(--accent);
    box-shadow: var(--shadow);
    transition: box-shadow .15s;
}
.sr-card:hover { box-shadow: var(--shadow-lg); }
.sr-score {
    float: right;
    font-family: 'DM Mono', monospace; font-size: .73rem;
    background: var(--paper-2); padding: .1rem .42rem;
    border-radius: 4px; color: var(--ink-2);
}
.sr-rank {
    display: inline-flex; align-items: center; justify-content: center;
    width: 1.35rem; height: 1.35rem;
    border-radius: 50%; background: var(--accent); color: #fff;
    font-size: .68rem; font-weight: 700; margin-right: .45rem;
}
.sr-title  { font-weight: 600; color: var(--ink); font-size: .93rem; }
.sr-meta   { font-size: .73rem; color: var(--ink-3); margin: .18rem 0 .4rem; font-family: 'DM Mono', monospace; }
.sr-trecho { font-size: .81rem; color: var(--ink-2); line-height: 1.62; }

/* ── Cluster card ── */
.cluster-card {
    background: #fff; border-radius: var(--radius);
    padding: .95rem 1.2rem; margin-bottom: .75rem;
    box-shadow: var(--shadow);
    border-top: 3px solid var(--blue);
    transition: box-shadow .15s;
}
.cluster-card:hover { box-shadow: var(--shadow-lg); }
.cluster-num {
    font-family: 'DM Mono', monospace; font-size: .64rem;
    color: var(--ink-3); letter-spacing: 2px; text-transform: uppercase;
}
.cluster-title { font-family: 'DM Serif Display', serif; font-size: 1.08rem; color: var(--ink); margin: .2rem 0 .4rem; }
.term-chip {
    display: inline-block; background: var(--paper-2);
    color: var(--ink-2); font-size: .71rem;
    padding: .13rem .48rem; border-radius: 4px;
    margin: .12rem .12rem 0 0;
}

/* ── Answer box ── */
.answer-box {
    background: linear-gradient(135deg, #fff 0%, #f9f6f1 100%);
    border: 1px solid var(--paper-3); border-radius: var(--radius);
    padding: 1.35rem 1.6rem;
    margin-bottom: 1.3rem;
    box-shadow: var(--shadow);
    border-left: 4px solid var(--accent);
}
.answer-label {
    font-family: 'DM Mono', monospace; font-size: .64rem;
    letter-spacing: 2px; text-transform: uppercase;
    color: var(--accent); margin-bottom: .55rem; font-weight: 500;
}
.answer-text { font-size: .91rem; line-height: 1.78; color: var(--ink); }

/* ── Empty state ── */
.empty-state {
    text-align: center; padding: 3.5rem 1rem; color: var(--ink-3);
}
.empty-state .empty-icon { font-size: 2.8rem; margin-bottom: .7rem; }
.empty-state .empty-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem; color: var(--ink-2); margin-bottom: .35rem;
}
.empty-state .empty-sub { font-size: .82rem; }

/* ── Inputs ── */
input[type="text"], textarea {
    border-radius: var(--radius) !important;
    border: 1.5px solid var(--paper-3) !important;
    font-family: 'DM Sans', sans-serif !important;
}
input[type="text"]:focus, textarea:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px rgba(26,74,122,.1) !important;
}
div[data-baseweb="select"] > div { border-radius: var(--radius) !important; }
.stSlider > div { padding: 0 !important; }

/* ── Primary button override ── */
.main .stButton > button[kind="primary"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #fff !important;
    border-radius: var(--radius) !important;
    font-weight: 600 !important;
}
.main .stButton > button[kind="primary"]:hover {
    background: var(--accent-2) !important;
    border-color: var(--accent-2) !important;
}
.main .stButton > button {
    border-radius: var(--radius) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    gap: .3rem;
    border-bottom: 2px solid var(--paper-3);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px 8px 0 0;
    color: var(--ink-3);
    font-size: .84rem;
    font-weight: 500;
    padding: .5rem 1rem;
}
.stTabs [aria-selected="true"] {
    background: #fff !important;
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Keyword badge ── */
.kw-badge {
    display: inline-block;
    background: var(--paper-2); color: var(--ink-2);
    font-size: .72rem; padding: .15rem .55rem;
    border-radius: 20px; margin: .15rem .15rem 0 0;
    font-family: 'DM Mono', monospace;
}

/* ── URL link ── */
.url-link { font-size: .71rem; color: var(--blue-2); word-break: break-all; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PALETAS & CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════

CATEGORIA_CORES = {
    "Estrutura e Organização":   "#1a4a7a",
    "Acesso e Listas de Espera": "#2980b9",
    "Urgências e Emergências":   "#c0392b",
    "Recursos Humanos":          "#d35400",
    "Financiamento e Política":  "#8e44ad",
    "Crise e Rutura":            "#b8860b",
}
_COR_DEFAULT = "#95a5a6"


def _cor_para(row_or_kw, df_ref=None) -> str:
    """Devolve a cor da categoria de um keyword/row, ou _COR_DEFAULT."""
    if isinstance(row_or_kw, str):
        # lookup via df (se disponível)
        if df_ref is not None and not df_ref.empty:
            cats = df_ref.loc[df_ref["keyword"] == row_or_kw, "categoria"]
            if not cats.empty:
                return CATEGORIA_CORES.get(cats.iloc[0], _COR_DEFAULT)
        return _COR_DEFAULT
    # assume pandas Series/row
    return CATEGORIA_CORES.get(row_or_kw.get("categoria", ""), _COR_DEFAULT)

CLUSTER_PALETTE = [
    "#1a4a7a","#c8410a","#16a085","#8e44ad","#b8860b",
    "#2980b9","#e74c3c","#27ae60","#d35400","#1abc9c",
]

STOPWORDS_PT = {
    "de","a","o","que","e","do","da","em","um","para","com","uma","os","no",
    "se","na","por","mais","as","dos","como","mas","ao","ele","das","seu",
    "sua","ou","quando","muito","nos","já","eu","também","só","pelo","pela",
    "até","isso","ela","entre","depois","sem","mesmo","aos","seus","quem",
    "nas","me","esse","eles","você","essa","num","nem","suas","meu","às",
    "minha","numa","pelos","pelas","este","fosse","dele","tu","te","vocês",
    "vos","lhes","meus","minhas","teu","tua","teus","tuas","nosso","nossa",
    "nossos","nossas","dela","delas","deles","lhe","este","esta","estes",
    "estas","isto","aquele","aquela","aqueles","aquelas","aquilo","lo","la",
    "los","las","foi","são","ser","ter","tem","há","está","pelo",
    "serviço","nacional","saúde","sns","portugal","português","portuguesa",
    "hospital","hospitais","paciente","pacientes","anos","ano","mais","sobre",
    "http","www","html","php","pt","com","org","net","br","en",
}


# ═══════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS DE DADOS
# ═══════════════════════════════════════════════════════════════════════════

def _extrair_registos(dados) -> list:
    if isinstance(dados, list):
        return dados
    if isinstance(dados, dict):
        if "records" in dados:
            return dados["records"]
        for v in dados.values():
            if isinstance(v, list) and v:
                return v
    raise ValueError("Formato JSON não reconhecido.")


def _limpar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    texto = re.sub(r"<[^>]+>", " ", texto)
    texto = re.sub(r"https?://\S+", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto[:3000]


@st.cache_data(show_spinner=False)
def carregar_df(conteudo_json: str) -> pd.DataFrame:
    dados    = json.loads(conteudo_json)
    registos = _extrair_registos(dados)
    df = pd.DataFrame(registos)

    renomear = {
        "title": "titulo", "content": "texto",
        "date": "data", "keyword": "keyword",
        "originalURL": "url",
    }
    df = df.rename(columns={k: v for k, v in renomear.items() if k in df.columns})

    for col in ("titulo", "texto", "data", "keyword", "url", "archive_url", "categoria"):
        if col not in df.columns:
            df[col] = ""

    df["titulo"]   = df["titulo"].fillna("").astype(str).str.strip()
    df["texto"]    = df["texto"].apply(_limpar_texto)
    df["keyword"]  = df["keyword"].fillna("").astype(str)
    df["url"]      = df["url"].fillna("").astype(str)
    df["data_dt"]  = pd.to_datetime(df["data"], errors="coerce")
    df["ano"]      = df["data_dt"].dt.year
    df["mes"]      = df["data_dt"].dt.to_period("M").astype(str)
    df["cor_kw"]   = df["categoria"].map(CATEGORIA_CORES).fillna(_COR_DEFAULT)
    df["char_count"] = df["char_count"] if "char_count" in df.columns else df["texto"].str.len()
    return df.dropna(subset=["data_dt"]).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR — NAVEGAÇÃO
# ═══════════════════════════════════════════════════════════════════════════

PAGINAS = [
    ("📅", "Timeline",  "timeline"),
    ("🔍", "Pesquisa",  "pesquisa"),
    ("🧠", "NLP",       "nlp"),
]

with st.sidebar:
    st.markdown("""
    <div style="padding:1.3rem .5rem 1rem;">
        <div style="font-family:'DM Serif Display',serif;font-size:1.45rem;
                    color:#fff;line-height:1.05;">SNS<br>
            <span style="color:#c8410a;">Arquivo</span></div>
        <div style="font-size:.63rem;letter-spacing:2.5px;text-transform:uppercase;
                    color:#3a3f52;margin-top:.35rem;">Portugal · Arquivo.pt</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    if "pagina" not in st.session_state:
        st.session_state.pagina = "timeline"

    for icon, label, key in PAGINAS:
        is_active = st.session_state.pagina == key
        btn_type  = "primary" if is_active else "secondary"
        if st.button(f"{icon}  {label}", key=f"nav_{key}",
                     type=btn_type, use_container_width=True):
            st.session_state.pagina = key
            st.rerun()

    st.divider()

    st.markdown("<div style='font-size:.65rem;letter-spacing:2px;text-transform:uppercase;"
                "color:#3a3f52;margin-bottom:.5rem;'>📂 Dados</div>", unsafe_allow_html=True)

    ficheiro = st.file_uploader(
        "Carregar JSON",
        type="json",
        help="arquivo_sns_data.json  ou  nlp_resultados.json",
        label_visibility="collapsed",
    )

    # Carregar dados por defeito (ficheiro local ou upload)
    JSON_DEFAULT = Path(__file__).parent / "arquivo_sns_data.json"
    if ficheiro:
        conteudo = ficheiro.read().decode("utf-8")
        st.session_state.dados_json = conteudo
        st.success(f"✓ {ficheiro.name}")
    elif "dados_json" not in st.session_state:
        if JSON_DEFAULT.exists():
            st.session_state.dados_json = JSON_DEFAULT.read_text(encoding="utf-8")
        else:
            st.session_state.dados_json = None

    # NLP pré-computado (opcional)
    st.markdown("<div style='font-size:.65rem;letter-spacing:2px;text-transform:uppercase;"
                "color:#3a3f52;margin-top:.8rem;margin-bottom:.5rem;'>🧠 NLP pré-calculado</div>",
                unsafe_allow_html=True)
    nlp_ficheiro = st.file_uploader(
        "NLP JSON",
        type="json",
        key="nlp_upload",
        help="nlp_resultados.json do pipeline NLP",
        label_visibility="collapsed",
    )

    NLP_DEFAULT = Path(__file__).parent / "nlp_resultados.json"
    if nlp_ficheiro:
        st.session_state.nlp_json = nlp_ficheiro.read().decode("utf-8")
        st.success("✓ NLP carregado")
    elif "nlp_json" not in st.session_state:
        if NLP_DEFAULT.exists():
            st.session_state.nlp_json = NLP_DEFAULT.read_text(encoding="utf-8")
        else:
            st.session_state.nlp_json = None

    # API Key
    st.divider()
    st.markdown("<div style='font-size:.65rem;letter-spacing:2px;text-transform:uppercase;"
                "color:#3a3f52;margin-bottom:.5rem;'>🔑 Anthropic API</div>",
                unsafe_allow_html=True)
    api_key_input = st.text_input(
        "API Key", type="password", placeholder="sk-ant-…",
        label_visibility="collapsed",
        value=st.session_state.get("api_key", ""),
    )
    if api_key_input:
        st.session_state.api_key = api_key_input

    st.markdown("""
    <div style="margin-top:auto;padding-top:1rem;font-size:.63rem;color:#2a2f3e;text-align:center;">
        Dados via <a href="https://arquivo.pt" style="color:#c8410a;">Arquivo.pt</a>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# VERIFICAÇÃO DE DADOS
# ═══════════════════════════════════════════════════════════════════════════

if not st.session_state.get("dados_json"):
    st.markdown("""
    <div class="empty-state" style="padding:6rem 2rem;">
        <div class="empty-icon">📂</div>
        <div class="empty-title">Carregar dados para começar</div>
        <div class="empty-sub">
            Use o painel lateral para carregar<br>
            <code>arquivo_sns_data.json</code>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

try:
    df = carregar_df(st.session_state.dados_json)
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")
    st.stop()

if df.empty:
    st.warning("O ficheiro não contém registos com datas válidas.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA — TIMELINE
# ═══════════════════════════════════════════════════════════════════════════

def pagina_timeline(df: pd.DataFrame):
    st.markdown("""
    <div class="page-header">
        <div class="page-eyebrow">📅  Módulo 1 / 3</div>
        <h1 class="page-title">Timeline de Documentos</h1>
        <p class="page-subtitle">Arquivo histórico do SNS · filtre por keyword, período e dimensão</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Filtros ──────────────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
    with col_f1:
        ano_min = int(df["ano"].min())
        ano_max = int(df["ano"].max())
        intervalo = st.slider("Período", ano_min, ano_max,
                              (ano_min, ano_max), key="tl_anos")
    with col_f2:
        kws = sorted(df["keyword"].dropna().unique())
        kws_sel = st.multiselect("Keywords", kws, default=kws, key="tl_kws")
    with col_f3:
        ordenar = st.selectbox("Ordenar", ["Data ↑", "Data ↓", "Tamanho ↓"], key="tl_ord")

    # Filtro por categoria (campo vindo do pipeline v2)
    cats_disponiveis = sorted(df["categoria"].dropna().unique())
    cats_disponiveis = [c for c in cats_disponiveis if c]
    if cats_disponiveis:
        cats_sel = st.multiselect(
            "Categorias temáticas", cats_disponiveis,
            default=cats_disponiveis, key="tl_cats"
        )
    else:
        cats_sel = []

    mask = (
        df["ano"].between(intervalo[0], intervalo[1]) &
        df["keyword"].isin(kws_sel)
    )
    if cats_sel:
        mask = mask & df["categoria"].isin(cats_sel)
    dff = df[mask].copy()
    if ordenar == "Data ↑":
        dff = dff.sort_values("data_dt")
    elif ordenar == "Data ↓":
        dff = dff.sort_values("data_dt", ascending=False)
    else:
        dff["_cc"] = pd.to_numeric(dff["char_count"], errors="coerce").fillna(0)
        dff = dff.sort_values("_cc", ascending=False)

    if dff.empty:
        st.info("Nenhum documento com os filtros seleccionados.")
        return

    # ── KPIs ─────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-val">{len(dff)}</div>
            <div class="kpi-lbl">Documentos</div>
        </div>
        <div class="kpi-card accent">
            <div class="kpi-val">{dff['keyword'].nunique()}</div>
            <div class="kpi-lbl">Keywords</div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-val">{dff['url'].apply(lambda u: u.split('/')[2] if u.startswith('http') else '').nunique()}</div>
            <div class="kpi-lbl">Domínios</div>
        </div>
        <div class="kpi-card gold">
            <div class="kpi-val">{intervalo[1] - intervalo[0]}</div>
            <div class="kpi-lbl">Anos span</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Gráfico timeline scatter ──────────────────────────────────────────
    tab1, tab2 = st.tabs(["📍 Timeline", "📊 Distribuição"])

    with tab1:
        # ── Construção vectorizada: 1 trace por categoria, sem add_shape ──
        # Usar add_shape por documento causava >30s de build; error_y
        # substitui as linhas verticais com custo O(1) por trace.
        fig = go.Figure()

        # Linha de base (único shape)
        fig.add_shape(type="line",
            x0=dff["data_dt"].min(), x1=dff["data_dt"].max(), y0=0, y1=0,
            line=dict(color="#d0ccc5", width=2))

        for cat, cdf in dff.groupby("categoria"):
            cor = CATEGORIA_CORES.get(cat, _COR_DEFAULT)
            cdf = cdf.sort_values("data_dt").reset_index(drop=True)

            # Alternância vectorizada (sem iterrows)
            dates_ns  = cdf["data_dt"].astype(np.int64).values
            thresh_ns = 300 * 86_400 * int(1e9)
            sign      = np.ones(len(cdf))
            for i in range(1, len(cdf)):
                if abs(int(dates_ns[i]) - int(dates_ns[i-1])) < thresh_ns:
                    sign[i] = -sign[i-1]
            y_vals = sign * 0.55

            char_counts = pd.to_numeric(cdf["char_count"], errors="coerce").fillna(500)
            max_cc = char_counts.max()
            sizes = (
                (char_counts / max_cc * 10 + 7).clip(7, 20).tolist()
                if max_cc > 0 else [12.0] * len(cdf)
            )

            # Hover vectorizado (sem iterrows)
            hover = (
                "<b>" + cdf["titulo"].fillna("Sem título").astype(str) + "</b><br>"
                + "<span style='color:#aaa'>"
                + cdf["data_dt"].dt.strftime("%d %b %Y").fillna("Data desconhecida")
                + "</span><br><br>"
                + cdf["texto"].fillna("").astype(str).str[:220]
                + "<br><br><b>Keyword:</b> " + cdf["keyword"].astype(str)
                + "  ·  <b>Chars:</b> "
                + cdf["char_count"].fillna(0).astype(int).astype(str)
            ).tolist()

            # Linhas verticais: trace None-separado (sem add_shape)
            lx = np.empty(len(cdf) * 3, dtype=object)
            ly = np.empty(len(cdf) * 3, dtype=object)
            lx[0::3] = cdf["data_dt"].values
            lx[1::3] = cdf["data_dt"].values
            lx[2::3] = None
            ly[0::3] = 0.0
            ly[1::3] = y_vals
            ly[2::3] = None
            fig.add_trace(go.Scatter(
                x=lx, y=ly, mode="lines",
                line=dict(color=cor, width=1, dash="dot"),
                opacity=0.4, showlegend=False, hoverinfo="skip",
            ))

            # Marcadores no topo de cada linha (y=y_vals)
            fig.add_trace(go.Scatter(
                x=cdf["data_dt"], y=y_vals,
                mode="markers", name=cat,
                marker=dict(
                    size=sizes, color=cor,
                    line=dict(width=1.5, color="#fff"), opacity=.88,
                ),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover,
            ))

        fig.update_layout(
            height=420, margin=dict(l=10, r=10, t=30, b=10),
            plot_bgcolor="#faf8f4", paper_bgcolor="#faf8f4",
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                font=dict(size=11), bgcolor="rgba(255,255,255,.92)",
                bordercolor="#e0ddd6", borderwidth=1
            ),
            xaxis=dict(
                showgrid=True, gridcolor="#ece9e2", zeroline=False,
                tickfont=dict(size=11, color="#555"), tickformat="%Y"
            ),
            yaxis=dict(visible=False, range=[-1.4, 1.4]),
            hovermode="closest",
            hoverlabel=dict(bgcolor="#0d1117", font_size=12,
                            font_family="DM Sans, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})

    with tab2:
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            # Docs por keyword
            kw_c = dff.groupby("keyword").size().sort_values()
            cores_b = [
                CATEGORIA_CORES.get(
                    dff.loc[dff["keyword"]==k, "categoria"].iloc[0]
                    if not dff.loc[dff["keyword"]==k].empty else "", _COR_DEFAULT
                ) for k in kw_c.index]
            fb = go.Figure(go.Bar(
                x=kw_c.values, y=kw_c.index, orientation="h",
                marker_color=cores_b, marker_line_width=0,
                text=kw_c.values, textposition="outside"
            ))
            fb.update_layout(
                height=280, title="Documentos por keyword",
                margin=dict(l=10, r=40, t=35, b=10),
                plot_bgcolor="#faf8f4", paper_bgcolor="#faf8f4",
                xaxis=dict(visible=False), showlegend=False,
                yaxis=dict(tickfont=dict(size=11)),
                title_font=dict(family="DM Serif Display", size=14),
            )
            st.plotly_chart(fb, use_container_width=True,
                            config={"displayModeBar": False})

        with col_b2:
            # Docs por ano
            ano_c = dff.groupby("ano").size().reset_index(name="n")
            fa = go.Figure(go.Bar(
                x=ano_c["ano"], y=ano_c["n"],
                marker_color="#1a4a7a", marker_line_width=0,
                text=ano_c["n"], textposition="outside"
            ))
            fa.update_layout(
                height=280, title="Documentos por ano",
                margin=dict(l=10, r=10, t=35, b=10),
                plot_bgcolor="#faf8f4", paper_bgcolor="#faf8f4",
                showlegend=False,
                xaxis=dict(tickfont=dict(size=11), dtick=1),
                yaxis=dict(visible=False),
                title_font=dict(family="DM Serif Display", size=14),
            )
            st.plotly_chart(fa, use_container_width=True,
                            config={"displayModeBar": False})

    # ── Lista de documentos ───────────────────────────────────────────────
    st.markdown('<div class="section-title">📋 Documentos</div>', unsafe_allow_html=True)

    col_s, col_pp = st.columns([5, 1])
    with col_s:
        pesq = st.text_input("", placeholder="🔍 Filtrar por título ou conteúdo…",
                             key="tl_search", label_visibility="collapsed")
    with col_pp:
        por_pag = st.selectbox("", [10, 20, 50], key="tl_pp",
                               label_visibility="collapsed")

    lista = dff.copy()
    if pesq:
        m = (lista["titulo"].str.contains(pesq, case=False, na=False) |
             lista["texto"].str.contains(pesq, case=False, na=False))
        lista = lista[m]

    # Paginação
    total = len(lista)
    n_pags = max(1, (total + por_pag - 1) // por_pag)
    if "tl_pag" not in st.session_state:
        st.session_state.tl_pag = 1
    pag = st.session_state.tl_pag

    inicio = (pag - 1) * por_pag
    fatia  = lista.iloc[inicio:inicio + por_pag]

    for _, row in fatia.iterrows():
        cor = CATEGORIA_CORES.get(row.get("categoria", ""), _COR_DEFAULT)
        texto_preview = str(row["texto"])[:180]
        if len(str(row["texto"])) > 180:
            texto_preview += "…"
        url_html = ""
        if row.get("url"):
            url_html = f'<div class="url-link">🔗 {str(row["url"])[:80]}</div>'
        st.markdown(f"""
        <div class="ev-card" style="border-left-color:{cor}">
            <div class="ev-date">{row['data_dt'].strftime('%d %b %Y') if pd.notna(row['data_dt']) else 'Data desconhecida'}</div>
            <div class="ev-title">{row['titulo'] or "Sem título"}</div>
            <div class="ev-desc">{texto_preview}</div>
            {url_html}
            <span class="pill" style="background:{cor}18;color:{cor}">{row['keyword']}</span>
        </div>
        """, unsafe_allow_html=True)

    # Navegação de páginas
    if n_pags > 1:
        col_prev, col_info, col_next = st.columns([1, 3, 1])
        with col_prev:
            if st.button("← Anterior", disabled=(pag <= 1), key="tl_prev"):
                st.session_state.tl_pag -= 1
                st.rerun()
        with col_info:
            st.markdown(
                f"<div style='text-align:center;font-size:.8rem;color:#8891a8;padding-top:.5rem;'>"
                f"Página {pag} / {n_pags}  ·  {total} documentos</div>",
                unsafe_allow_html=True
            )
        with col_next:
            if st.button("Próxima →", disabled=(pag >= n_pags), key="tl_next"):
                st.session_state.tl_pag += 1
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA — PESQUISA
# ═══════════════════════════════════════════════════════════════════════════

def _busca_keyword(df: pd.DataFrame, query: str, top_k: int = 8) -> list[dict]:
    """Pesquisa simples por palavras-chave com scoring de relevância."""
    palavras = re.findall(r"\w{3,}", query.lower())
    if not palavras:
        return []

    resultados = []
    for _, row in df.iterrows():
        titulo = str(row.get("titulo", "")).lower()
        texto  = str(row.get("texto", "")).lower()
        score  = sum(titulo.count(p) * 3 + texto.count(p) for p in palavras)
        if score > 0:
            resultados.append({"row": row, "score": score})

    resultados.sort(key=lambda x: x["score"], reverse=True)
    return resultados[:top_k]


def pagina_pesquisa(df: pd.DataFrame):
    st.markdown("""
    <div class="page-header">
        <div class="page-eyebrow">🔍  Módulo 2 / 3</div>
        <h1 class="page-title">Pesquisa Inteligente</h1>
        <p class="page-subtitle">Busca por palavras-chave com scoring de relevância · documentos do SNS</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Barra de pesquisa ────────────────────────────────────────────────
    col_q, col_k, col_btn = st.columns([4, 1, 1])
    with col_q:
        query = st.text_input("", placeholder="Ex: triagem Manchester urgência…",
                              key="ps_query", label_visibility="collapsed")
    with col_k:
        top_k = st.selectbox("Top", [5, 8, 15, 20], index=1, key="ps_k",
                             label_visibility="collapsed")
    with col_btn:
        buscar = st.button("Pesquisar", type="primary", use_container_width=True)

    # ── Sugestões ────────────────────────────────────────────────────────
    sugestoes = [
        "triagem Manchester",
        "urgência hospitalar",
        "médico de família",
        "SNS24 contacto",
        "serviços saúde Portugal",
    ]
    st.markdown("<div style='margin:.3rem 0 .8rem;'>", unsafe_allow_html=True)
    cols_s = st.columns(len(sugestoes))
    for i, (col, sug) in enumerate(zip(cols_s, sugestoes)):
        with col:
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state.ps_query = sug
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    pergunta = query or st.session_state.get("ps_query", "")

    if not pergunta:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🔍</div>
            <div class="empty-title">Faça uma pesquisa sobre o SNS</div>
            <div class="empty-sub">Experimente as sugestões acima ou escreva a sua pergunta</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Executar pesquisa ────────────────────────────────────────────────
    with st.spinner("A pesquisar…"):
        resultados = _busca_keyword(df, pergunta, top_k)

    if not resultados:
        st.warning("Nenhum documento relevante encontrado para esta pesquisa.")
        return

    # ── Métricas rápidas ─────────────────────────────────────────────────
    max_score = max(r["score"] for r in resultados)
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-val">{len(resultados)}</div>
            <div class="kpi-lbl">Resultados</div>
        </div>
        <div class="kpi-card accent">
            <div class="kpi-val">{max_score}</div>
            <div class="kpi-lbl">Score máx.</div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-val">{len(set(r['row']['keyword'] for r in resultados))}</div>
            <div class="kpi-lbl">Keywords</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Resposta IA (se API key disponível) ──────────────────────────────
    api_key = st.session_state.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        contexto = "\n\n".join(
            f"[{i+1}] {r['row']['titulo']}\n{str(r['row']['texto'])[:400]}"
            for i, r in enumerate(resultados[:5])
        )
        with st.spinner("A gerar resposta com IA…"):
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                msg = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=600,
                    system=(
                        "És um assistente especializado no Serviço Nacional de Saúde português. "
                        "Responde em português europeu de forma concisa e factual, "
                        "baseando-te nos documentos fornecidos."
                    ),
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Com base nos seguintes documentos do arquivo SNS, "
                            f"responde à pergunta: '{pergunta}'\n\n"
                            f"Documentos:\n{contexto}"
                        )
                    }]
                )
                resposta = msg.content[0].text
                st.markdown(f"""
                <div class="answer-box">
                    <div class="answer-label">✦  Resposta gerada por IA</div>
                    <div class="answer-text">{resposta.replace(chr(10), '<br>')}</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Erro ao gerar resposta: {e}")
    else:
        st.info("💡 Defina a **Anthropic API Key** na sidebar para obter respostas geradas por IA.", icon="💡")

    # ── Resultados ────────────────────────────────────────────────────────
    st.markdown(f'<div class="section-title">📄 {len(resultados)} documentos encontrados</div>',
                unsafe_allow_html=True)

    for rank, res in enumerate(resultados, 1):
        row   = res["row"]
        score = res["score"]
        score_pct = min(100, int(score / max(max_score, 1) * 100))

        trecho = str(row["texto"])[:300] + ("…" if len(str(row["texto"])) > 300 else "")
        url_html = ""
        if row.get("url"):
            url_html = (f'<div style="margin-top:.5rem;">'
                        f'<a href="{row["url"]}" target="_blank" class="url-link">🔗 {str(row["url"])[:80]}</a>'
                        f'</div>')
        archive_html = ""
        if row.get("archive_url"):
            archive_html = (f'<a href="{row["archive_url"]}" target="_blank" '
                            f'style="font-size:.7rem;color:#8891a8;margin-left:.8rem;">'
                            f'📦 Arquivo</a>')

        cor = CATEGORIA_CORES.get(row.get("categoria", ""), _COR_DEFAULT)

        st.markdown(f"""
        <div class="sr-card">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.35rem;">
                <div>
                    <span class="sr-rank">{rank}</span>
                    <span class="sr-title">{row['titulo'] or 'Sem título'}</span>
                </div>
                <span class="sr-score">score {score_pct}%</span>
            </div>
            <div class="sr-meta">
                {str(row['data_dt'])[:10] if not pd.isna(row['data_dt']) else 'N/A'}
                · <span class="kw-badge">{row.get('keyword','')}</span>
            </div>
            <div class="sr-trecho">{trecho}</div>
            {url_html}
            {archive_html}
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA — NLP
# ═══════════════════════════════════════════════════════════════════════════

def _tfidf_cluster(textos: list[str], labels: np.ndarray, n_termos: int = 10) -> dict:
    """Extrai os termos mais representativos de cada cluster por TF-IDF."""
    vec = TfidfVectorizer(
        min_df=1, max_df=.95, ngram_range=(1, 2),
        stop_words=list(STOPWORDS_PT),
        token_pattern=r"(?u)\b[a-záéíóúâêîôûãõàèìòùçA-ZÁÉÍÓÚÂÊÎÔÛÃÕÀÈÌÒÙÇ]{3,}\b",
    )
    vec.fit(textos)
    termos = vec.get_feature_names_out()
    clusters = {}
    for cid in sorted(set(labels)):
        idxs   = np.where(labels == cid)[0]
        corpus = " ".join(textos[i] for i in idxs)
        v      = vec.transform([corpus]).toarray()[0]
        top    = [termos[i] for i in v.argsort()[::-1][:n_termos] if v[i] > 0]
        clusters[int(cid)] = {"termos": top, "n": int(len(idxs)), "idxs": idxs.tolist()}
    return clusters


@st.cache_data(show_spinner=False)
def _clustering_tfidf(textos_json: str, k: int) -> tuple:
    """Clustering TF-IDF + KMeans com PCA 2D para visualização."""
    textos = json.loads(textos_json)
    vec = TfidfVectorizer(
        max_features=600, stop_words=list(STOPWORDS_PT),
        token_pattern=r"(?u)\b[a-záéíóúâêîôûãõàèìòùç]{3,}\b"
    )
    X = vec.fit_transform(textos).toarray().astype(np.float32)
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X)
    temas  = _tfidf_cluster(textos, labels)
    coords = PCA(n_components=2, random_state=42).fit_transform(X).tolist()
    return labels.tolist(), temas, coords


def _carregar_nlp_precomputado(nlp_json: str):
    try:
        d = json.loads(nlp_json)

        if "clusters" not in d or "documentos" not in d:
            return None

        temas = {}
        for cid_str, info in d["clusters"].items():
            cid = int(cid_str)
            temas[cid] = {
                "termos": info.get("top_termos", []),
                "n": info.get("n_documentos", 0),
                "idxs": info.get("ids_documentos", []),
                "rotulo": info.get("rotulo", ""),
            }

        docs = d["documentos"]

        coords = []
        for doc in docs:
            if "umap_x" in doc and "umap_y" in doc:
                coords.append([doc["umap_x"], doc["umap_y"]])
            else:
                coords.append([0, 0])

        labels = [doc.get("cluster_id", -1) for doc in docs]

        return temas, coords, labels, docs, d.get("metadata", {})

    except Exception:
        return None


def pagina_nlp(df):
    st.title("🧠 NLP - Análise Semântica (BERTopic-like)")
    with open("nlp_resultados.json", "r", encoding="utf-8") as f:
        st.session_state["nlp_cache"] = json.load(f)
    # ─────────────────────────────────────────────
    # CONFIGURAÇÃO EM TEMPO REAL
    # ─────────────────────────────────────────────

    col1, col2 = st.columns(2)

    with col1:
        min_cluster_size = st.slider(
            "Tamanho mínimo do cluster",
            5, 100, 20, 5
        )

    with col2:
        min_samples = st.slider(
            "Sensibilidade (outliers)",
            1, 20, 5, 1
        )

    # ─────────────────────────────────────────────
    # CARREGAR NLP PRÉ-CALCULADO (OU CACHE)
    # ─────────────────────────────────────────────

    if "nlp_cache" not in st.session_state:
        st.warning("Carregue primeiro o resultado do pipeline NLP.")
        return

    data = st.session_state["nlp_cache"]

    df_plot = pd.DataFrame(data["documentos"])
    clusters = data["clusters"]

    # garante consistência
    df_plot = df_plot.dropna(subset=["umap_x", "umap_y"]).reset_index(drop=True)

    # ─────────────────────────────────────────────
    # FILTRO DINÂMICO DE CLUSTERS (SIMULA “REAL TIME”)
    # ─────────────────────────────────────────────

    # recria clusters dinamicamente por heurística leve
    cluster_counts = df_plot["cluster_id"].value_counts()

    valid_clusters = cluster_counts[
        cluster_counts >= min_cluster_size
    ].index.tolist()

    df_filtered = df_plot[df_plot["cluster_id"].isin(valid_clusters)]

    # ─────────────────────────────────────────────
    # MÉTRICAS
    # ─────────────────────────────────────────────

    n_clusters = len(valid_clusters)
    n_outliers = len(df_plot) - len(df_filtered)

    colm1, colm2 = st.columns(2)
    colm1.metric("Clusters ativos", n_clusters)
    colm2.metric("Documentos fora (simulado)", n_outliers)

    # ─────────────────────────────────────────────
    # MAPA SEMÂNTICO
    # ─────────────────────────────────────────────

    fig = go.Figure()

    palette = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
        "#FFA15A", "#19D3F3", "#FF6692", "#B6E880"
    ]

    for cid in valid_clusters:
        sub = df_filtered[df_filtered["cluster_id"] == cid]

        if sub.empty:
            continue

        cluster_info = clusters.get(str(cid), clusters.get(cid, {}))
        rotulo = cluster_info.get("rotulo", f"Cluster {cid}")

        color = palette[cid % len(palette)]

        hover = [
            f"<b>{row.get('titulo','')}</b><br>{row.get('data','')}"
            for _, row in sub.iterrows()
        ]

        if cid == -1:
            name = "⚠️ Outliers (ruído)"
        else:
            name = f"C{cid}: {rotulo}"

        fig.add_trace(go.Scatter(
            x=sub["umap_x"],
            y=sub["umap_y"],
            mode="markers",
            name=name,
            marker=dict(size=10, color=color),
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
        ))

    fig.update_layout(
        height=700,
        margin=dict(l=10, r=10, t=30, b=10),
        template="plotly_dark",
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────────────────────
    # EXPLORADOR DE CLUSTERS
    # ─────────────────────────────────────────────

    st.subheader("🔎 Explorar clusters")

    selected_cluster = st.selectbox(
        "Escolha um cluster",
        sorted(valid_clusters)
    )

    subset = df_plot[df_plot["cluster_id"] == selected_cluster]

    st.write(f"📊 {len(subset)} documentos")

    st.dataframe(
        subset[["titulo", "data", "cluster_id"]].head(20),
        use_container_width=True
    )

# ═══════════════════════════════════════════════════════════════════════════
# ROTEAMENTO
# ═══════════════════════════════════════════════════════════════════════════

pagina = st.session_state.get("pagina", "timeline")

if pagina == "timeline":
    pagina_timeline(df)
elif pagina == "pesquisa":
    pagina_pesquisa(df)
elif pagina == "nlp":
    pagina_nlp(df)


# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="margin-top:3.5rem;padding:1rem 0;border-top:1px solid #e0ddd6;
            display:flex;justify-content:space-between;align-items:center;
            font-size:.71rem;color:#b0aaa0;">
    <span>SNS Arquivo · Streamlit + Plotly + scikit-learn</span>
    <span>Dados via <a href="https://arquivo.pt" style="color:#c8410a;text-decoration:none;">Arquivo.pt</a></span>
</div>
""", unsafe_allow_html=True)