"""
Timeline Interativa do SNS — Streamlit + Plotly
================================================
Visualiza eventos históricos do SNS português com filtros por categoria,
impacto e intervalo temporal.

Instalar dependências:
    pip install streamlit plotly pandas

Executar:
    streamlit run timeline_sns.py
"""

import json
import pathlib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Timeline SNS Portugal",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Estilos CSS personalizados
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Importar fonte editorial */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Source+Sans+3:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Source Sans 3', sans-serif;
    }

    /* Cabeçalho principal */
    .header-block {
        background: linear-gradient(135deg, #0a2342 0%, #1a4a7a 60%, #1e6fa8 100%);
        border-radius: 12px;
        padding: 2.5rem 2rem 2rem 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .header-block::before {
        content: '';
        position: absolute;
        top: -40px; right: -40px;
        width: 200px; height: 200px;
        border-radius: 50%;
        background: rgba(255,255,255,0.04);
    }
    .header-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.4rem;
        color: #ffffff;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .header-sub {
        font-size: 1rem;
        color: #a8c8e8;
        font-weight: 300;
        margin: 0;
    }
    .header-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.2);
        color: #fff;
        font-size: 0.72rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        margin-bottom: 0.8rem;
    }

    /* Cards de métricas */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        flex: 1;
        background: #ffffff;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid #1a4a7a;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .metric-card.green  { border-left-color: #2ecc71; }
    .metric-card.orange { border-left-color: #e67e22; }
    .metric-card.red    { border-left-color: #e74c3c; }
    .metric-value {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        color: #0a2342;
        line-height: 1;
        margin-bottom: 0.2rem;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Tabela de eventos */
    .event-card {
        background: #fff;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        border-left: 3px solid #1a4a7a;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        transition: box-shadow 0.2s;
    }
    .event-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .event-date {
        font-size: 0.75rem;
        color: #999;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .event-title {
        font-size: 1rem;
        font-weight: 600;
        color: #0a2342;
        margin: 0.1rem 0;
    }
    .event-desc {
        font-size: 0.85rem;
        color: #555;
        line-height: 1.5;
    }
    .tag {
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        padding: 0.15rem 0.6rem;
        border-radius: 20px;
        margin-top: 0.4rem;
        margin-right: 0.3rem;
    }
    .tag-alto   { background: #fde8e8; color: #c0392b; }
    .tag-medio  { background: #fef3cd; color: #d35400; }
    .tag-baixo  { background: #e8f5e9; color: #27ae60; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #f4f7fb;
    }
    .sidebar-section {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #1a4a7a;
        margin: 1.2rem 0 0.4rem 0;
    }
    div[data-baseweb="select"] > div {
        border-radius: 8px !important;
    }
    .stSlider > div { padding: 0 !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Paleta de cores por categoria
# ---------------------------------------------------------------------------
CATEGORIA_CORES = {
    "Legislação":    "#1a4a7a",
    "Infraestrutura":"#2980b9",
    "Gestão":        "#16a085",
    "Urgência":      "#c0392b",
    "Política":      "#8e44ad",
    "Saúde Pública": "#27ae60",
}

IMPACTO_CORES = {
    "alto":  "#e74c3c",
    "medio": "#e67e22",
    "baixo": "#2ecc71",
}

IMPACTO_SIMBOLOS = {
    "alto":  "diamond",
    "medio": "circle",
    "baixo": "circle-open",
}

IMPACTO_TAMANHOS = {
    "alto":  18,
    "medio": 13,
    "baixo": 10,
}


# ---------------------------------------------------------------------------
# Carregamento de dados
# ---------------------------------------------------------------------------

def _extrair_registos(dados) -> list:
    """
    Normaliza o JSON de entrada para uma lista de registos.

    Suporta dois formatos:
      - Lista simples:          [{...}, {...}]           → eventos_sns.json
      - Envelope com metadata:  {"metadata": ...,        → arquivo_pt_pipeline.py
                                 "records": [{...}]}
    """
    if isinstance(dados, list):
        return dados
    if isinstance(dados, dict):
        # Formato do pipeline: chave "records"
        if "records" in dados:
            return dados["records"]
        # Fallback: tenta qualquer chave cujo valor seja uma lista
        for v in dados.values():
            if isinstance(v, list) and v:
                return v
    raise ValueError(
        "Formato JSON não reconhecido. "
        "Esperado: lista de eventos ou {'records': [...]}."
    )


def _enriquecer_df(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona colunas derivadas de data e paleta de cores."""
    # Garante campo 'titulo' mesmo quando o pipeline usa 'title'
    if "titulo" not in df.columns and "title" in df.columns:
        df = df.rename(columns={"title": "titulo"})
    # Garante campo 'descricao' mesmo quando o pipeline usa 'content'
    if "descricao" not in df.columns and "content" in df.columns:
        df["descricao"] = df["content"].str[:200] + "…"
    # Garante campo 'categoria' com fallback
    if "categoria" not in df.columns:
        df["categoria"] = df.get("keyword", "Outro")
    # Garante campo 'impacto' com fallback
    if "impacto" not in df.columns:
        df["impacto"] = "medio"

    df["data"]      = pd.to_datetime(df["date"] if "date" in df.columns and "data" not in df.columns else df["data"], errors="coerce")
    df              = df.dropna(subset=["data"])
    df["ano"]       = df["data"].dt.year
    df["cor_cat"]   = df["categoria"].map(CATEGORIA_CORES).fillna("#95a5a6")
    df["cor_imp"]   = df["impacto"].map(IMPACTO_CORES).fillna("#95a5a6")
    df["simbolo"]   = df["impacto"].map(IMPACTO_SIMBOLOS).fillna("circle")
    df["tamanho"]   = df["impacto"].map(IMPACTO_TAMANHOS).fillna(10)
    return df


@st.cache_data
def carregar_dados(caminho: str) -> pd.DataFrame:
    """Carrega o JSON local e converte para DataFrame com tipos corretos."""
    with open(caminho, encoding="utf-8") as f:
        dados = json.load(f)
    return _enriquecer_df(pd.DataFrame(_extrair_registos(dados)))


# Tenta carregar ficheiro local; se não existir, usa dados embutidos de exemplo
CAMINHO_JSON = pathlib.Path(__file__).parent / "eventos_sns.json"


# ---------------------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------------------
st.markdown("""
<div class="header-block">
    <div class="header-badge">🏥 História da Saúde em Portugal</div>
    <h1 class="header-title">Timeline do SNS</h1>
    <p class="header-sub">Serviço Nacional de Saúde · 1979 – presente</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — controlos
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Filtros")

    # Upload de JSON personalizado
    st.markdown('<div class="sidebar-section">📂 Fonte de dados</div>', unsafe_allow_html=True)
    ficheiro = st.file_uploader(
        "Carregar JSON de eventos",
        type="json",
        help="Campos esperados: titulo, data (YYYY-MM-DD), categoria, descricao, impacto",
    )

    if ficheiro:
        try:
            dados = json.load(ficheiro)
            df_base = _enriquecer_df(pd.DataFrame(_extrair_registos(dados)))
        except (ValueError, KeyError) as e:
            st.error(f"Erro ao carregar JSON: {e}")
            st.stop()
    elif CAMINHO_JSON.exists():
        df_base = carregar_dados(str(CAMINHO_JSON))
    else:
        st.error("Ficheiro eventos_sns.json não encontrado. Carregue um JSON.")
        st.stop()

    # Filtro de intervalo de anos
    st.markdown('<div class="sidebar-section">📅 Período</div>', unsafe_allow_html=True)
    ano_min, ano_max = int(df_base["ano"].min()), int(df_base["ano"].max())
    intervalo = st.slider(
        "Intervalo de anos",
        min_value=ano_min,
        max_value=ano_max,
        value=(ano_min, ano_max),
        step=1,
        label_visibility="collapsed",
    )

    # Filtro de categorias
    st.markdown('<div class="sidebar-section">🏷️ Categorias</div>', unsafe_allow_html=True)
    categorias_disp = sorted(df_base["categoria"].unique())
    categorias_sel  = st.multiselect(
        "Categorias",
        options=categorias_disp,
        default=categorias_disp,
        label_visibility="collapsed",
    )

    # Filtro de impacto
    st.markdown('<div class="sidebar-section">⚡ Impacto</div>', unsafe_allow_html=True)
    impactos_disp = [i for i in ["alto", "medio", "baixo"] if i in df_base["impacto"].values]
    impactos_sel  = st.multiselect(
        "Impacto",
        options=impactos_disp,
        default=impactos_disp,
        label_visibility="collapsed",
    )

    # Opções de visualização
    st.markdown('<div class="sidebar-section">👁 Visualização</div>', unsafe_allow_html=True)
    colorir_por = st.radio(
        "Colorir por",
        options=["Categoria", "Impacto"],
        horizontal=True,
        label_visibility="collapsed",
    )
    mostrar_anotacoes = st.toggle("Mostrar títulos no gráfico", value=False)
    mostrar_tabela    = st.toggle("Mostrar lista de eventos", value=True)

    st.divider()
    st.caption("Fonte: Arquivo.pt · SNS Portugal")


# ---------------------------------------------------------------------------
# Aplicar filtros
# ---------------------------------------------------------------------------
mask = (
    df_base["ano"].between(intervalo[0], intervalo[1]) &
    df_base["categoria"].isin(categorias_sel) &
    df_base["impacto"].isin(impactos_sel)
)
df = df_base[mask].copy().sort_values("data")

if df.empty:
    st.warning("Nenhum evento encontrado com os filtros selecionados.")
    st.stop()


# ---------------------------------------------------------------------------
# Métricas de resumo
# ---------------------------------------------------------------------------
n_total  = len(df)
n_alto   = (df["impacto"] == "alto").sum()
n_cats   = df["categoria"].nunique()
span_anos = df["ano"].max() - df["ano"].min()

cols = st.columns(4)
metricas = [
    (n_total,   "Eventos",       ""),
    (n_alto,    "Alto Impacto",  "red"),
    (n_cats,    "Categorias",    "green"),
    (span_anos, "Anos de história", "orange"),
]
for col, (val, lbl, cor) in zip(cols, metricas):
    with col:
        st.markdown(f"""
        <div class="metric-card {cor}">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Construção do gráfico Plotly
# ---------------------------------------------------------------------------
def build_timeline(df: pd.DataFrame, colorir_por: str, anotacoes: bool) -> go.Figure:
    fig = go.Figure()

    # Linha base da timeline
    fig.add_shape(
        type="line",
        x0=df["data"].min(), x1=df["data"].max(),
        y0=0, y1=0,
        line=dict(color="#dce3ec", width=2, dash="dot"),
    )

    # Agrupa por categoria ou impacto para legenda
    grupo_col = "categoria" if colorir_por == "Categoria" else "impacto"
    cor_col   = "cor_cat"   if colorir_por == "Categoria" else "cor_imp"

    for grupo, gdf in df.groupby(grupo_col):
        cor = gdf[cor_col].iloc[0]

        # Texto de hover enriquecido
        hover_texts = []
        for _, row in gdf.iterrows():
            hover_texts.append(
                f"<b>{row['titulo']}</b><br>"
                f"<span style='color:#aaa'>{row['data'].strftime('%d %b %Y')}</span><br><br>"
                f"{row['descricao']}<br><br>"
                f"<b>Categoria:</b> {row['categoria']}  |  <b>Impacto:</b> {row['impacto'].upper()}"
            )

        # Posição Y alternada para evitar sobreposição de labels
        y_vals = []
        ultimo_x = None
        y_atual  = 0.5
        for _, row in gdf.iterrows():
            if ultimo_x and abs((row["data"] - ultimo_x).days) < 500:
                y_atual = -y_atual  # alterna acima/abaixo da linha
            else:
                y_atual = 0.5
            y_vals.append(y_atual)
            ultimo_x = row["data"]

        # Linhas verticais de cada evento até a linha base
        for i, (_, row) in enumerate(gdf.iterrows()):
            fig.add_shape(
                type="line",
                x0=row["data"], x1=row["data"],
                y0=0, y1=y_vals[i],
                line=dict(color=cor, width=1.2, dash="dot"),
                opacity=0.5,
            )

        # Scatter principal
        fig.add_trace(go.Scatter(
            x=gdf["data"],
            y=y_vals,
            mode="markers+text" if anotacoes else "markers",
            name=str(grupo),
            marker=dict(
                size=gdf["tamanho"].tolist(),
                color=cor,
                symbol=gdf["simbolo"].tolist(),
                line=dict(width=2, color="#ffffff"),
                opacity=0.92,
            ),
            text=gdf["titulo"] if anotacoes else None,
            textposition="top center",
            textfont=dict(size=9, color="#333"),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_texts,
        ))

    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="#f8fafd",
        paper_bgcolor="#f8fafd",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left",   x=0,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#dce3ec",
            borderwidth=1,
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#e8edf4",
            gridwidth=1,
            zeroline=False,
            tickfont=dict(size=11, color="#555"),
            tickformat="%Y",
        ),
        yaxis=dict(
            visible=False,
            range=[-1.4, 1.4],
        ),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#0a2342",
            font_size=12,
            font_family="Source Sans 3, sans-serif",
            bordercolor="#1a4a7a",
        ),
    )
    return fig


fig = build_timeline(df, colorir_por, mostrar_anotacoes)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Distribuição por categoria — gráfico de barras horizontal
# ---------------------------------------------------------------------------
with st.expander("📊 Distribuição por categoria", expanded=False):
    cat_counts = df.groupby("categoria").size().sort_values(ascending=True)
    cores_barras = [CATEGORIA_CORES.get(c, "#95a5a6") for c in cat_counts.index]

    fig_bar = go.Figure(go.Bar(
        x=cat_counts.values,
        y=cat_counts.index,
        orientation="h",
        marker_color=cores_barras,
        marker_line_width=0,
        text=cat_counts.values,
        textposition="outside",
    ))
    fig_bar.update_layout(
        height=280,
        margin=dict(l=10, r=40, t=10, b=10),
        plot_bgcolor="#f8fafd",
        paper_bgcolor="#f8fafd",
        xaxis=dict(visible=False),
        yaxis=dict(tickfont=dict(size=12), gridcolor="#e8edf4"),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Lista de eventos
# ---------------------------------------------------------------------------
if mostrar_tabela:
    st.markdown("### 📋 Eventos selecionados")

    # Barra de pesquisa de texto livre
    pesquisa = st.text_input(
        "🔍 Pesquisar evento",
        placeholder="Ex: triagem, pandemia, USF…",
        label_visibility="collapsed",
    )

    df_lista = df.copy()
    if pesquisa:
        mask_txt = (
            df_lista["titulo"].str.contains(pesquisa, case=False, na=False) |
            df_lista["descricao"].str.contains(pesquisa, case=False, na=False)
        )
        df_lista = df_lista[mask_txt]

    if df_lista.empty:
        st.info("Nenhum evento encontrado para essa pesquisa.")
    else:
        for _, row in df_lista.sort_values("data").iterrows():
            cor_borda = CATEGORIA_CORES.get(row["categoria"], "#95a5a6")
            tag_imp   = f'<span class="tag tag-{row["impacto"]}">{row["impacto"].upper()}</span>'
            tag_cat   = f'<span class="tag" style="background:#eef2ff;color:{cor_borda}">{row["categoria"]}</span>'
            st.markdown(f"""
            <div class="event-card" style="border-left-color:{cor_borda}">
                <div class="event-date">{row['data'].strftime('%d %b %Y')}</div>
                <div class="event-title">{row['titulo']}</div>
                <div class="event-desc">{row['descricao']}</div>
                {tag_cat}{tag_imp}
            </div>
            """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div style="margin-top:3rem;padding:1rem;border-top:1px solid #dce3ec;
            text-align:center;font-size:0.78rem;color:#aaa;">
    Timeline SNS Portugal · construído com Streamlit + Plotly ·
    Dados via <a href="https://arquivo.pt" style="color:#1a4a7a">Arquivo.pt</a>
</div>
""", unsafe_allow_html=True)