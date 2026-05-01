"""
Sistema RAG — Busca Semântica sobre Documentos Históricos do SNS
================================================================
Pipeline:
  1. Ingestão   — carrega JSON (arquivo_pt_pipeline.py ou eventos_sns.json)
  2. Chunking   — divide textos longos em chunks com sobreposição
  3. Indexação  — embeddings (sentence-transformers) → índice FAISS persistido
  4. Retrieval  — busca semântica top-k por pergunta do utilizador
  5. Generation — resposta gerada pela API da Anthropic com os chunks recuperados

Instalar dependências:
    pip install sentence-transformers faiss-cpu anthropic numpy

Executar (modo interativo):
    python rag_pipeline.py --input arquivo_sns_data.json

Re-indexar forçando rebuild:
    python rag_pipeline.py --input arquivo_sns_data.json --reindex

Modo pergunta única (não interativo):
    python rag_pipeline.py --input arquivo_sns_data.json --query "Como funciona a triagem de Manchester?"

Sem chave Anthropic (apenas retrieval, sem geração):
    python rag_pipeline.py --input arquivo_sns_data.json --no-llm
"""

import argparse
import hashlib
import json
import logging
import os
import pickle
import re
import sys
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

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
# Configuração global
# ---------------------------------------------------------------------------
DEFAULT_MODEL_EMBED  = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_MODEL_LLM    = "claude-sonnet-4-20250514"
CHUNK_SIZE           = 400      # caracteres por chunk
CHUNK_OVERLAP        = 80       # sobreposição entre chunks consecutivos
TOP_K_DEFAULT        = 5        # documentos recuperados por query
MIN_CHUNK_LEN        = 60       # descarta chunks demasiado curtos
INDEX_DIR            = Path("rag_index")   # directório de persistência
ANTHROPIC_MAX_TOKENS = 1024

# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """Unidade mínima de retrieval — fragmento de um documento."""
    chunk_id:   str
    doc_id:     str
    titulo:     str
    data:       str
    categoria:  str
    url:        str
    texto:      str
    chunk_idx:  int   # índice dentro do documento de origem


@dataclass
class ResultadoBusca:
    """Par (chunk, score de similaridade) retornado pelo retrieval."""
    chunk:      Chunk
    score:      float   # cosine similarity ∈ [0, 1]
    rank:       int


@dataclass
class RespostaRAG:
    """Resposta final do sistema, com fontes rastreáveis."""
    pergunta:   str
    resposta:   str
    fontes:     list[ResultadoBusca]
    modelo_llm: str


# ---------------------------------------------------------------------------
# 1. CARREGAMENTO E NORMALIZAÇÃO
# ---------------------------------------------------------------------------

def _extrair_registos(dados) -> list[dict]:
    """Normaliza JSON de entrada: lista simples ou envelope {records:[...]}."""
    if isinstance(dados, list):
        return dados
    if isinstance(dados, dict):
        if "records" in dados:
            return dados["records"]
        for v in dados.values():
            if isinstance(v, list) and v:
                return v
    raise ValueError("Formato JSON não reconhecido. Esperado lista ou {'records':[...]}")


def carregar_documentos(caminho: str) -> list[dict]:
    """
    Carrega o JSON e normaliza campos para o schema interno:
      doc_id, titulo, data, categoria, url, texto
    """
    with open(caminho, encoding="utf-8") as f:
        dados = json.load(f)

    registos = _extrair_registos(dados)
    documentos = []

    for i, r in enumerate(registos):
        # Normaliza nomes de campo entre os dois formatos
        titulo    = r.get("titulo") or r.get("title", "")
        texto     = r.get("texto") or r.get("content") or r.get("descricao") or r.get("snippet", "")
        data      = r.get("data") or r.get("date", "")
        categoria = r.get("categoria") or r.get("keyword", "")
        url       = r.get("url") or r.get("originalURL", "")

        if not isinstance(texto, str) or len(texto.strip()) < MIN_CHUNK_LEN:
            continue

        doc_id = hashlib.md5(f"{url or titulo or i}".encode()).hexdigest()[:12]
        documentos.append({
            "doc_id":    doc_id,
            "titulo":    str(titulo).strip(),
            "data":      str(data)[:10],
            "categoria": str(categoria).strip(),
            "url":       str(url).strip(),
            "texto":     texto.strip(),
        })

    log.info("  Documentos carregados: %d  (de %d registos)", len(documentos), len(registos))
    return documentos


# ---------------------------------------------------------------------------
# 2. CHUNKING
# ---------------------------------------------------------------------------

def _limpar_texto(texto: str) -> str:
    """Remove HTML residual, URLs e normaliza espaços."""
    texto = re.sub(r"<[^>]+>", " ", texto)
    texto = re.sub(r"https?://\S+", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def chunkar_documento(doc: dict, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """
    Divide o texto de um documento em chunks com sobreposição.

    A sobreposição preserva contexto semântico nas fronteiras dos chunks,
    evitando que uma frase importante seja cortada entre dois chunks sem
    nenhum deles a conter por inteiro.
    """
    texto = _limpar_texto(doc["texto"])
    chunks = []
    inicio = 0
    idx = 0

    while inicio < len(texto):
        fim = inicio + chunk_size

        # Tenta terminar no fim de uma frase para não cortar no meio
        if fim < len(texto):
            fim_frase = texto.rfind(".", inicio, fim)
            if fim_frase > inicio + chunk_size // 2:
                fim = fim_frase + 1   # inclui o ponto final

        trecho = texto[inicio:fim].strip()

        if len(trecho) >= MIN_CHUNK_LEN:
            chunk_id = f"{doc['doc_id']}_{idx}"
            chunks.append(Chunk(
                chunk_id  = chunk_id,
                doc_id    = doc["doc_id"],
                titulo    = doc["titulo"],
                data      = doc["data"],
                categoria = doc["categoria"],
                url       = doc["url"],
                texto     = trecho,
                chunk_idx = idx,
            ))
            idx += 1

        inicio = fim - overlap   # passo com sobreposição
        if inicio >= fim:
            break

    return chunks


def construir_corpus(documentos: list[dict]) -> list[Chunk]:
    """Aplica chunking a todos os documentos e devolve lista plana de Chunks."""
    todos_chunks: list[Chunk] = []
    for doc in documentos:
        todos_chunks.extend(chunkar_documento(doc))
    log.info("  Total de chunks: %d  (média %.1f por documento)",
             len(todos_chunks), len(todos_chunks) / max(len(documentos), 1))
    return todos_chunks


# ---------------------------------------------------------------------------
# 3. EMBEDDINGS
# ---------------------------------------------------------------------------

def carregar_modelo_embed(nome: str = DEFAULT_MODEL_EMBED):
    """Carrega o modelo sentence-transformers (download automático na 1ª vez)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.error("sentence-transformers não instalado.")
        log.error("  Execute: pip install sentence-transformers")
        sys.exit(1)
    log.info("  Modelo de embeddings: %s", nome)
    return SentenceTransformer(nome)


def gerar_embeddings(textos: list[str], modelo) -> np.ndarray:
    """
    Gera embeddings L2-normalizados.
    Com normalização, produto interno == cosine similarity.
    """
    embeddings = modelo.encode(
        textos,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# 4. ÍNDICE FAISS
# ---------------------------------------------------------------------------

class IndiceRAG:
    """
    Encapsula o índice FAISS e o mapeamento chunk_id → Chunk.

    Usa IndexFlatIP (produto interno) que, com vetores normalizados,
    equivale a busca por cosine similarity — resultado ∈ [0, 1].
    """

    def __init__(self):
        self.index:    "faiss.Index | None" = None
        self.chunks:   list[Chunk] = []
        self.dim:      int = 0

    # ── Construção ──────────────────────────────────────────────────────────

    def construir(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Constrói o índice FAISS a partir de embeddings pré-calculados."""
        import faiss

        assert len(chunks) == embeddings.shape[0], "chunks e embeddings devem ter o mesmo tamanho"
        self.dim    = embeddings.shape[1]
        self.chunks = chunks

        # IndexFlatIP = produto interno exacto (sem aproximação)
        # Para corpora > 100k chunks, substituir por IndexIVFFlat ou IndexHNSWFlat
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        log.info("  FAISS IndexFlatIP: %d vetores  dim=%d", self.index.ntotal, self.dim)

    # ── Persistência ────────────────────────────────────────────────────────

    def guardar(self, directorio: Path) -> None:
        """Persiste índice FAISS + metadados em disco."""
        import faiss
        directorio.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directorio / "index.faiss"))
        with open(directorio / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        with open(directorio / "meta.json", "w") as f:
            json.dump({"dim": self.dim, "n_chunks": len(self.chunks)}, f)
        log.info("  Índice guardado em: %s", directorio)

    @classmethod
    def carregar(cls, directorio: Path) -> "IndiceRAG":
        """Carrega índice do disco. Lança FileNotFoundError se não existir."""
        import faiss
        idx = cls()
        idx.index  = faiss.read_index(str(directorio / "index.faiss"))
        with open(directorio / "chunks.pkl", "rb") as f:
            idx.chunks = pickle.load(f)
        with open(directorio / "meta.json") as f:
            meta = json.load(f)
        idx.dim = meta["dim"]
        log.info("  Índice carregado: %d chunks  dim=%d", len(idx.chunks), idx.dim)
        return idx

    # ── Busca ────────────────────────────────────────────────────────────────

    def buscar(self, query_emb: np.ndarray, top_k: int = TOP_K_DEFAULT) -> list[ResultadoBusca]:
        """
        Retorna os top_k chunks mais similares à query.

        Aplica deduplicação por doc_id: nunca retorna dois chunks do mesmo
        documento nos resultados, para maximizar cobertura de fontes.
        """
        # FAISS espera array 2D
        query_emb = query_emb.reshape(1, -1).astype(np.float32)

        # Busca k*4 candidatos para ter margem após deduplicação
        k_candidatos = min(top_k * 4, len(self.chunks))
        scores, indices = self.index.search(query_emb, k_candidatos)

        resultados: list[ResultadoBusca] = []
        vistos_docs: set[str] = set()

        for rank_raw, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < 0:   # FAISS retorna -1 para slots vazios
                continue
            chunk = self.chunks[idx]

            # Deduplicação por doc_id
            if chunk.doc_id in vistos_docs:
                continue
            vistos_docs.add(chunk.doc_id)

            resultados.append(ResultadoBusca(
                chunk = chunk,
                score = float(score),
                rank  = len(resultados) + 1,
            ))

            if len(resultados) >= top_k:
                break

        return resultados


# ---------------------------------------------------------------------------
# 5. GERAÇÃO DE RESPOSTA (LLM via Anthropic)
# ---------------------------------------------------------------------------

PROMPT_SISTEMA = """És um assistente especializado na história do Serviço Nacional de Saúde (SNS) português.
Respondes exclusivamente com base nos documentos fornecidos.
Quando a informação não está nos documentos, dizes claramente que não tens dados suficientes.
Citas sempre as fontes usando o formato [Fonte N].
Respondes em português europeu, de forma clara e estruturada."""


def _formatar_contexto(resultados: list[ResultadoBusca]) -> str:
    """Formata os chunks recuperados como contexto para o LLM."""
    blocos = []
    for r in resultados:
        c = r.chunk
        meta = f"[Fonte {r.rank}] {c.titulo or 'Sem título'}"
        if c.data:
            meta += f" ({c.data[:4]})"
        if c.categoria:
            meta += f" — {c.categoria}"
        blocos.append(f"{meta}\n{c.texto}")
    return "\n\n---\n\n".join(blocos)


def gerar_resposta_llm(
    pergunta: str,
    resultados: list[ResultadoBusca],
    modelo: str = DEFAULT_MODEL_LLM,
    api_key: Optional[str] = None,
) -> str:
    """
    Gera resposta usando a API da Anthropic com os chunks como contexto.

    A chave pode ser passada explicitamente ou via variável de ambiente
    ANTHROPIC_API_KEY.
    """
    try:
        import anthropic
    except ImportError:
        log.warning("anthropic não instalado — instale com: pip install anthropic")
        return "[LLM indisponível — instale o pacote anthropic]"

    chave = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not chave:
        log.warning("ANTHROPIC_API_KEY não definida — saltando geração LLM")
        return "[Sem chave API — defina ANTHROPIC_API_KEY para gerar respostas]"

    contexto = _formatar_contexto(resultados)
    mensagem_utilizador = (
        f"Com base nos documentos abaixo, responde à seguinte pergunta:\n\n"
        f"**Pergunta:** {pergunta}\n\n"
        f"**Documentos:**\n\n{contexto}"
    )

    cliente = anthropic.Anthropic(api_key=chave)
    resposta = cliente.messages.create(
        model=modelo,
        max_tokens=ANTHROPIC_MAX_TOKENS,
        system=PROMPT_SISTEMA,
        messages=[{"role": "user", "content": mensagem_utilizador}],
    )
    return resposta.content[0].text


# ---------------------------------------------------------------------------
# 6. PIPELINE RAG COMPLETO
# ---------------------------------------------------------------------------

class PipelineRAG:
    """
    Orquestra todo o ciclo RAG:
      indexar() → buscar() → responder()

    O índice é persistido em disco e reutilizado entre execuções.
    """

    def __init__(
        self,
        modelo_embed: str = DEFAULT_MODEL_EMBED,
        modelo_llm:   str = DEFAULT_MODEL_LLM,
        top_k:        int = TOP_K_DEFAULT,
        index_dir:    Path = INDEX_DIR,
        usar_llm:     bool = True,
        api_key:      Optional[str] = None,
    ):
        self.modelo_embed = modelo_embed
        self.modelo_llm   = modelo_llm
        self.top_k        = top_k
        self.index_dir    = index_dir
        self.usar_llm     = usar_llm
        self.api_key      = api_key
        self._embed_model = None   # lazy load
        self._indice: Optional[IndiceRAG] = None

    def _get_embed_model(self):
        if self._embed_model is None:
            self._embed_model = carregar_modelo_embed(self.modelo_embed)
        return self._embed_model

    # ── Indexação ────────────────────────────────────────────────────────────

    def indexar(self, caminho_json: str, forcar: bool = False) -> None:
        """
        Constrói e persiste o índice FAISS.

        Se o índice já existir em disco, carrega-o (a menos que forcar=True).
        """
        # Verificar se há índice guardado e se o ficheiro fonte não mudou
        hash_ficheiro = _hash_ficheiro(caminho_json)
        ficheiro_hash = self.index_dir / "source_hash.txt"

        indice_valido = (
            not forcar
            and (self.index_dir / "index.faiss").exists()
            and ficheiro_hash.exists()
            and ficheiro_hash.read_text().strip() == hash_ficheiro
        )

        if indice_valido:
            log.info("  Índice existente e actualizado — carregando do disco")
            self._indice = IndiceRAG.carregar(self.index_dir)
            return

        log.info("  Construindo índice de raiz ...")
        documentos = carregar_documentos(caminho_json)
        if not documentos:
            raise ValueError("Nenhum documento válido encontrado no ficheiro de entrada.")

        chunks = construir_corpus(documentos)
        textos = [c.texto for c in chunks]

        modelo = self._get_embed_model()
        log.info("  Gerando embeddings para %d chunks ...", len(textos))
        embeddings = gerar_embeddings(textos, modelo)

        self._indice = IndiceRAG()
        self._indice.construir(chunks, embeddings)
        self._indice.guardar(self.index_dir)

        # Guarda hash do ficheiro fonte para detectar mudanças futuras
        ficheiro_hash.write_text(hash_ficheiro)
        log.info("  Indexação completa.")

    # ── Retrieval ────────────────────────────────────────────────────────────

    def buscar(self, pergunta: str) -> list[ResultadoBusca]:
        """Converte a pergunta em embedding e executa busca no índice FAISS."""
        if self._indice is None:
            raise RuntimeError("Índice não inicializado. Execute indexar() primeiro.")

        modelo = self._get_embed_model()
        query_emb = modelo.encode(
            [pergunta],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        resultados = self._indice.buscar(query_emb, top_k=self.top_k)
        return resultados

    # ── Resposta ─────────────────────────────────────────────────────────────

    def responder(self, pergunta: str) -> RespostaRAG:
        """
        Pipeline completo: retrieval → (opcional) geração LLM → RespostaRAG.

        Se usar_llm=False ou a chave API não estiver disponível, devolve
        os chunks formatados sem geração, útil para debugging e testes.
        """
        resultados = self.buscar(pergunta)

        if not resultados:
            return RespostaRAG(
                pergunta  = pergunta,
                resposta  = "Não foram encontrados documentos relevantes para esta pergunta.",
                fontes    = [],
                modelo_llm = "none",
            )

        if self.usar_llm:
            texto_resposta = gerar_resposta_llm(
                pergunta, resultados,
                modelo  = self.modelo_llm,
                api_key = self.api_key,
            )
        else:
            # Modo retrieval-only: formata os chunks como resposta directa
            texto_resposta = _formatar_contexto(resultados)

        return RespostaRAG(
            pergunta   = pergunta,
            resposta   = texto_resposta,
            fontes     = resultados,
            modelo_llm = self.modelo_llm if self.usar_llm else "retrieval-only",
        )


# ---------------------------------------------------------------------------
# UTILITÁRIOS
# ---------------------------------------------------------------------------

def _hash_ficheiro(caminho: str) -> str:
    """SHA-256 dos primeiros 64KB do ficheiro — detecção rápida de mudanças."""
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()


def imprimir_resposta(rag_resp: RespostaRAG, largura: int = 72) -> None:
    """Imprime a resposta RAG formatada no terminal."""
    sep = "─" * largura
    print(f"\n{'═' * largura}")
    print(f"  PERGUNTA: {rag_resp.pergunta}")
    print(f"{'═' * largura}")
    print()

    # Resposta com quebra de linha
    for linha in rag_resp.resposta.splitlines():
        if linha.strip():
            print(textwrap.fill(linha, width=largura, initial_indent="  ",
                                subsequent_indent="  "))
        else:
            print()

    print(f"\n{sep}")
    print(f"  FONTES CONSULTADAS  (modelo: {rag_resp.modelo_llm})")
    print(sep)

    for r in rag_resp.fontes:
        c = r.chunk
        titulo_curto = (c.titulo[:55] + "…") if len(c.titulo) > 56 else c.titulo
        print(f"  [{r.rank}] score={r.score:.3f}  {titulo_curto}")
        print(f"       data={c.data[:7] or 'N/A'}  cat={c.categoria or 'N/A'}")
        if c.url:
            print(f"       {c.url[:70]}")
    print(f"{'═' * largura}\n")


def exportar_resposta(rag_resp: RespostaRAG, caminho: str) -> None:
    """Guarda a resposta e fontes em JSON para integração com outros sistemas."""
    dados = {
        "pergunta":   rag_resp.pergunta,
        "resposta":   rag_resp.resposta,
        "modelo_llm": rag_resp.modelo_llm,
        "fontes": [
            {
                "rank":      r.rank,
                "score":     round(r.score, 4),
                "titulo":    r.chunk.titulo,
                "data":      r.chunk.data,
                "categoria": r.chunk.categoria,
                "url":       r.chunk.url,
                "trecho":    r.chunk.texto[:300] + ("…" if len(r.chunk.texto) > 300 else ""),
            }
            for r in rag_resp.fontes
        ],
    }
    Path(caminho).write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("  Resposta exportada: %s", caminho)


# ---------------------------------------------------------------------------
# INTERFACE DE LINHA DE COMANDO
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RAG — Busca semântica sobre documentos históricos do SNS"
    )
    p.add_argument("--input",   "-i", default="arquivo_sns_data.json",
                   help="JSON de entrada (pipeline ou eventos_sns.json)")
    p.add_argument("--query",   "-q", default=None,
                   help="Pergunta única (omitir = modo interactivo)")
    p.add_argument("--top-k",   "-k", type=int, default=TOP_K_DEFAULT,
                   help=f"Número de documentos a recuperar (default: {TOP_K_DEFAULT})")
    p.add_argument("--reindex", "-r", action="store_true",
                   help="Força reconstrução do índice mesmo se existir")
    p.add_argument("--no-llm",  action="store_true",
                   help="Desactiva geração LLM — mostra apenas documentos recuperados")
    p.add_argument("--export",  "-e", default=None,
                   help="Exporta última resposta para ficheiro JSON")
    p.add_argument("--modelo-embed", default=DEFAULT_MODEL_EMBED,
                   help=f"Modelo sentence-transformers (default: {DEFAULT_MODEL_EMBED})")
    p.add_argument("--api-key", default=None,
                   help="Chave Anthropic (ou usar variável ANTHROPIC_API_KEY)")
    return p.parse_args()


def _loop_interactivo(rag: PipelineRAG, exportar: Optional[str]) -> None:
    """Ciclo interactivo de perguntas e respostas."""
    print("\n" + "═" * 60)
    print("  RAG SNS — Modo interactivo")
    print("  Digite a sua pergunta ou 'sair' para terminar.")
    print("═" * 60)

    perguntas_exemplo = [
        "Quando foi criado o SNS em Portugal?",
        "Como funciona a triagem de Manchester nas urgências?",
        "Quais foram os impactos da pandemia COVID-19 no SNS?",
        "O que são as Unidades de Saúde Familiar (USF)?",
        "Quais os cortes sofridos pelo SNS na crise financeira de 2011?",
    ]
    print("\n  Exemplos de perguntas:")
    for i, p in enumerate(perguntas_exemplo, 1):
        print(f"    {i}. {p}")
    print()

    ultima_resposta = None
    while True:
        try:
            pergunta = input("  Pergunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  A terminar.")
            break

        if not pergunta:
            continue
        if pergunta.lower() in ("sair", "exit", "quit", "q"):
            print("  Até logo!")
            break

        resp = rag.responder(pergunta)
        imprimir_resposta(resp)
        ultima_resposta = resp

    if exportar and ultima_resposta:
        exportar_resposta(ultima_resposta, exportar)


if __name__ == "__main__":
    args = _parse_args()

    if not Path(args.input).exists():
        log.error("Ficheiro não encontrado: %s", args.input)
        log.error("Gere o ficheiro com: python arquivo_pt_pipeline.py")
        sys.exit(1)

    # Inicializar pipeline
    rag = PipelineRAG(
        modelo_embed = args.modelo_embed,
        top_k        = args.top_k,
        usar_llm     = not args.no_llm,
        api_key      = args.api_key,
    )

    # Indexar (ou carregar índice existente)
    log.info("━━━━  INDEXAÇÃO  ━━━━")
    rag.indexar(args.input, forcar=args.reindex)

    # Executar pergunta(s)
    log.info("━━━━  RETRIEVAL  ━━━━")
    if args.query:
        resp = rag.responder(args.query)
        imprimir_resposta(resp)
        if args.export:
            exportar_resposta(resp, args.export)
    else:
        _loop_interactivo(rag, args.export)