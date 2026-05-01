"""
Pipeline de coleta de dados do Arquivo.pt sobre o SNS (Serviço Nacional de Saúde).

Melhorias v2:
  ✓ Keywords organizadas por categoria temática (campo 'categoria' no registo)
  ✓ Remoção de keywords duplicadas com preservação da ordem
  ✓ Checkpoints incrementais — retoma execução após falha de rede
  ✓ Progresso detalhado no terminal (barra de keywords + ETA)
  ✓ Relatório final por categoria
  ✓ Argumentos de linha de comandos (--max-items, --categoria, --dry-run, etc.)
  ✓ Logging simultâneo para consola + ficheiro pipeline.log

Fluxo:
  1. Pesquisa por keywords via TextSearch API
  2. Download do texto extraído de cada resultado
  3. Limpeza de HTML residual
  4. Normalização e estruturação dos registos (inclui campo 'categoria')
  5. Persistência incremental em JSON (checkpoint por keyword)
  6. Deduplicação por URL — nunca processa o mesmo documento duas vezes

Dependências: requests, beautifulsoup4
  pip install requests beautifulsoup4

Exemplos de uso:
  python arquivo_pt_pipeline.py                          # execução normal
  python arquivo_pt_pipeline.py --dry-run                # ver keywords sem correr
  python arquivo_pt_pipeline.py --categoria "Crise e Rutura"
  python arquivo_pt_pipeline.py --max-items 50 --no-resume
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


# ═══════════════════════════════════════════════════════════════════════════
# LOGGING — consola + ficheiro
# ═══════════════════════════════════════════════════════════════════════════

LOG_FILE = "pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# KEYWORDS — organizadas por categoria temática
# ═══════════════════════════════════════════════════════════════════════════
#
# O campo 'categoria' é guardado em cada registo e usado pelo NLP pipeline
# para análise temática segmentada.
#
# Duplicados entre categorias são detectados e removidos automaticamente
# em _build_keyword_list() — ver log de auditoria ao iniciar.

KEYWORD_CATEGORIES: dict[str, list[str]] = {

    "Estrutura e Organização": [
        "SNS",
        "Serviço Nacional de Saúde",
        "hospital público",
        "centro de saúde",
        "unidade de saúde familiar",
        "USF",
        "ACES saúde",
        "administração regional de saúde",
        "ARS Lisboa e Vale do Tejo",
        "ARS Norte",
        "ARS Algarve",
        "cuidados de saúde primários",
        "cuidados continuados",
        "rede nacional de cuidados continuados",
        "saúde Portugal",
    ],

    "Acesso e Listas de Espera": [
        "tempo de espera SNS",
        "listas de espera cirurgia",
        "listas de espera consultas",
        "doentes sem médico de família",
        "falta de médico de família",
        "médico família SNS",
        "linha SNS 24",
        "atendimento hospitalar",
        "demora atendimento",
    ],

    "Urgências e Emergências": [
        "urgência hospitalar",
        "triagem Manchester",
        "encerramento urgências",
        "urgência fechada",
        "sobrelotação hospital",
        "macas nos corredores",
        "transferência hospitalar",
        "INEM atraso",
        "ambulâncias demora",
        "internamento hospitalar",
    ],

    "Recursos Humanos": [
        "falta de médicos",
        "falta de enfermeiros",
        "escassez profissionais saúde",
        "médicos em falta",
        "enfermeiros em falta",
        "greve médicos",
        "greve enfermeiros",
        "horas extraordinárias médicos",
        "burnout médicos",
        "burnout enfermeiros",
        "contratação SNS",
        "salários médicos Portugal",
        "carreira médica",
        "médicos estrangeiros Portugal",
        "saída de médicos",
        "retenção profissionais saúde",
    ],

    "Financiamento e Política": [
        "orçamento SNS",
        "financiamento saúde Portugal",
        "investimento SNS",
        "parcerias público-privadas saúde",
        "PPP hospitais",
        "privatização saúde",
        "sector privado saúde",
        "seguros de saúde Portugal",
        "reformas SNS",
        "plano de emergência saúde",
        "ministério da saúde Portugal",
    ],

    "Crise e Rutura": [
        "crise no SNS",
        "colapso SNS",
        "sustentabilidade SNS",
        "colapso hospitalar",
        "serviço em rutura",
        "falta de condições",
        "degradação do SNS",
        "pressão sobre o SNS",
        "situação crítica saúde",
        "urgências em rutura",
        "caos nas urgências",
        "sistema sobrecarregado",
    ],
}


def _build_keyword_list() -> tuple[list[tuple[str, str]], list[str]]:
    """
    Achata KEYWORD_CATEGORIES numa lista ordenada de (keyword, categoria)
    sem duplicados. Preserva a ordem de declaração.

    Retorna:
      - lista de tuplos únicos (keyword, categoria)
      - lista de duplicados removidos (para log de auditoria)
    """
    seen:       set[str]              = set()
    duplicates: list[str]             = []
    result:     list[tuple[str, str]] = []

    for categoria, keywords in KEYWORD_CATEGORIES.items():
        for kw in keywords:
            kw_norm = kw.strip()
            if kw_norm.lower() in seen:
                duplicates.append(f"'{kw_norm}' em [{categoria}]")
                continue
            seen.add(kw_norm.lower())
            result.append((kw_norm, categoria))

    return result, duplicates


# Lista global única de (keyword, categoria)
KEYWORD_LIST, DUPLICATE_KEYWORDS = _build_keyword_list()

# Mapa rápido para lookup em build_record
KEYWORD_TO_CATEGORY: dict[str, str] = {kw: cat for kw, cat in KEYWORD_LIST}


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════════════════

TEXT_SEARCH_URL = "http://arquivo.pt/textsearch"

# Resultados por keyword (a API é lenta — cada keyword demora 45-90s)
DEFAULT_MAX_ITEMS = 50

# Timeouts (segundos)
SEARCH_TIMEOUT  = 120   # não reduzir abaixo de 90s
CONTENT_TIMEOUT = 60

# Retries em caso de falha
RETRY_ATTEMPTS = 3
RETRY_BACKOFF  = 3.0    # multiplica pelo número da tentativa

# Filtro de conteúdo mínimo
MIN_CONTENT_CHARS = 50

# Intervalo temporal (None = sem filtro de data)
DATE_FROM: Optional[str] = None   # ex: "20100101000000"
DATE_TO:   Optional[str] = None   # ex: "20231231235959"

# Ficheiros
OUTPUT_PATH     = "arquivo_sns_data.json"
CHECKPOINT_PATH = "arquivo_sns_checkpoint.json"


# ═══════════════════════════════════════════════════════════════════════════
# CHECKPOINT — persistência incremental
# ═══════════════════════════════════════════════════════════════════════════

def checkpoint_load(path: str) -> tuple[list[dict], set[str], set[str]]:
    """
    Carrega estado de uma execução anterior.

    Retorna:
      - registos já recolhidos
      - URLs já vistos (deduplicação)
      - keywords já concluídas (para saltar)
    """
    cp = Path(path)
    if not cp.exists():
        return [], set(), set()
    try:
        data          = json.loads(cp.read_text(encoding="utf-8"))
        records       = data.get("records", [])
        seen_urls     = set(data.get("seen_urls", []))
        done_keywords = set(data.get("done_keywords", []))
        logger.info(
            "♻  Checkpoint encontrado: %d registos, %d keywords concluídas",
            len(records), len(done_keywords)
        )
        return records, seen_urls, done_keywords
    except Exception as exc:
        logger.warning("Checkpoint corrompido — começando do início. (%s)", exc)
        return [], set(), set()


def checkpoint_save(
    path: str,
    records: list[dict],
    seen_urls: set[str],
    done_keywords: set[str],
) -> None:
    """Persiste o estado actual após cada keyword concluída."""
    data = {
        "saved_at":      datetime.now(timezone.utc).isoformat(),
        "records":       records,
        "seen_urls":     list(seen_urls),
        "done_keywords": list(done_keywords),
    }
    cp = Path(path)
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def checkpoint_delete(path: str) -> None:
    """Remove o checkpoint após conclusão bem-sucedida."""
    cp = Path(path)
    if cp.exists():
        cp.unlink()
        logger.info("Checkpoint removido após conclusão.")


# ═══════════════════════════════════════════════════════════════════════════
# FILTRO DE DOMÍNIO PORTUGUÊS
# ═══════════════════════════════════════════════════════════════════════════

_PT_DOMAIN_SUFFIXES  = (".pt",)
_PT_DOMAIN_ALLOWLIST = (
    "sns.gov.pt", "dgs.pt", "acss.min-saude.pt", "spms.pt",
    "publico.pt", "dn.pt", "rtp.pt", "cmjornal.pt", "jn.pt",
    "expresso.pt", "sicnoticias.pt", "tvi24.pt", "sapo.pt",
    "min-saude.pt", "portaldasaude.pt", "ordemdosmedicos.pt",
    "omedicamento.com.pt",
)


def _is_portuguese_domain(url: str) -> bool:
    """Devolve True se o URL pertence a um domínio português relevante."""
    try:
        host = urlparse(url).netloc.lower().split(":")[0]
        return host.endswith(_PT_DOMAIN_SUFFIXES) or any(
            host == d or host.endswith("." + d) for d in _PT_DOMAIN_ALLOWLIST
        )
    except Exception:
        return True  # em caso de dúvida, não descarta


# ═══════════════════════════════════════════════════════════════════════════
# CAMADA DE ACESSO À API
# ═══════════════════════════════════════════════════════════════════════════

def _build_params(keyword: str, max_items: int) -> dict:
    """
    Constrói os parâmetros de query para a API TextSearch.

    NOTA: 'fields' foi removido intencionalmente — quando presente, a API
    omite 'linkToExtractedText', impedindo o download do conteúdo.
    """
    params: dict = {
        "q":          keyword,
        "maxItems":   max_items,
        "siteSearch": ".pt",
    }
    if DATE_FROM:
        params["from"] = DATE_FROM
    if DATE_TO:
        params["to"] = DATE_TO
    return params


def fetch_search_results(
    keyword: str,
    max_items: int = DEFAULT_MAX_ITEMS,
) -> list[dict]:
    """
    Consulta a API TextSearch do Arquivo.pt para uma keyword.
    Retorna lista de items ou [] em caso de erro persistente.
    """
    params = _build_params(keyword, max_items)

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            logger.info(
                "  🔎 Pesquisando '%s' (tentativa %d/%d)",
                keyword, attempt, RETRY_ATTEMPTS
            )
            resp = requests.get(TEXT_SEARCH_URL, params=params, timeout=SEARCH_TIMEOUT)
            resp.raise_for_status()
            items = resp.json().get("response_items", [])
            logger.info("     → %d resultados encontrados", len(items))
            return items

        except requests.exceptions.HTTPError as exc:
            logger.warning(
                "     HTTP %s para '%s': %s",
                exc.response.status_code, keyword, exc
            )
        except requests.exceptions.ConnectionError:
            logger.warning("     Erro de ligação ao pesquisar '%s'", keyword)
        except requests.exceptions.Timeout:
            logger.warning("     Timeout ao pesquisar '%s'", keyword)
        except (ValueError, KeyError) as exc:
            logger.error("     Resposta inesperada para '%s': %s", keyword, exc)
            return []   # erro não recuperável — não repetir

        if attempt < RETRY_ATTEMPTS:
            wait = RETRY_BACKOFF * attempt
            logger.info("     Aguardando %.0fs antes de retry…", wait)
            time.sleep(wait)

    logger.error("  ✗ Falha definitiva ao pesquisar '%s'", keyword)
    return []


def fetch_extracted_text(url: str) -> Optional[str]:
    """
    Descarrega o texto extraído pelo Arquivo.pt para um URL.
    Retorna o texto bruto (pode conter HTML residual) ou None.
    """
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(url, timeout=CONTENT_TIMEOUT)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
            return resp.text
        except requests.exceptions.RequestException as exc:
            logger.debug(
                "    Tentativa %d/%d falhou: %s — %s",
                attempt, RETRY_ATTEMPTS, url, exc
            )
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BACKOFF)

    logger.warning("    ✗ Não foi possível obter texto: %s", url)
    return None


# ═══════════════════════════════════════════════════════════════════════════
# LIMPEZA DE TEXTO
# ═══════════════════════════════════════════════════════════════════════════

def clean_html(raw: str) -> str:
    """Remove tags HTML e normaliza espaços em branco via BeautifulSoup."""
    if not raw:
        return ""
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    text  = soup.get_text(separator=" ")
    lines = (line.strip() for line in text.splitlines())
    return " ".join(word for line in lines for word in line.split())


# ═══════════════════════════════════════════════════════════════════════════
# TRANSFORMAÇÃO DE ITEM → REGISTO
# ═══════════════════════════════════════════════════════════════════════════

def parse_timestamp(ts: str) -> str:
    """Converte timestamp Arquivo.pt (YYYYMMDDHHmmss) → ISO-8601."""
    try:
        return datetime.strptime(ts, "%Y%m%d%H%M%S").isoformat()
    except (ValueError, TypeError):
        return ts or ""


def build_record(item: dict, keyword: str, categoria: str) -> Optional[dict]:
    """
    Constrói um registo normalizado a partir de um item da API.

    Inclui download, limpeza e categorização do conteúdo.
    Retorna None se não for possível obter conteúdo útil.
    """
    url           = item.get("originalURL", "")
    archive_url   = item.get("linkToArchive", "")
    extracted_url = item.get("linkToExtractedText", "")
    title         = item.get("title", "").strip()
    timestamp     = parse_timestamp(item.get("tstamp", ""))

    if not url:
        logger.debug("    Item sem URL — ignorado")
        return None

    if not _is_portuguese_domain(url):
        logger.debug("    Domínio não-PT ignorado: %s", url)
        return None

    if not extracted_url:
        logger.warning(
            "    'linkToExtractedText' ausente para %s  (chaves: %s)",
            url, list(item.keys())
        )
        return None

    raw_content = fetch_extracted_text(extracted_url)
    content     = clean_html(raw_content) if raw_content else ""

    if len(content) < MIN_CONTENT_CHARS:
        logger.warning(
            "    Conteúdo insuficiente (%d chars) para %s — ignorado",
            len(content), url
        )
        return None

    return {
        "keyword":      keyword,
        "categoria":    categoria,          # ← campo novo v2
        "title":        title,
        "date":         timestamp,
        "url":          url,
        "archive_url":  archive_url,
        "content":      content,
        "char_count":   len(content),
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS DE TERMINAL
# ═══════════════════════════════════════════════════════════════════════════

def _progress_bar(current: int, total: int, width: int = 28) -> str:
    filled = int(width * current / max(total, 1))
    bar    = "█" * filled + "░" * (width - filled)
    pct    = int(100 * current / max(total, 1))
    return f"[{bar}] {pct:3d}%  {current}/{total}"


def _sep(char: str = "─", width: int = 70) -> None:
    print(char * width)


def print_keyword_summary() -> None:
    """Imprime resumo das keywords carregadas — útil para auditoria."""
    _sep("═")
    print(f"  KEYWORDS CARREGADAS: {len(KEYWORD_LIST)} únicas")
    _sep()
    for cat, keywords in KEYWORD_CATEGORIES.items():
        kws = [kw for kw, c in KEYWORD_LIST if c == cat]
        print(f"\n  [{cat}]  ({len(kws)} keywords)")
        for kw in kws:
            print(f"    • {kw}")
    if DUPLICATE_KEYWORDS:
        _sep()
        print(f"\n  ⚠  {len(DUPLICATE_KEYWORDS)} duplicado(s) removido(s):")
        for d in DUPLICATE_KEYWORDS:
            print(f"    – {d}")
    _sep("═")


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(
    keyword_list:          list[tuple[str, str]] = KEYWORD_LIST,
    max_items_per_keyword: int                   = DEFAULT_MAX_ITEMS,
    output_path:           str                   = OUTPUT_PATH,
    checkpoint_path:       str                   = CHECKPOINT_PATH,
    resume:                bool                  = True,
) -> list[dict]:
    """
    Executa o pipeline completo com checkpoint incremental.

    Parâmetros:
      keyword_list          — lista de (keyword, categoria)
      max_items_per_keyword — max resultados por keyword
      output_path           — destino do JSON final
      checkpoint_path       — ficheiro de checkpoint entre keywords
      resume                — se True, retoma execução interrompida

    Retorna lista de registos recolhidos.
    """

    # ── Cabeçalho ──────────────────────────────────────────────────────────
    _sep("═")
    print("  SNS Arquivo.pt — Pipeline v2")
    print(f"  {len(keyword_list)} keywords  ·  {max_items_per_keyword} resultados/keyword")
    print(f"  Saída      : {output_path}")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Log        : {LOG_FILE}")
    _sep("═")

    if DUPLICATE_KEYWORDS:
        logger.info(
            "⚠  %d keyword(s) duplicada(s) removida(s): %s",
            len(DUPLICATE_KEYWORDS), "; ".join(DUPLICATE_KEYWORDS)
        )

    # ── Carregar checkpoint ────────────────────────────────────────────────
    if resume:
        all_records, seen_urls, done_keywords = checkpoint_load(checkpoint_path)
    else:
        all_records, seen_urls, done_keywords = [], set(), set()
        logger.info("Modo --no-resume: ignorando checkpoint anterior.")

    # ── Contadores ─────────────────────────────────────────────────────────
    n_total       = len(keyword_list)
    n_new         = 0
    n_skipped     = 0
    stats_by_cat: dict[str, int] = {}
    start_time    = time.time()

    # ── Iteração por keyword ───────────────────────────────────────────────
    for idx, (keyword, categoria) in enumerate(keyword_list, 1):

        print(f"\n{_progress_bar(idx - 1, n_total)}  [{categoria}]  '{keyword}'")

        # Saltar keywords já concluídas (retoma de checkpoint)
        if keyword in done_keywords:
            logger.info("  ⏭  '%s' já processada — a saltar", keyword)
            n_skipped += 1
            continue

        # Pesquisar na API
        items        = fetch_search_results(keyword, max_items_per_keyword)
        n_kw_novos   = 0

        for item in items:
            url = item.get("originalURL", "")

            # Deduplicação por URL
            if url and url in seen_urls:
                logger.debug("  Duplicado ignorado: %s", url)
                continue
            if url:
                seen_urls.add(url)

            record = build_record(item, keyword, categoria)
            if record:
                all_records.append(record)
                n_new        += 1
                n_kw_novos   += 1
                stats_by_cat[categoria] = stats_by_cat.get(categoria, 0) + 1
                logger.info(
                    "  ✓ [%s] %s",
                    categoria[:22], record["title"] or url
                )

        # Marcar como concluída e guardar checkpoint
        done_keywords.add(keyword)
        checkpoint_save(checkpoint_path, all_records, seen_urls, done_keywords)

        logger.info(
            "  Keyword '%s' concluída — %d novos (total acumulado: %d)",
            keyword, n_kw_novos, len(all_records)
        )

        # ETA estimado
        elapsed    = time.time() - start_time
        done_count = idx - n_skipped
        if done_count > 0:
            avg        = elapsed / done_count
            remaining  = n_total - idx
            eta_min    = (avg * remaining) / 60
            logger.info("  ⏱  ETA estimado: %.0f min", eta_min)

        # Pausa entre keywords para não sobrecarregar a API
        time.sleep(1.0)

    # ── Guardar resultado final ─────────────────────────────────────────────
    save_to_json(all_records, output_path, max_items_per_keyword)
    checkpoint_delete(checkpoint_path)

    # ── Relatório final ────────────────────────────────────────────────────
    elapsed_total = time.time() - start_time
    _sep("═")
    print(f"\n  ✅ Pipeline concluído em {elapsed_total / 60:.1f} min")
    print(f"  Total de registos    : {len(all_records)}")
    print(f"  Novos nesta sessão   : {n_new}")
    print(f"  Keywords processadas : {len(done_keywords)}")
    _sep()
    print("  Registos por categoria:")
    for cat, n in sorted(stats_by_cat.items(), key=lambda x: -x[1]):
        bar = "█" * min(32, n)
        print(f"    {cat:<36} {bar} {n}")
    _sep("═")
    print(f"  Ficheiro: {output_path}\n")

    return all_records


# ═══════════════════════════════════════════════════════════════════════════
# PERSISTÊNCIA FINAL
# ═══════════════════════════════════════════════════════════════════════════

def save_to_json(
    records: list[dict],
    path: str,
    max_items: int = DEFAULT_MAX_ITEMS,
) -> None:
    """Guarda registos em JSON estruturado com metadados completos."""
    categorias = sorted({r.get("categoria", "") for r in records} - {""})

    output = {
        "metadata": {
            "source":                       "Arquivo.pt TextSearch API",
            "pipeline_version":             "2.0",
            "date_from":                    DATE_FROM,
            "date_to":                      DATE_TO,
            "max_items_per_kw":             max_items,
            "total_keywords":               len(KEYWORD_LIST),
            "duplicate_keywords_removed":   len(DUPLICATE_KEYWORDS),
            "total_records":                len(records),
            "categorias":                   categorias,
            "generated_at":                 datetime.now(timezone.utc).isoformat(),
        },
        "keyword_categories": {
            cat: [kw for kw, c in KEYWORD_LIST if c == cat]
            for cat in KEYWORD_CATEGORIES
        },
        "records": records,
    }

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Dados guardados em: %s  (%d registos)", out.resolve(), len(records))


# ═══════════════════════════════════════════════════════════════════════════
# ENTRADA DE EXECUÇÃO
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Pipeline Arquivo.pt SNS v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python arquivo_pt_pipeline.py
  python arquivo_pt_pipeline.py --dry-run
  python arquivo_pt_pipeline.py --categoria "Crise e Rutura"
  python arquivo_pt_pipeline.py --max-items 50 --no-resume
  python arquivo_pt_pipeline.py --output dados_sns_v2.json
        """,
    )
    parser.add_argument(
        "--max-items", type=int, default=DEFAULT_MAX_ITEMS,
        help=f"Resultados por keyword (padrão: {DEFAULT_MAX_ITEMS})"
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_PATH,
        help=f"Ficheiro de saída (padrão: {OUTPUT_PATH})"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignora checkpoint e começa do início"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Apenas mostra as keywords sem executar o pipeline"
    )
    parser.add_argument(
        "--categoria", type=str, default=None,
        help="Corre apenas uma categoria (ex: 'Crise e Rutura')"
    )
    args = parser.parse_args()

    # Modo diagnóstico
    if args.dry_run:
        print_keyword_summary()
        raise SystemExit(0)

    # Filtrar por categoria (opcional)
    kw_list = KEYWORD_LIST
    if args.categoria:
        kw_list = [(kw, cat) for kw, cat in KEYWORD_LIST if cat == args.categoria]
        if not kw_list:
            print(f"❌ Categoria '{args.categoria}' não encontrada.")
            print("Categorias disponíveis:")
            for cat in KEYWORD_CATEGORIES:
                print(f"  • {cat}")
            raise SystemExit(1)
        logger.info(
            "Filtrando para categoria: '%s' (%d keywords)",
            args.categoria, len(kw_list)
        )

    # Executar pipeline
    records = run_pipeline(
        keyword_list          = kw_list,
        max_items_per_keyword = args.max_items,
        output_path           = args.output,
        checkpoint_path       = CHECKPOINT_PATH,
        resume                = not args.no_resume,
    )

    # Amostra de registo no terminal
    if records:
        sample = records[0]
        _sep()
        print("Exemplo de registo recolhido:")
        print(f"  Categoria : {sample.get('categoria', 'N/A')}")
        print(f"  Keyword   : {sample['keyword']}")
        print(f"  Título    : {sample['title']}")
        print(f"  Data      : {sample['date']}")
        print(f"  URL       : {sample['url']}")
        print(f"  Conteúdo  : {sample['content'][:200]}…")
        _sep()