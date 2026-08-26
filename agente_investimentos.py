"""
Agente de Monitoramento e Análise de Ativos Financeiros
------------------------------------------------------
Pipeline automatizado para coleta, consolidação e análise inteligente de portfólio.

Fontes de dados integradas:
  - Yahoo Finance (yfinance): Cotações diárias (B3 e Globais) e volatilidade.
  - Banco Central do Brasil (SGS API): Indicadores macroeconômicos (IPCA 12m, Taxa Selic).
  - Tesouro Direto (B3 API): Taxas de rendimento e spreads de títulos públicos.
  - Google News (RSS): Coleta direcionada de notícias recentes por ativo e gestão.
  - Fundamentus: Indicadores de análise fundamentalista (P/L, P/VP, Dividend Yield, ROE).
  - Google Gemini API: Consolidação de contexto e geração de relatório analítico diário.

Exemplo de schema esperado para 'carteira.json':
{
  "acoes_br":   [{"ticker": "PETR4", "nome": "Petrobras"}],
  "etfs_br":    [{"ticker": "BOVA11", "nome": "iShares Ibovespa"}],
  "fiis":       [{"ticker": "KNRI11", "nome": "Kinea Renda Imobiliária"}],
  "acoes_intl": [{"ticker": "AAPL", "nome": "Apple Inc."}],
  "reits":      [{"ticker": "AMT", "nome": "American Tower"}],
  "etfs_intl":  [{"ticker": "VOO", "nome": "Vanguard S&P 500"}]
}
"""

import json
import os
import re
import smtplib
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import feedparser
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ── Configurações de Ambiente ─────────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SMTP_EMAIL     = os.getenv("SMTP_EMAIL", "")
SMTP_PASSWORD  = os.getenv("SMTP_PASSWORD", "")
DEST_EMAIL     = os.getenv("DEST_EMAIL", SMTP_EMAIL)
CARTEIRA_FILE  = "carteira.json"
PROMPT_FILE    = "prompt.txt"
MODELO_GEMINI  = "gemini-2.5-flash"

HEADERS_BROWSER = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
}

HEADERS_TD = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Referer": "https://www.tesourodireto.com.br/titulos/precos-e-taxas.htm",
}

# ── Carga de Arquivos de Configuração ─────────────────────────────────────────

def carregar_prompt(caminho: str = PROMPT_FILE) -> str:
    """Carrega as diretrizes do sistema para a IA a partir de arquivo externo."""
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"[config] Arquivo {caminho} não encontrado. Usando prompt padrão.")
        return "Você é um assistente financeiro pessoal. Analise os ativos e forneça um relatório objetivo."

def carregar_carteira(caminho: str = CARTEIRA_FILE) -> dict:
    """Lê a composição da carteira de investimentos a partir do arquivo JSON."""
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)

# ── Coleta de Dados de Mercado ────────────────────────────────────────────────

def buscar_cotacoes_yf(tickers: list[str], sufixo: str = "") -> dict:
    """Busca preço de fechamento e variação percentual dos ativos via Yahoo Finance."""
    resultado = {}
    for t in tickers:
        simbolo = f"{t}{sufixo}"
        try:
            ticker_obj = yf.Ticker(simbolo)
            hist = ticker_obj.history(period="5d").dropna()
            if hist.empty:
                continue

            preco_atual = float(hist["Close"].iloc[-1])
            preco_anterior = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else preco_atual
            variacao_pct = ((preco_atual - preco_anterior) / preco_anterior) * 100 if preco_anterior else 0.0
            variacao_abs = preco_atual - preco_anterior

            resultado[t] = {
                "nome": ticker_obj.info.get("longName", t),
                "preco": round(preco_atual, 2),
                "preco_ant": round(preco_anterior, 2),
                "variacao_pct": round(variacao_pct, 2),
                "variacao_abs": round(variacao_abs, 2),
                "moeda": "BRL" if sufixo == ".SA" else "USD",
            }
        except Exception as e:
            print(f"[yfinance:{t}] Erro: {e}")
    return resultado

def buscar_indicadores_bcb() -> tuple[list[dict], float]:
    """Coleta indicadores macroeconômicos oficiais da API do SGS (Banco Central)."""
    indicadores = []
    ipca_float = 0.0
    try:
        # IPCA Acumulado 12 meses (Série 13522)
        r_ipca = requests.get(
            "https://api.bcb.gov.br/dados/serie/bcdata.sgs.13522/dados/ultimos/1?formato=json",
            timeout=10,
        )
        if r_ipca.status_code == 200:
            ipca_float = float(r_ipca.json()[0]["valor"])
            indicadores.append({
                "nome": "Inflação (IPCA 12 meses)",
                "valor": f"{ipca_float:.2f}%",
            })

        # Taxa Selic Meta (Série 432)
        r_selic = requests.get(
            "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados/ultimos/1?formato=json",
            timeout=10,
        )
        if r_selic.status_code == 200:
            selic_val = float(r_selic.json()[0]["valor"])
            indicadores.append({
                "nome": "Taxa Selic (Meta)",
                "valor": f"{selic_val:.2f}% a.a.",
            })
    except Exception as e:
        print(f"[bcb] Erro na requisição: {e}")
    return indicadores, ipca_float

def buscar_tesouro_direto(ipca_atual: float = 0.0) -> list[dict]:
    """Busca cotações e taxas dos títulos públicos na API do Tesouro Direto."""
    try:
        url = "https://www.tesourodireto.com.br/json/br/com/b3/tesourodireto/service/api/treasurybond/price_and_yield.json"
        resp = requests.get(url, headers=HEADERS_TD, timeout=15)
        resp.raise_for_status()
        titulos = resp.json().get("response", {}).get("TrsrBdTradgList", [])

        alvos = ["Tesouro IPCA+ 2032", "Tesouro IPCA+ 2040", "Tesouro Selic 2032"]
        encontrados = []

        for t in titulos:
            nome = t["TrsrBd"].get("nm", "")
            for alvo in alvos:
                if alvo.lower() in nome.lower():
                    spread = float(t["TrsrBd"].get("anulInvstmtRate", 0))
                    preco = float(t["TrsrBd"].get("untrInvstmtVal", 0))

                    if "IPCA" in nome.upper() and ipca_atual > 0:
                        nominal = ((1 + ipca_atual / 100) * (1 + spread / 100) - 1) * 100
                    else:
                        nominal = spread

                    encontrados.append({
                        "nome": nome,
                        "taxa_spread": spread,
                        "preco_compra": preco,
                        "nominal_estimado": round(nominal, 2),
                        "eh_ipca": "IPCA" in nome.upper(),
                    })
        return encontrados
    except Exception as e:
        print(f"[tesouro] Serviço indisponível ou bloqueado temporariamente: {e}")
        return []

def buscar_fundamentus(ticker: str) -> dict:
    """Extrai múltiplos fundamentalistas via web scraping estruturado."""
    try:
        url = f"https://www.fundamentus.com.br/detalhes.php?papel={ticker}"
        resp = requests.get(url, headers=HEADERS_BROWSER, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")

        dados_brutos = {}
        for td in soup.find_all("td"):
            span = td.find("span", class_="txt")
            if not span:
                continue
            label = span.get_text(strip=True)
            prox = td.find_next_sibling("td")
            if prox:
                val_span = prox.find("span")
                if val_span:
                    dados_brutos[label] = val_span.get_text(strip=True)

        def _sanitizar(val: str | None) -> str | None:
            if val is None:
                return None
            val = val.strip()
            return None if val in ("N/A", "", "-", "0,00%", "0%") else val

        resultado = {}
        mapeamento = [
            ("P/L", "P/L"),
            ("P/L\xa0", "P/L"),
            ("P/VP", "P/VP"),
            ("Div. Yield", "DY"),
            ("DY", "DY"),
            ("ROE", "ROE"),
        ]

        for chave_orig, chave_saida in mapeamento:
            v = _sanitizar(dados_brutos.get(chave_orig))
            if v and chave_saida not in resultado:
                resultado[chave_saida] = v

        return resultado
    except Exception as e:
        print(f"[fundamentus:{ticker}] Erro: {e}")
        return {}

def buscar_noticias(query: str, max_n: int = 3) -> list[str]:
    """Consulta o feed RSS do Google News filtrando eventos dos últimos 3 dias."""
    query_fmt = query.replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={query_fmt}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
    cutoff = datetime.now() - timedelta(days=3)
    noticias = []
    try:
        for entry in feedparser.parse(url).entries[:10]:
            try:
                pub = datetime(*entry.published_parsed[:6])
                if pub < cutoff:
                    continue
            except Exception:
                pass
            titulo = re.sub(r"<[^>]+>", "", entry.title).strip()
            noticias.append(titulo)
            if len(noticias) >= max_n:
                break
    except Exception as e:
        print(f"[news:{query}] Erro: {e}")
    return noticias

# ── Pipeline de Análise por LLM ───────────────────────────────────────────────

def montar_contexto(
    carteira: dict,
    cotacoes_br: dict,
    cotacoes_intl: dict,
    indicadores: list[dict],
    ipca_atual: float,
    tesouro: list[dict],
    fund_dados: dict,
) -> str:
    """Constrói o payload textual com todos os dados agregados para o prompt do modelo."""
    ctx = "CARTEIRA DO INVESTIDOR — DADOS DO DIA:\n\n"

    ctx += "=== AÇÕES BRASILEIRAS ===\n"
    for a in carteira.get("acoes_br", []):
        t = a["ticker"]
        d = cotacoes_br.get(t, {})
        pct = d.get("variacao_pct", 0)
        sinal = "+" if pct >= 0 else ""
        ctx += f"• {t}: R${d.get('preco', 0):.2f} ({sinal}{pct:.2f}% no dia)\n"
        f = fund_dados.get(t, {})
        if f:
            partes = [f"{k}={v}" for k, v in f.items() if v]
            if partes:
                ctx += f"  Fundamentos: {' | '.join(partes)}\n"
        noticias = buscar_noticias(f"{t} {a.get('nome', '')}")
        ctx += "  Notícias (3d): " + (" | ".join(noticias) if noticias else "Sem notícias relevantes") + "\n"

    ctx += "\n=== ETFs BR ===\n"
    for a in carteira.get("etfs_br", []):
        t = a["ticker"]
        d = cotacoes_br.get(t, {})
        pct = d.get("variacao_pct", 0)
        sinal = "+" if pct >= 0 else ""
        ctx += f"• {t}: R${d.get('preco', 0):.2f} ({sinal}{pct:.2f}% no dia)\n"
        noticias = buscar_noticias(t)
        ctx += "  Notícias (3d): " + (" | ".join(noticias) if noticias else "Sem notícias relevantes") + "\n"

    ctx += "\n=== FIIs ===\n"
    for a in carteira.get("fiis", []):
        t = a["ticker"]
        d = cotacoes_br.get(t, {})
        pct = d.get("variacao_pct", 0)
        sinal = "+" if pct >= 0 else ""
        ctx += f"• {t}: R${d.get('preco', 0):.2f} ({sinal}{pct:.2f}% no dia)\n"
        f = fund_dados.get(t, {})
        if f:
            partes = [f"{k}={v}" for k, v in f.items() if v]
            if partes:
                ctx += f"  Fundamentos: {' | '.join(partes)}\n"
        noticias = buscar_noticias(f"{t} {a.get('nome', '')} gestora fundo imobiliario")
        ctx += "  Notícias (3d): " + (" | ".join(noticias) if noticias else "Sem notícias relevantes") + "\n"

    ctx += "\n=== INTERNACIONAIS (Ações, REITs e ETFs) ===\n"
    todos_intl = carteira.get("acoes_intl", []) + carteira.get("reits", []) + carteira.get("etfs_intl", [])
    for a in todos_intl:
        t = a["ticker"]
        d = cotacoes_intl.get(t, {})
        pct = d.get("variacao_pct", 0)
        sinal = "+" if pct >= 0 else ""
        ctx += f"• {t}: US${d.get('preco', 0):.2f} ({sinal}{pct:.2f}% no dia)\n"
        noticias = buscar_noticias(f"{t} {a.get('nome', '')}")
        ctx += "  Notícias (3d): " + (" | ".join(noticias) if noticias else "Sem notícias relevantes") + "\n"

    ctx += "\n=== CENÁRIO MACROECONÔMICO ===\n"
    for ind in indicadores:
        ctx += f"• {ind['nome']}: {ind['valor']}\n"

    ctx += "\n=== TESOURO DIRETO ===\n"
    for t in tesouro:
        spread = t.get("taxa_spread", 0)
        nominal = t.get("nominal_estimado", 0)
        preco = t.get("preco_compra", 0)
        if t.get("eh_ipca"):
            ctx += (
                f"• {t['nome']}: R${preco:.2f} | "
                f"Taxa real: IPCA + {spread:.2f}% a.a. | "
                f"IPCA 12m: {ipca_atual:.2f}% | "
                f"Retorno nominal est.: ~{nominal:.2f}% a.a.\n"
            )
        else:
            ctx += f"• {t['nome']}: R${preco:.2f} | Selic + {spread:.4f}% a.a.\n"

    ctx += "\nGere o briefing completo agora."
    return ctx

def analisar_carteira(contexto: str, system_prompt: str) -> str:
    """Envia o contexto consolidado ao Google Gemini e retorna a avaliação financeira."""
    try:
        if not GEMINI_API_KEY:
            return "Erro: GEMINI_API_KEY não configurada no ambiente."
        client = genai.Client(api_key=GEMINI_API_KEY)
        resp = client.models.generate_content(
            model=MODELO_GEMINI,
            contents=contexto,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,
            ),
        )
        return resp.text.replace("*", "").replace("#", "")
    except Exception as e:
        return f"Erro na análise com IA: {e}"

# ── Renderização e Disparo de E-mail ──────────────────────────────────────────

def _cor_variacao(pct: float) -> str:
    return "#16A34A" if pct > 0 else ("#DC2626" if pct < 0 else "#64748B")

def _seta_variacao(pct: float) -> str:
    return "▲" if pct > 0 else ("▼" if pct < 0 else "━")

def _linha_ativo(ticker: str, dados: dict) -> str:
    moeda = "R$" if dados.get("moeda") == "BRL" else "US$"
    pct = dados.get("variacao_pct", 0.0)
    cor = _cor_variacao(pct)
    seta = _seta_variacao(pct)
    sinal = "+" if pct > 0 else ""
    nome = dados.get("nome", ticker)[:30]
    return f"""
      <tr style="border-bottom:1px solid #F1F5F9;">
        <td style="padding:10px 8px;font-weight:700;color:#0F172A;white-space:nowrap;">{ticker}</td>
        <td style="padding:10px 8px;color:#64748B;font-size:13px;">{nome}</td>
        <td style="padding:10px 8px;font-weight:700;white-space:nowrap;">{moeda} {dados.get('preco', 0):.2f}</td>
        <td style="padding:10px 8px;font-weight:600;color:{cor};white-space:nowrap;">
          {seta} {sinal}{pct:.2f}%
        </td>
      </tr>"""

def montar_html(
    analise_ia: str,
    cotacoes_br: dict,
    cotacoes_intl: dict,
    indicadores: list[dict],
    tesouro: list[dict],
    fund_dados: dict,
    ipca_atual: float,
) -> str:
    """Gera o template HTML responsivo contendo os dados e a análise textual."""
    header_ativos = """
      <tr style="background:#F8FAFC;">
        <th style="padding:8px;text-align:left;color:#64748B;font-size:12px;">Ticker</th>
        <th style="padding:8px;text-align:left;color:#64748B;font-size:12px;">Nome</th>
        <th style="padding:8px;text-align:left;color:#64748B;font-size:12px;">Preço</th>
        <th style="padding:8px;text-align:left;color:#64748B;font-size:12px;">Dia</th>
      </tr>"""
    linhas_br = header_ativos + "".join(_linha_ativo(t, d) for t, d in cotacoes_br.items())
    linhas_intl = header_ativos + "".join(_linha_ativo(t, d) for t, d in cotacoes_intl.items())

    rows_fund = ""
    for ticker, f in fund_dados.items():
        if not f or not (f.get("P/VP") or f.get("DY")):
            continue
        pl = f.get("P/L", "—")
        pvp = f.get("P/VP", "—")
        dy = f.get("DY", "—")
        rows_fund += f"""
          <tr style="border-bottom:1px solid #F1F5F9;">
            <td style="padding:8px;font-weight:700;">{ticker}</td>
            <td style="padding:8px;">{pvp}</td>
            <td style="padding:8px;color:#16A34A;font-weight:600;">{dy}</td>
            <td style="padding:8px;">{pl}</td>
          </tr>"""

    bloco_fund = f"""
      <h2 style="font-size:15px;font-weight:700;color:#1E3A5F;margin:20px 0 10px;">📐 Dados Fundamentalistas</h2>
      <table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:4px;">
        <tr style="background:#F8FAFC;">
          <th style="padding:7px;text-align:left;color:#64748B;font-size:12px;">Ticker</th>
          <th style="padding:7px;text-align:left;color:#64748B;font-size:12px;">P/VP</th>
          <th style="padding:7px;text-align:left;color:#64748B;font-size:12px;">DY</th>
          <th style="padding:7px;text-align:left;color:#64748B;font-size:12px;">P/L</th>
        </tr>
        {rows_fund}
      </table>""" if rows_fund else ""

    linhas_ind = "".join(
        f"""<tr style="border-bottom:1px solid #F1F5F9;">
          <td style="padding:10px 8px;font-weight:600;color:#0F172A;">{i.get('nome', '')}</td>
          <td style="padding:10px 8px;color:#16A34A;font-weight:700;text-align:right;">{i.get('valor', '')}</td>
        </tr>"""
        for i in indicadores
    )

    linhas_td = ""
    for t in tesouro:
        spread = t.get("taxa_spread", 0)
        preco = t.get("preco_compra", 0)
        nominal = t.get("nominal_estimado", 0)
        if t.get("eh_ipca"):
            linhas_td += f"""
              <tr style="border-bottom:0px;">
                <td colspan="2" style="padding:10px 8px 2px;font-weight:700;color:#0F172A;">{t.get('nome', '')}</td>
                <td style="padding:10px 8px 2px;font-weight:700;">R$ {preco:.2f}</td>
                <td style="padding:10px 8px 2px;color:#16A34A;font-weight:700;white-space:nowrap;">IPCA + {spread:.2f}% a.a.</td>
              </tr>
              <tr style="border-bottom:1px solid #F1F5F9;">
                <td colspan="4" style="padding:2px 8px 10px;font-size:12px;color:#64748B;">
                  IPCA 12m: {ipca_atual:.2f}% &nbsp;|&nbsp; Retorno nominal estimado: ~{nominal:.2f}% a.a.
                </td>
              </tr>"""
        else:
            linhas_td += f"""
              <tr style="border-bottom:1px solid #F1F5F9;">
                <td colspan="2" style="padding:10px 8px;font-weight:700;color:#0F172A;">{t.get('nome', '')}</td>
                <td style="padding:10px 8px;font-weight:700;">R$ {preco:.2f}</td>
                <td style="padding:10px 8px;color:#16A34A;font-weight:700;white-space:nowrap;">Selic + {spread:.4f}% a.a.</td>
              </tr>"""

    analise_html = analise_ia.replace("\n", "<br>")

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<body style="font-family:system-ui,-apple-system,sans-serif;background:#F8FAFC;margin:0;padding:0;">
<div style="max-width:680px;margin:28px auto;background:#fff;border-radius:14px;overflow:hidden;box-shadow:0 2px 16px rgba(0,0,0,.08);">
  <div style="background:#1E3A5F;padding:24px 28px;">
    <p style="margin:0;color:#0EA5E9;font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;">
      Carteira · {datetime.now().strftime('%d/%m/%Y %H:%M')}
    </p>
    <h1 style="margin:6px 0 0;color:#fff;font-size:22px;">📊 Briefing Diário</h1>
  </div>
  <div style="padding:24px 28px;">
    <h2 style="font-size:15px;font-weight:700;color:#1E3A5F;margin:0 0 10px;">🇧🇷 Ativos Brasileiros</h2>
    <table style="width:100%;border-collapse:collapse;font-size:14px;margin-bottom:4px;">{linhas_br}</table>
    {bloco_fund}
    <h2 style="font-size:15px;font-weight:700;color:#1E3A5F;margin:20px 0 10px;">💵 Internacionais</h2>
    <table style="width:100%;border-collapse:collapse;font-size:14px;margin-bottom:4px;">{linhas_intl}</table>
    <h2 style="font-size:15px;font-weight:700;color:#1E3A5F;margin:20px 0 10px;">🏛️ Renda Fixa e Cenário Macro</h2>
    <table style="width:100%;border-collapse:collapse;font-size:14px;margin-bottom:6px;">{linhas_ind}</table>
    <table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:20px;">{linhas_td}</table>
    <div style="background:#F0F9FF;border-radius:10px;padding:20px;border-left:4px solid #0EA5E9;">
      <h2 style="font-size:15px;font-weight:700;color:#1E3A5F;margin:0 0 12px;">🤖 Análise do Assessor</h2>
      <div style="font-size:14px;color:#1E293B;line-height:1.75;">{analise_html}</div>
    </div>
  </div>
</div>
</body>
</html>"""

def enviar_email(html: str, tem_alerta: bool = False):
    """Envia o e-mail transacional via protocolo SMTP."""
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        print("[email] Credenciais não configuradas — pulando envio.")
        return
    assunto = f"{'⚠️ ALERTA ' if tem_alerta else ''}📊 Briefing Carteira — {datetime.now().strftime('%d/%m/%Y')}"
    msg = MIMEMultipart("alternative")
    msg["Subject"] = assunto
    msg["From"] = SMTP_EMAIL
    msg["To"] = DEST_EMAIL
    msg.attach(MIMEText(html, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(SMTP_EMAIL, SMTP_PASSWORD)
            s.sendmail(SMTP_EMAIL, DEST_EMAIL, msg.as_string())
        print("[email] Relatório enviado com sucesso.")
    except Exception as e:
        print(f"[email] Falha no disparo: {e}")

# ── Execução Principal ────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"Agente de Investimentos — {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print("=" * 60)

    system_prompt = carregar_prompt()
    carteira = carregar_carteira()

    tickers_acoes = [a["ticker"] for a in carteira.get("acoes_br", [])]
    tickers_etf = [a["ticker"] for a in carteira.get("etfs_br", [])]
    tickers_fii = [a["ticker"] for a in carteira.get("fiis", [])]
    tickers_br = tickers_acoes + tickers_etf + tickers_fii
    tickers_intl = (
        [a["ticker"] for a in carteira.get("acoes_intl", [])] +
        [a["ticker"] for a in carteira.get("reits", [])] +
        [a["ticker"] for a in carteira.get("etfs_intl", [])]
    )

    print("\n[1/5] Cotações e Variação Diária (Yahoo Finance)...")
    cotacoes_br = buscar_cotacoes_yf(tickers_br, ".SA")
    cotacoes_intl = buscar_cotacoes_yf(tickers_intl, "")
    print(f"  BR: {len(cotacoes_br)} ativos | Intl: {len(cotacoes_intl)} ativos")

    print("\n[2/5] Indicadores Macroeconômicos (Banco Central)...")
    indicadores, ipca_atual = buscar_indicadores_bcb()
    print(f"  {len(indicadores)} indicador(es) coletado(s) | IPCA 12m: {ipca_atual:.2f}%")

    print("\n[3/5] Estrutura da Renda Fixa (Tesouro Direto)...")
    tesouro = buscar_tesouro_direto(ipca_atual)
    print(f"  {len(tesouro)} título(s) ativo(s)")

    print("\n[4/5] Análise Fundamentalista (Fundamentus)...")
    fund_dados = {}
    for ticker in tickers_acoes + tickers_fii:
        fund_dados[ticker] = buscar_fundamentus(ticker)
        campos = list(fund_dados[ticker].keys())
        print(f"  {ticker}: {campos if campos else 'sem dados'}")

    print("\n[5/5] Coleta de Notícias e Síntese por Inteligência Artificial...")
    contexto = montar_contexto(
        carteira, cotacoes_br, cotacoes_intl,
        indicadores, ipca_atual, tesouro, fund_dados,
    )
    analise = analisar_carteira(contexto, system_prompt)
    tem_alerta = "🔴" in analise or "ATENÇÃO" in analise.upper()

    print("\n" + "─" * 60)
    print(analise)
    print("─" * 60)

    html = montar_html(
        analise, cotacoes_br, cotacoes_intl,
        indicadores, tesouro, fund_dados, ipca_atual,
    )
    enviar_email(html, tem_alerta)
    print("\n✓ Pipeline finalizado com sucesso.")

if __name__ == "__main__":
    main()