
"""
BOT V3.3 ELITE PATTERN + NEWS AUTO

Features
- Top 30 USA scanner
- 20-pattern engine (shows only the best pattern)
- Finnhub news catalyst engine
- Telegram pretty alerts with explanation
- Journal open/closed trades
- HTML dashboard
- Auto loop with deduplicated alerts
- Simple backtest + Monte Carlo

Install:
    python -m pip install yfinance pandas numpy requests

Main commands:
    python .\bot_v3_3_elite_pattern_news.py --mode telegram_test
    python .\bot_v3_3_elite_pattern_news.py --menu
    python .\bot_v3_3_elite_pattern_news.py --mode auto --interval-minutes 15
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import html
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# =========================================================
# CONFIG
# =========================================================
TOP30_TICKERS: List[str] = [
    "MSFT", "AAPL", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "AVGO", "TSLA", "AMD",
    "NFLX", "COST", "ADBE", "CRM", "CSCO", "QCOM", "INTU", "AMAT", "MU", "PLTR",
    "JPM", "V", "MA", "BRK-B", "UNH", "LLY", "XOM", "WMT", "UBER", "CAT",
]
BENCHMARKS = ["SPY", "QQQ", "DIA"]

LOOKBACK_SCAN = "2y"
LOOKBACK_BACKTEST = "6y"

ACCOUNT_SIZE = 10_000.0
RISK_PER_TRADE = 0.005
MAX_OPEN_POSITIONS = 5
MIN_AVG_DOLLAR_VOLUME = 250_000_000
MIN_PRICE = 10.0
DEFAULT_RR = 2.0
SLIPPAGE_BPS = 5
COMMISSION_PER_TRADE = 2.95
TOP_N_SIGNALS = 10
MC_RUNS = 1000
EARNINGS_DAYS_BLOCK = 2
AUTO_INTERVAL_MINUTES_DEFAULT = 15
NEWS_LOOKBACK_DAYS = 3
NEWS_MAX_ITEMS = 8

REPORT_DIR = Path("reports_v3")
JOURNAL_DIR = Path("journal_v3")
DASHBOARD_DIR = Path("dashboard_v3")
STATE_DIR = Path("state_v3")
for d in [REPORT_DIR, JOURNAL_DIR, DASHBOARD_DIR, STATE_DIR]:
    d.mkdir(exist_ok=True)

OPEN_TRADES_CSV = JOURNAL_DIR / "open_trades.csv"
CLOSED_TRADES_CSV = JOURNAL_DIR / "closed_trades.csv"
DAILY_HISTORY_CSV = JOURNAL_DIR / "daily_scan_history.csv"
WATCHLIST_CSV = JOURNAL_DIR / "elite_watchlist.csv"
LAST_ALERTS_CSV = STATE_DIR / "last_alerts.csv"

# TELEGRAM
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

# NEWS
NEWS_PROVIDER = "FINNHUB"  # FINNHUB / NONE
FINNHUB_API_KEY = ""


# =========================================================
# DATA CLASSES
# =========================================================
@dataclass
class MarketRegime:
    direction: str
    score: int
    comment: str
    risk_on: bool


@dataclass
class PatternResult:
    name: str
    category: str
    side: str
    score: int
    quality: str
    reason: str


@dataclass
class NewsSummary:
    provider: str
    bias: str
    score: int
    headline: str
    bullets: str
    article_count: int


@dataclass
class SignalRow:
    timestamp_utc: str
    symbol: str
    regime: str
    regime_score: int
    signal: str
    score: int
    confidence: str
    watch_rank: int
    pattern_name: str
    pattern_category: str
    pattern_side: str
    pattern_score: int
    pattern_quality: str
    pattern_reason: str
    news_bias: str
    news_score: int
    news_headline: str
    news_bullets: str
    price: float
    entry: float
    stop_loss: float
    take_profit: float
    rr: float
    qty: int
    risk_amount: float
    ema20: float
    ema50: float
    ema200: float
    rsi14: float
    adx14: float
    atr14: float
    volume_ratio: float
    avg_dollar_volume: float
    ret5: float
    ret20: float
    earnings_block: bool
    notes: str
    explanation: str


@dataclass
class TradeRow:
    symbol: str
    side: str
    entry_date: str
    entry_price: float
    stop_loss: float
    take_profit: float
    qty: int
    risk_amount: float
    signal_score: int
    status: str = "OPEN"
    exit_date: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0


# =========================================================
# INDICATORS
# =========================================================
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def macd(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    e12 = ema(series, 12)
    e26 = ema(series, 26)
    line = e12 - e26
    sig = line.ewm(span=9, adjust=False).mean()
    hist = line - sig
    return line, sig, hist


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)

    atr_s = tr.rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr_s
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr_s
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.rolling(period).mean()


def slope(series: pd.Series, period: int = 10) -> pd.Series:
    out = pd.Series(index=series.index, dtype=float)
    if len(series) < period:
        return out
    for i in range(period, len(series) + 1):
        y = series.iloc[i - period:i].values
        x = np.arange(period)
        out.iloc[i - 1] = np.polyfit(x, y, 1)[0]
    return out


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    out["EMA20"] = ema(out["Close"], 20)
    out["EMA50"] = ema(out["Close"], 50)
    out["EMA200"] = ema(out["Close"], 200)
    out["RSI14"] = rsi(out["Close"], 14)
    out["ATR14"] = atr(out, 14)
    out["ADX14"] = adx(out, 14)
    _, _, hist = macd(out["Close"])
    out["MACD_HIST"] = hist
    out["VOL20"] = out["Volume"].rolling(20).mean()
    out["DOLLAR_VOL20"] = (out["Close"] * out["Volume"]).rolling(20).mean()
    out["RET5"] = out["Close"].pct_change(5)
    out["RET20"] = out["Close"].pct_change(20)
    out["EMA20_SLOPE"] = slope(out["EMA20"], 10)
    out["EMA50_SLOPE"] = slope(out["EMA50"], 10)
    out["HH20"] = out["High"].rolling(20).max()
    out["LL20"] = out["Low"].rolling(20).min()
    out["HH50"] = out["High"].rolling(50).max()
    out["LL50"] = out["Low"].rolling(50).min()
    out["RANGE_PCT"] = (out["High"] - out["Low"]) / out["Close"].replace(0, np.nan)
    out["ATR_PCT"] = out["ATR14"] / out["Close"].replace(0, np.nan)
    out["INSIDE_BAR"] = (
        (out["High"] <= out["High"].shift(1)) &
        (out["Low"] >= out["Low"].shift(1))
    )
    return out


# =========================================================
# UTILS
# =========================================================
def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def h(text: str) -> str:
    return html.escape(str(text))


def safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def to_single_symbol_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        try:
            return df.xs(symbol, axis=1, level=1).copy()
        except Exception:
            try:
                return df.xs(symbol, axis=1, level=-1).copy()
            except Exception:
                return pd.DataFrame()
    return df.copy()


def load_csv_or_empty(path: Path, columns: List[str]) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame(columns=columns)


# =========================================================
# TELEGRAM
# =========================================================
def send_telegram_html(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "INSERISCI_QUI_IL_TUO_TOKEN" or not TELEGRAM_CHAT_ID:
        print("Telegram non configurato correttamente.")
        return False

    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={
                "chat_id": str(TELEGRAM_CHAT_ID),
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            },
            timeout=10,
        )
        if r.status_code != 200:
            print(f"Errore Telegram: {r.status_code} - {r.text}")
            return False
        return True
    except Exception as e:
        print(f"Errore Telegram: {e}")
        return False


def telegram_test() -> None:
    msg = (
        f"✅ <b>Test Telegram OK</b>\n"
        f"🕒 <b>Ora:</b> {h(utc_now_str())}\n"
        f"🤖 <b>Bot:</b> V3.3 ELITE PATTERN + NEWS"
    )
    ok = send_telegram_html(msg)
    print("Test Telegram inviato con successo." if ok else "Test Telegram fallito.")


# =========================================================
# CSV HELPERS
# =========================================================
def get_trade_columns() -> List[str]:
    return list(asdict(
        TradeRow(
            symbol="",
            side="",
            entry_date="",
            entry_price=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            qty=0,
            risk_amount=0.0,
            signal_score=0
        )
    ).keys())


def get_last_alerts_columns() -> List[str]:
    return ["symbol", "signal", "score", "entry", "alert_key", "ts_utc"]


def load_last_alerts() -> pd.DataFrame:
    return load_csv_or_empty(LAST_ALERTS_CSV, get_last_alerts_columns())


def save_last_alerts(df: pd.DataFrame) -> None:
    df.to_csv(LAST_ALERTS_CSV, index=False)


# =========================================================
# DATA DOWNLOAD
# =========================================================
def download_prices(symbols: List[str], period: str, interval: str = "1d") -> pd.DataFrame:
    try:
        df = yf.download(
            symbols,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=True
        )
        if df is None:
            return pd.DataFrame()
        return df
    except Exception as e:
        print(f"Errore download Yahoo: {e}")
        return pd.DataFrame()


def get_earnings_block(symbol: str) -> bool:
    try:
        cal = yf.Ticker(symbol).calendar
        if cal is None or len(cal) == 0:
            return False

        dates = []
        if isinstance(cal, pd.DataFrame):
            for v in cal.values.flatten():
                if pd.notna(v):
                    try:
                        dates.append(pd.Timestamp(v).to_pydatetime())
                    except Exception:
                        pass

        today = datetime.now(timezone.utc).date()
        for dt in dates:
            if abs((dt.date() - today).days) <= EARNINGS_DAYS_BLOCK:
                return True
        return False
    except Exception:
        return False


# =========================================================
# NEWS ENGINE
# =========================================================
POSITIVE_NEWS_WORDS = [
    "beat", "beats", "surge", "rally", "upgrade", "raises", "strong", "growth",
    "partnership", "buyback", "record", "expands", "bullish", "outperform", "wins"
]
NEGATIVE_NEWS_WORDS = [
    "miss", "misses", "downgrade", "lawsuit", "probe", "fall", "drops", "weak",
    "cuts", "cut", "warning", "bearish", "delay", "decline", "investigation"
]


def score_news_text(text: str) -> int:
    t = text.lower()
    score = 0
    for w in POSITIVE_NEWS_WORDS:
        if w in t:
            score += 1
    for w in NEGATIVE_NEWS_WORDS:
        if w in t:
            score -= 1
    return score


def fetch_finnhub_news(symbol: str) -> List[dict]:
    if NEWS_PROVIDER != "FINNHUB":
        return []
    if not FINNHUB_API_KEY or FINNHUB_API_KEY == "INSERISCI_QUI_LA_TUA_FINNHUB_API_KEY":
        return []

    date_to = datetime.now(timezone.utc).date()
    date_from = date_to - timedelta(days=NEWS_LOOKBACK_DAYS)

    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": symbol,
        "from": str(date_from),
        "to": str(date_to),
        "token": FINNHUB_API_KEY,
    }

    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            print(f"News error {symbol}: {r.status_code} - {r.text[:200]}")
            return []
        data = r.json()
        if isinstance(data, list):
            return data[:NEWS_MAX_ITEMS]
        return []
    except Exception as e:
        print(f"News exception {symbol}: {e}")
        return []


def summarize_news(symbol: str) -> NewsSummary:
    items = fetch_finnhub_news(symbol)
    if not items:
        return NewsSummary(
            provider=NEWS_PROVIDER,
            bias="NEUTRAL",
            score=0,
            headline="Nessuna news recente rilevante o API non configurata",
            bullets="",
            article_count=0,
        )

    scored = []
    for item in items:
        headline = str(item.get("headline", "") or "")
        summary = str(item.get("summary", "") or "")
        source = str(item.get("source", "") or "")
        text = f"{headline}. {summary}"
        s = score_news_text(text)
        scored.append((s, headline, summary, source))

    total_score = sum(s for s, _, _, _ in scored)
    bias = "NEUTRAL"
    if total_score >= 2:
        bias = "POSITIVE"
    elif total_score <= -2:
        bias = "NEGATIVE"

    headline = scored[0][1] if scored else "Nessuna headline"
    bullets = []
    for s, hline, _, src in scored[:3]:
        tag = "🟢" if s > 0 else "🔴" if s < 0 else "🟡"
        bullets.append(f"{tag} {hline} ({src})")

    return NewsSummary(
        provider=NEWS_PROVIDER,
        bias=bias,
        score=int(total_score),
        headline=headline,
        bullets=" | ".join(bullets),
        article_count=len(items),
    )


# =========================================================
# MARKET REGIME
# =========================================================
def compute_regime(bench_data: pd.DataFrame) -> MarketRegime:
    score = 0
    comments: List[str] = []

    for b in BENCHMARKS:
        sdf = to_single_symbol_df(bench_data, b).dropna()
        df = compute_indicators(sdf)
        if len(df) < 220:
            comments.append(f"{b}:no_data")
            continue

        row = df.iloc[-1]
        local = 0
        if row["Close"] > row["EMA20"]:
            local += 1
        if row["EMA20"] > row["EMA50"]:
            local += 1
        if row["EMA50"] > row["EMA200"]:
            local += 1
        if row["ADX14"] > 18:
            local += 1
        if row["MACD_HIST"] > 0:
            local += 1

        score += local
        comments.append(f"{b}:{local}/5")

    if score >= 11:
        return MarketRegime("BULLISH", score, " | ".join(comments), True)
    if score <= 5:
        return MarketRegime("BEARISH", score, " | ".join(comments), False)
    return MarketRegime("NEUTRAL", score, " | ".join(comments), False)


# =========================================================
# PATTERN ENGINE (20 patterns, best only)
# =========================================================
def quality_from_score(score: int) -> str:
    if score >= 9:
        return "HIGH"
    if score >= 6:
        return "MEDIUM"
    return "LOW"


def detect_best_pattern(df: pd.DataFrame) -> PatternResult:
    if len(df) < 60:
        return PatternResult("No pattern", "NONE", "FLAT", 0, "LOW", "Storico insufficiente")

    row = df.iloc[-1]
    prev = df.iloc[-2]
    recent10 = df.tail(10)
    recent20 = df.tail(20)
    recent50 = df.tail(50)

    patterns: List[PatternResult] = []

    close = float(row["Close"])
    high = float(row["High"])
    low = float(row["Low"])
    vol = float(row["Volume"])
    vol20 = safe_float(row["VOL20"], 0.0)

    # 1 Breakout 20
    if close >= float(recent20["High"].iloc[:-1].max()):
        patterns.append(PatternResult("Breakout 20 barre", "BREAKOUT", "LONG", 9, "HIGH", "Prezzo sopra i massimi recenti a 20 barre"))

    # 2 Breakout 50
    if close >= float(recent50["High"].iloc[:-1].max()):
        patterns.append(PatternResult("Breakout 50 barre", "BREAKOUT", "LONG", 10, "HIGH", "Rottura dei massimi a 50 barre"))

    # 3 Breakout volume spike
    if close >= float(recent20["High"].iloc[:-1].max()) and vol20 > 0 and vol > vol20 * 1.5:
        patterns.append(PatternResult("Breakout con volume spike", "BREAKOUT", "LONG", 10, "HIGH", "Breakout confermato da forte espansione dei volumi"))

    # 4 Breakout from tight consolidation
    if close >= float(recent20["High"].iloc[:-1].max()) and recent10["RANGE_PCT"].mean() < recent20["RANGE_PCT"].mean() * 0.75:
        patterns.append(PatternResult("Breakout da consolidamento stretto", "BREAKOUT", "LONG", 10, "HIGH", "Compressione dei range prima della rottura"))

    # 5 Bull flag
    if float(row["EMA20"]) > float(row["EMA50"]) and recent10["Close"].iloc[-1] > recent10["Close"].iloc[0]:
        mid = recent10["Close"].iloc[3:8]
        if len(mid) >= 3 and mid.is_monotonic_decreasing and close > mid.max():
            patterns.append(PatternResult("Bull flag", "CONTINUATION", "LONG", 8, "HIGH", "Pausa ordinata nel trend seguita da ripartenza"))

    # 6 Bear flag
    if float(row["EMA20"]) < float(row["EMA50"]) and recent10["Close"].iloc[-1] < recent10["Close"].iloc[0]:
        mid = recent10["Close"].iloc[3:8]
        if len(mid) >= 3 and mid.is_monotonic_increasing and close < mid.min():
            patterns.append(PatternResult("Bear flag", "CONTINUATION", "SHORT", 8, "HIGH", "Rimbalzo tecnico debole dentro trend ribassista"))

    # 7 Range expansion
    if recent10["RANGE_PCT"].iloc[-1] > recent10["RANGE_PCT"].iloc[:-1].mean() * 1.8:
        side = "LONG" if close > float(prev["Close"]) else "SHORT"
        patterns.append(PatternResult("Range expansion", "VOLATILITY", side, 7, "MEDIUM", "Espansione improvvisa del range dopo fase più calma"))

    # 8 Pullback EMA20
    if float(row["EMA20"]) > float(row["EMA50"]) > float(row["EMA200"]) and low <= float(row["EMA20"]) * 1.003 and close > float(row["EMA20"]):
        patterns.append(PatternResult("Pullback su EMA20", "PULLBACK", "LONG", 8, "HIGH", "Ritracciamento controllato su EMA20 con tenuta del trend"))

    # 9 Pullback EMA50
    if float(row["EMA50"]) > float(row["EMA200"]) and low <= float(row["EMA50"]) * 1.003 and close > float(row["EMA50"]):
        patterns.append(PatternResult("Pullback su EMA50", "PULLBACK", "LONG", 7, "MEDIUM", "Ritorno su supporto dinamico intermedio"))

    # 10 Trend alignment long
    if float(row["EMA20"]) > float(row["EMA50"]) > float(row["EMA200"]) and close > float(row["EMA20"]):
        patterns.append(PatternResult("Trend alignment long", "TREND", "LONG", 7, "MEDIUM", "Trend rialzista pulito su tre medie"))

    # 11 Higher highs / lows
    if recent10["High"].iloc[-1] > recent10["High"].iloc[-5] and recent10["Low"].iloc[-1] > recent10["Low"].iloc[-5]:
        patterns.append(PatternResult("Higher highs / higher lows", "TREND", "LONG", 7, "MEDIUM", "Struttura rialzista con massimi e minimi crescenti"))

    # 12 Lower highs / lows
    if recent10["High"].iloc[-1] < recent10["High"].iloc[-5] and recent10["Low"].iloc[-1] < recent10["Low"].iloc[-5]:
        patterns.append(PatternResult("Lower highs / lower lows", "TREND", "SHORT", 7, "MEDIUM", "Struttura ribassista con massimi e minimi decrescenti"))

    # 13 Double bottom
    lows = recent20["Low"].nsmallest(2).sort_values()
    if len(lows) == 2 and abs(lows.iloc[1] - lows.iloc[0]) / max(lows.iloc[0], 1e-9) < 0.02 and close > recent20["Close"].median():
        patterns.append(PatternResult("Double bottom", "REVERSAL", "LONG", 8, "MEDIUM", "Due minimi simili con recupero della neckline locale"))

    # 14 Double top
    highs = recent20["High"].nlargest(2).sort_values()
    if len(highs) == 2 and abs(highs.iloc[1] - highs.iloc[0]) / max(highs.iloc[0], 1e-9) < 0.02 and close < recent20["Close"].median():
        patterns.append(PatternResult("Double top", "REVERSAL", "SHORT", 8, "MEDIUM", "Doppio massimo con perdita della forza"))

    # 15 False breakout up
    if high > float(recent20["High"].iloc[:-1].max()) and close < float(recent20["High"].iloc[:-1].max()):
        patterns.append(PatternResult("False breakout rialzista", "REVERSAL", "SHORT", 7, "MEDIUM", "Rottura dei massimi non confermata in chiusura"))

    # 16 False breakdown down
    if low < float(recent20["Low"].iloc[:-1].min()) and close > float(recent20["Low"].iloc[:-1].min()):
        patterns.append(PatternResult("False breakdown ribassista", "REVERSAL", "LONG", 7, "MEDIUM", "Violazione dei minimi con rapido recupero"))

    # 17 Support bounce
    if low <= float(recent20["Low"].iloc[:-1].min()) * 1.01 and close > float(prev["Close"]):
        patterns.append(PatternResult("Support bounce", "REVERSAL", "LONG", 6, "MEDIUM", "Reazione positiva su area di supporto 20 barre"))

    # 18 Resistance rejection
    if high >= float(recent20["High"].iloc[:-1].max()) * 0.99 and close < float(prev["Close"]):
        patterns.append(PatternResult("Resistance rejection", "REVERSAL", "SHORT", 6, "MEDIUM", "Rifiuto della resistenza dei massimi recenti"))

    # 19 ATR contraction
    if recent10["ATR_PCT"].mean() < recent20["ATR_PCT"].mean() * 0.8:
        side = "LONG" if close > float(row["EMA20"]) else "SHORT"
        patterns.append(PatternResult("ATR contraction", "VOLATILITY", side, 6, "MEDIUM", "Compressione di volatilità potenzialmente pronta a espansione"))

    # 20 Inside bar cluster
    inside_count = int(recent10["INSIDE_BAR"].sum())
    if inside_count >= 3 or recent10["RANGE_PCT"].mean() < recent20["RANGE_PCT"].mean() * 0.7:
        side = "LONG" if close > float(row["EMA20"]) else "SHORT"
        patterns.append(PatternResult("Inside bar cluster", "VOLATILITY", side, 6, "MEDIUM", "Cluster di barre compresse / range stretto"))

    if not patterns:
        return PatternResult("No dominant pattern", "NONE", "FLAT", 0, "LOW", "Nessun pattern strutturale dominante")

    best = max(patterns, key=lambda p: p.score)
    return PatternResult(best.name, best.category, best.side, best.score, quality_from_score(best.score), best.reason)


# =========================================================
# SIGNAL / EXPLANATION
# =========================================================
def compute_watch_rank(df: pd.DataFrame, regime: MarketRegime) -> int:
    row = df.iloc[-1]
    score = 0
    if row["DOLLAR_VOL20"] >= MIN_AVG_DOLLAR_VOLUME:
        score += 2
    if row["Close"] > row["EMA20"] > row["EMA50"]:
        score += 2
    if row["EMA50"] > row["EMA200"]:
        score += 1
    if row["ADX14"] > 20:
        score += 1
    if row["MACD_HIST"] > 0:
        score += 1
    if row["RET20"] > 0:
        score += 1
    if row["Volume"] > row["VOL20"]:
        score += 1
    if regime.direction == "BULLISH" and row["Close"] > row["EMA200"]:
        score += 1
    return int(score)


def build_explanation(symbol: str, side: str, pattern: PatternResult, regime: MarketRegime, news: NewsSummary, row: pd.Series) -> str:
    parts = []
    if float(row["EMA20"]) > float(row["EMA50"]) > float(row["EMA200"]):
        parts.append("trend rialzista sopra EMA20/50/200")
    elif float(row["EMA20"]) < float(row["EMA50"]) < float(row["EMA200"]):
        parts.append("trend ribassista sotto EMA20/50/200")

    if float(row["ADX14"]) > 20:
        parts.append("trend strength valida via ADX")
    if float(row["Volume"]) > float(row["VOL20"]):
        parts.append("volumi sopra media")
    if float(row["MACD_HIST"]) > 0:
        parts.append("momentum MACD positivo")
    elif float(row["MACD_HIST"]) < 0:
        parts.append("momentum MACD negativo")

    trend = "; ".join(parts) if parts else "contesto tecnico misto"

    news_part = "nessun catalyst news forte"
    if news.article_count > 0:
        if news.bias == "POSITIVE":
            news_part = f"news recenti favorevoli ({news.headline})"
        elif news.bias == "NEGATIVE":
            news_part = f"news recenti deboli/negative ({news.headline})"
        else:
            news_part = f"news recenti neutre/miste ({news.headline})"

    return (
        f"{side} su {symbol}: {pattern.name.lower()}. {pattern.reason}. "
        f"Motivo operativo: {trend}. "
        f"Contesto mercato: regime {regime.direction.lower()} ({regime.score}). "
        f"Catalyst: {news_part}."
    )


def build_signal(symbol: str, df: pd.DataFrame, regime: MarketRegime, watch_rank: int) -> Optional[SignalRow]:
    if len(df) < 220:
        return None

    row = df.iloc[-1]
    price = safe_float(row["Close"])
    if not np.isfinite(price) or price < MIN_PRICE:
        return None

    avg_dollar_vol = safe_float(row["DOLLAR_VOL20"])
    if not np.isfinite(avg_dollar_vol) or avg_dollar_vol < MIN_AVG_DOLLAR_VOLUME:
        return None

    volume_ratio = safe_float(row["Volume"] / row["VOL20"], 0.0)
    atr14 = safe_float(row["ATR14"], np.nan)
    adx14 = safe_float(row["ADX14"], np.nan)
    earnings_block = get_earnings_block(symbol)
    news = summarize_news(symbol)
    pattern = detect_best_pattern(df)

    long_score = 0
    short_score = 0
    notes: List[str] = []

    if price > row["EMA20"]:
        long_score += 1
    else:
        short_score += 1

    if row["EMA20"] > row["EMA50"]:
        long_score += 1
    else:
        short_score += 1

    if row["EMA50"] > row["EMA200"]:
        long_score += 1
    else:
        short_score += 1

    if row["MACD_HIST"] > 0:
        long_score += 1
    else:
        short_score += 1

    if row["RET5"] > 0:
        long_score += 1
    else:
        short_score += 1

    if row["RET20"] > 0:
        long_score += 1
    else:
        short_score += 1

    if row["RSI14"] > 55:
        long_score += 1
    if row["RSI14"] < 45:
        short_score += 1

    if adx14 > 18:
        if long_score >= short_score:
            long_score += 1
        else:
            short_score += 1

    if volume_ratio > 1.2:
        if long_score >= short_score:
            long_score += 1
        else:
            short_score += 1

    if pattern.side == "LONG":
        long_score += max(pattern.score // 2, 0)
    elif pattern.side == "SHORT":
        short_score += max(pattern.score // 2, 0)

    if news.bias == "POSITIVE":
        long_score += 1
        notes.append("Positive news flow")
    elif news.bias == "NEGATIVE":
        short_score += 1
        notes.append("Negative news flow")
    else:
        notes.append("Neutral/mixed news flow")

    signal = "NO TRADE"
    score = max(long_score, short_score)

    if earnings_block:
        notes.append("Earnings block")
        if score >= 8:
            signal = "WATCH"
    else:
        if long_score >= 10 and regime.direction != "BEARISH":
            signal = "LONG"
        elif short_score >= 10 and regime.direction != "BULLISH":
            signal = "SHORT"
        elif score >= 8:
            signal = "WATCH"

    if not np.isfinite(atr14) or atr14 <= 0:
        return None

    rr = DEFAULT_RR

    if signal in ["LONG", "WATCH"] and long_score >= short_score:
        entry = price
        stop = max(price - (1.6 * atr14), 0.01)
        tp = price + (price - stop) * rr
        side = "LONG"
    elif signal in ["SHORT", "WATCH"] and short_score > long_score:
        entry = price
        stop = price + (1.6 * atr14)
        tp = max(price - (stop - price) * rr, 0.01)
        side = "SHORT"
    else:
        entry, stop, tp, side = price, price, price, "FLAT"

    risk_per_share = abs(entry - stop)
    capital_risk = ACCOUNT_SIZE * RISK_PER_TRADE
    qty = int(capital_risk / risk_per_share) if risk_per_share > 0 else 0
    qty = max(qty, 0)

    confidence = "LOW"
    if score >= 12:
        confidence = "HIGH"
    elif score >= 9:
        confidence = "MEDIUM"

    if regime.direction == "BULLISH" and side == "LONG":
        notes.append("Aligned with regime")
    elif regime.direction == "BEARISH" and side == "SHORT":
        notes.append("Aligned with regime")
    elif side != "FLAT":
        notes.append("Counter/neutral regime")

    if row["Close"] >= row["HH20"] * 0.995:
        notes.append("Near breakout")
    if row["Close"] <= row["LL20"] * 1.005:
        notes.append("Near breakdown")
    if volume_ratio > 1.3:
        notes.append("Volume expansion")
    if pattern.name != "No dominant pattern":
        notes.append(f"Best pattern: {pattern.name}")
    if news.article_count > 0:
        notes.append(f"News bias: {news.bias}")

    explanation = build_explanation(symbol, side, pattern, regime, news, row)

    return SignalRow(
        timestamp_utc=utc_now_str(),
        symbol=symbol,
        regime=regime.direction,
        regime_score=regime.score,
        signal=signal,
        score=int(score),
        confidence=confidence,
        watch_rank=int(watch_rank),
        pattern_name=pattern.name,
        pattern_category=pattern.category,
        pattern_side=pattern.side,
        pattern_score=pattern.score,
        pattern_quality=pattern.quality,
        pattern_reason=pattern.reason,
        news_bias=news.bias,
        news_score=news.score,
        news_headline=news.headline,
        news_bullets=news.bullets,
        price=round(price, 4),
        entry=round(entry, 4),
        stop_loss=round(stop, 4),
        take_profit=round(tp, 4),
        rr=round(rr, 2),
        qty=qty,
        risk_amount=round(capital_risk, 2),
        ema20=round(safe_float(row["EMA20"]), 4),
        ema50=round(safe_float(row["EMA50"]), 4),
        ema200=round(safe_float(row["EMA200"]), 4),
        rsi14=round(safe_float(row["RSI14"]), 2),
        adx14=round(adx14, 2),
        atr14=round(atr14, 4),
        volume_ratio=round(volume_ratio, 2),
        avg_dollar_volume=round(avg_dollar_vol, 2),
        ret5=round(safe_float(row["RET5"]) * 100, 2),
        ret20=round(safe_float(row["RET20"]) * 100, 2),
        earnings_block=earnings_block,
        notes="; ".join(notes),
        explanation=explanation,
    )


def enrich_and_rank(signals: List[SignalRow]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(s) for s in signals])
    if df.empty:
        return df

    signal_boost = df["signal"].map({"LONG": 3, "SHORT": 3, "WATCH": 1, "NO TRADE": 0}).fillna(0)
    conf_boost = df["confidence"].map({"HIGH": 2, "MEDIUM": 1, "LOW": 0}).fillna(0)
    pattern_boost = (df["pattern_score"] / 2).fillna(0)
    news_boost = df["news_score"].clip(-2, 2).fillna(0)

    df["elite_rank"] = df["watch_rank"] + df["score"] + signal_boost + conf_boost + pattern_boost + news_boost
    df = df.sort_values(
        ["elite_rank", "score", "pattern_score", "avg_dollar_volume"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)
    return df


# =========================================================
# JOURNAL
# =========================================================
def append_open_trades(scan_df: pd.DataFrame) -> pd.DataFrame:
    cols = get_trade_columns()
    open_df = load_csv_or_empty(OPEN_TRADES_CSV, cols)

    current_open = open_df[open_df["status"] == "OPEN"].copy() if not open_df.empty else pd.DataFrame(columns=cols)
    existing_symbols = set(current_open["symbol"].astype(str).tolist())
    slots_left = max(MAX_OPEN_POSITIONS - len(current_open), 0)

    if slots_left <= 0:
        open_df.to_csv(OPEN_TRADES_CSV, index=False)
        return open_df

    to_open = scan_df[scan_df["signal"].isin(["LONG", "SHORT"])].head(TOP_N_SIGNALS)

    new_rows = []
    for _, r in to_open.iterrows():
        if len(new_rows) >= slots_left:
            break
        if r["symbol"] in existing_symbols:
            continue

        new_rows.append(asdict(TradeRow(
            symbol=str(r["symbol"]),
            side=str(r["signal"]),
            entry_date=str(datetime.now(timezone.utc).date()),
            entry_price=float(r["entry"]),
            stop_loss=float(r["stop_loss"]),
            take_profit=float(r["take_profit"]),
            qty=int(r["qty"]),
            risk_amount=float(r["risk_amount"]),
            signal_score=int(r["score"]),
        )))

    if new_rows:
        open_df = pd.concat([open_df, pd.DataFrame(new_rows)], ignore_index=True)

    open_df.to_csv(OPEN_TRADES_CSV, index=False)
    return open_df


def update_open_trades(market_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    open_cols = get_trade_columns()
    open_df = load_csv_or_empty(OPEN_TRADES_CSV, open_cols)
    closed_df = load_csv_or_empty(CLOSED_TRADES_CSV, open_cols)

    if open_df.empty:
        return open_df, closed_df

    updated_open_rows = []
    new_closed_rows = []

    for _, tr in open_df.iterrows():
        if str(tr.get("status", "OPEN")) != "OPEN":
            updated_open_rows.append(tr.to_dict())
            continue

        symbol = str(tr["symbol"])
        sdf = to_single_symbol_df(market_data, symbol).dropna()

        try:
            df = compute_indicators(sdf)
            if df.empty:
                updated_open_rows.append(tr.to_dict())
                continue
            row = df.iloc[-1]
            price = float(row["Close"])
        except Exception:
            updated_open_rows.append(tr.to_dict())
            continue

        side = str(tr["side"])
        entry = float(tr["entry_price"])
        stop = float(tr["stop_loss"])
        tp = float(tr["take_profit"])
        qty = int(float(tr["qty"]))

        status = "OPEN"
        exit_reason = ""

        if side == "LONG":
            if price <= stop:
                status, exit_reason = "CLOSED", "STOP"
            elif price >= tp:
                status, exit_reason = "CLOSED", "TARGET"
        elif side == "SHORT":
            if price >= stop:
                status, exit_reason = "CLOSED", "STOP"
            elif price <= tp:
                status, exit_reason = "CLOSED", "TARGET"

        if status == "CLOSED":
            gross = (price - entry) * qty if side == "LONG" else (entry - price) * qty
            costs = COMMISSION_PER_TRADE * 2 + (entry * qty + price * qty) * (SLIPPAGE_BPS / 10000)
            pnl = gross - costs
            pnl_pct = (pnl / max(entry * qty, 1e-9)) * 100

            row_out = tr.to_dict()
            row_out.update({
                "status": "CLOSED",
                "exit_date": str(datetime.now(timezone.utc).date()),
                "exit_price": round(price, 4),
                "exit_reason": exit_reason,
                "pnl_usd": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
            })
            new_closed_rows.append(row_out)
        else:
            updated_open_rows.append(tr.to_dict())

    open_new = pd.DataFrame(updated_open_rows, columns=open_cols)
    closed_new = pd.concat([closed_df, pd.DataFrame(new_closed_rows)], ignore_index=True) if new_closed_rows else closed_df

    open_new.to_csv(OPEN_TRADES_CSV, index=False)
    closed_new.to_csv(CLOSED_TRADES_CSV, index=False)

    return open_new, closed_new


# =========================================================
# TELEGRAM ALERTS
# =========================================================
def build_pretty_signal_message(row: pd.Series, regime: MarketRegime) -> str:
    signal = str(row["signal"])
    icon = "🟢" if signal == "LONG" else "🔴" if signal == "SHORT" else "🟡"

    return (
        f"{icon} <b>NUOVO SEGNALE {h(signal)}</b>\n"
        f"📌 <b>Ticker:</b> {h(row['symbol'])}\n"
        f"🌍 <b>Regime:</b> {h(regime.direction)} ({h(regime.score)})\n"
        f"⭐ <b>Score:</b> {h(row['score'])} | <b>Confidence:</b> {h(row['confidence'])}\n"
        f"🏅 <b>Elite Rank:</b> {h(round(float(row['elite_rank']), 2))}\n"
        f"🧩 <b>Pattern:</b> {h(row['pattern_name'])} | <b>Quality:</b> {h(row['pattern_quality'])}\n"
        f"📰 <b>News:</b> {h(row['news_bias'])} ({h(row['news_score'])})\n\n"
        f"💵 <b>Entry:</b> {h(row['entry'])}\n"
        f"🛑 <b>Stop:</b> {h(row['stop_loss'])}\n"
        f"🎯 <b>Target:</b> {h(row['take_profit'])}\n"
        f"⚖️ <b>R/R:</b> {h(row['rr'])}\n"
        f"📦 <b>Qty:</b> {h(row['qty'])}\n"
        f"💰 <b>Risk $:</b> {h(row['risk_amount'])}\n\n"
        f"🧠 <b>Motivo:</b> {h(row['explanation'])}\n"
        f"📰 <b>Headline:</b> {h(row['news_headline'])}\n"
        f"🕒 <b>Ora:</b> {h(utc_now_str())}"
    )


def build_telegram_summary(scan_df: pd.DataFrame, regime: MarketRegime) -> str:
    top = scan_df.head(5)

    lines = [
        "📡 <b>SCAN COMPLETATO</b>",
        f"🌍 <b>Regime:</b> {h(regime.direction)} ({h(regime.score)})",
        f"🕒 <b>Ora:</b> {h(utc_now_str())}",
        ""
    ]

    if top.empty:
        lines.append("Nessun setup trovato.")
        return "\n".join(lines)

    for i, (_, r) in enumerate(top.iterrows(), start=1):
        emoji = "🟢" if r["signal"] == "LONG" else "🔴" if r["signal"] == "SHORT" else "🟡"
        lines.append(
            f"{i}. {emoji} <b>{h(r['symbol'])}</b> | {h(r['signal'])} | "
            f"score {h(r['score'])} | pattern {h(r['pattern_name'])} | entry {h(r['entry'])}"
        )
    return "\n".join(lines)


def build_alert_key(row: pd.Series) -> str:
    return f"{row['symbol']}|{row['signal']}|{row['score']}|{row['entry']}|{row['pattern_name']}"


def send_new_signal_alerts(scan_df: pd.DataFrame, regime: MarketRegime) -> int:
    if scan_df.empty:
        return 0

    tradable = scan_df[scan_df["signal"].isin(["LONG", "SHORT"])].copy()
    if tradable.empty:
        return 0

    last_alerts = load_last_alerts()
    existing_keys = set(last_alerts["alert_key"].astype(str).tolist()) if not last_alerts.empty else set()

    sent_rows = []
    sent_count = 0

    for _, row in tradable.head(TOP_N_SIGNALS).iterrows():
        alert_key = build_alert_key(row)
        if alert_key in existing_keys:
            continue

        msg = build_pretty_signal_message(row, regime)
        ok = send_telegram_html(msg)
        if ok:
            sent_rows.append({
                "symbol": row["symbol"],
                "signal": row["signal"],
                "score": row["score"],
                "entry": row["entry"],
                "alert_key": alert_key,
                "ts_utc": utc_now_str(),
            })
            sent_count += 1

    if sent_rows:
        new_df = pd.concat([last_alerts, pd.DataFrame(sent_rows)], ignore_index=True)
        new_df = new_df.drop_duplicates(subset=["alert_key"], keep="last")
        save_last_alerts(new_df)

    return sent_count


# =========================================================
# SCAN
# =========================================================
def run_scan(send_alerts: bool = False) -> Tuple[pd.DataFrame, MarketRegime]:
    raw = download_prices(BENCHMARKS + TOP30_TICKERS, LOOKBACK_SCAN)

    if raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        print("Dati Yahoo non disponibili o formato inatteso.")
        return pd.DataFrame(), MarketRegime("UNKNOWN", 0, "No data", False)

    bench_mask = raw.columns.get_level_values(1).isin(BENCHMARKS)
    market_mask = raw.columns.get_level_values(1).isin(TOP30_TICKERS)

    bench_data = raw.loc[:, bench_mask]
    market_data = raw.loc[:, market_mask]

    regime = compute_regime(bench_data)

    signals: List[SignalRow] = []
    pre_rank = []

    for symbol in TOP30_TICKERS:
        sdf = to_single_symbol_df(market_data, symbol).dropna()
        df = compute_indicators(sdf)
        if len(df) < 220:
            continue
        watch_rank = compute_watch_rank(df, regime)
        pre_rank.append((symbol, watch_rank, float(df.iloc[-1]["DOLLAR_VOL20"])))

    pre_rank_sorted = [x[0] for x in sorted(pre_rank, key=lambda t: (t[1], t[2]), reverse=True)]
    top10 = set(pre_rank_sorted[:10])
    top5 = set(pre_rank_sorted[:5])

    for symbol in TOP30_TICKERS:
        sdf = to_single_symbol_df(market_data, symbol).dropna()
        df = compute_indicators(sdf)
        if len(df) < 220:
            continue

        watch_rank = compute_watch_rank(df, regime)
        sig = build_signal(symbol, df, regime, watch_rank)
        if sig:
            if symbol in top5:
                sig.watch_rank += 2
            elif symbol in top10:
                sig.watch_rank += 1
            signals.append(sig)

    scan_df = enrich_and_rank(signals)
    if scan_df.empty:
        return scan_df, regime

    ts = now_stamp()
    scan_df.to_csv(REPORT_DIR / f"scan_{ts}.csv", index=False)

    hist_df = load_csv_or_empty(DAILY_HISTORY_CSV, scan_df.columns.tolist())
    hist_df = pd.concat([hist_df, scan_df], ignore_index=True)
    hist_df.to_csv(DAILY_HISTORY_CSV, index=False)

    watchlist_df = scan_df[[
        "symbol", "elite_rank", "signal", "score", "confidence", "watch_rank",
        "pattern_name", "pattern_quality", "news_bias", "avg_dollar_volume"
    ]].copy().head(10)
    watchlist_df["tier"] = ["TOP5" if i < 5 else "TOP10" for i in range(len(watchlist_df))]
    watchlist_df.to_csv(WATCHLIST_CSV, index=False)

    update_open_trades(market_data)
    append_open_trades(scan_df)

    if send_alerts:
        sent_count = send_new_signal_alerts(scan_df, regime)
        print(f"Alert Telegram inviati: {sent_count}")

    return scan_df, regime


# =========================================================
# BACKTEST
# =========================================================
def backtest_symbol(symbol: str, raw_df: pd.DataFrame, regime_df_map: Dict[str, pd.DataFrame]) -> List[dict]:
    df = compute_indicators(raw_df.dropna())
    trades = []

    if len(df) < 220:
        return trades

    pos = None

    for i in range(220, len(df) - 1):
        row = df.iloc[i]
        next_bar = df.iloc[i + 1]
        next_open = float(next_bar["Open"])
        dt = str(df.index[i + 1].date())

        regime_score = 0
        bull = 0
        bear = 0

        for _, bdf in regime_df_map.items():
            if df.index[i] not in bdf.index:
                continue
            r = bdf.loc[df.index[i]]
            local = 0
            if r["Close"] > r["EMA20"]:
                local += 1
            if r["EMA20"] > r["EMA50"]:
                local += 1
            if r["EMA50"] > r["EMA200"]:
                local += 1
            if r["MACD_HIST"] > 0:
                local += 1
            if r["ADX14"] > 18:
                local += 1
            regime_score += local

        if regime_score >= 11:
            bull = 1
        elif regime_score <= 5:
            bear = 1

        long_setup = (
            row["Close"] > row["EMA20"] > row["EMA50"] > row["EMA200"] and
            row["MACD_HIST"] > 0 and row["RSI14"] > 55 and row["ADX14"] > 18 and bull == 1
        )

        short_setup = (
            row["Close"] < row["EMA20"] < row["EMA50"] < row["EMA200"] and
            row["MACD_HIST"] < 0 and row["RSI14"] < 45 and row["ADX14"] > 18 and bear == 1
        )

        if pos is None:
            if long_setup and np.isfinite(row["ATR14"]) and row["ATR14"] > 0:
                stop = next_open - 1.6 * row["ATR14"]
                tp = next_open + (next_open - stop) * DEFAULT_RR
                risk_ps = max(next_open - stop, 0.01)
                qty = int((ACCOUNT_SIZE * RISK_PER_TRADE) / risk_ps)
                pos = {"symbol": symbol, "side": "LONG", "entry_date": dt, "entry": next_open, "stop": stop, "tp": tp, "qty": qty, "bars": 0}
            elif short_setup and np.isfinite(row["ATR14"]) and row["ATR14"] > 0:
                stop = next_open + 1.6 * row["ATR14"]
                tp = next_open - (stop - next_open) * DEFAULT_RR
                risk_ps = max(stop - next_open, 0.01)
                qty = int((ACCOUNT_SIZE * RISK_PER_TRADE) / risk_ps)
                pos = {"symbol": symbol, "side": "SHORT", "entry_date": dt, "entry": next_open, "stop": stop, "tp": tp, "qty": qty, "bars": 0}
            continue

        pos["bars"] += 1
        low = float(next_bar["Low"])
        high = float(next_bar["High"])
        close = float(next_bar["Close"])

        exit_reason = None
        exit_px = None

        if pos["side"] == "LONG":
            if low <= pos["stop"]:
                exit_px, exit_reason = pos["stop"], "STOP"
            elif high >= pos["tp"]:
                exit_px, exit_reason = pos["tp"], "TARGET"
        else:
            if high >= pos["stop"]:
                exit_px, exit_reason = pos["stop"], "STOP"
            elif low <= pos["tp"]:
                exit_px, exit_reason = pos["tp"], "TARGET"

        if exit_reason is None and pos["bars"] >= 10:
            exit_px, exit_reason = close, "TIME"

        if exit_reason is not None:
            gross = (exit_px - pos["entry"]) * pos["qty"] if pos["side"] == "LONG" else (pos["entry"] - exit_px) * pos["qty"]
            costs = COMMISSION_PER_TRADE * 2 + (pos["entry"] * pos["qty"] + exit_px * pos["qty"]) * (SLIPPAGE_BPS / 10000)
            pnl = gross - costs
            trades.append({
                "symbol": symbol,
                "side": pos["side"],
                "entry_date": pos["entry_date"],
                "exit_date": dt,
                "entry": round(pos["entry"], 4),
                "exit": round(exit_px, 4),
                "qty": pos["qty"],
                "reason": exit_reason,
                "pnl": round(pnl, 2),
                "bars": pos["bars"],
            })
            pos = None

    return trades


def monte_carlo(pnls: np.ndarray, runs: int) -> pd.DataFrame:
    if len(pnls) == 0:
        return pd.DataFrame()

    records = []
    rng = np.random.default_rng(42)

    for i in range(runs):
        sample = rng.choice(pnls, size=len(pnls), replace=True)
        eq = np.cumsum(sample)
        dd = eq - np.maximum.accumulate(eq)
        records.append({
            "run": i + 1,
            "net_profit": round(float(eq[-1]), 2),
            "max_drawdown": round(float(dd.min()), 2),
            "win_rate": round(float((sample > 0).mean() * 100), 2),
        })

    return pd.DataFrame(records)


def run_backtest() -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = download_prices(BENCHMARKS + TOP30_TICKERS, LOOKBACK_BACKTEST)
    if raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        print("Backtest non disponibile: dati Yahoo assenti.")
        return pd.DataFrame(), pd.DataFrame()

    bench_mask = raw.columns.get_level_values(1).isin(BENCHMARKS)
    market_mask = raw.columns.get_level_values(1).isin(TOP30_TICKERS)

    bench_raw = raw.loc[:, bench_mask]
    market_raw = raw.loc[:, market_mask]

    regime_map = {b: compute_indicators(to_single_symbol_df(bench_raw, b).dropna()) for b in BENCHMARKS}

    all_trades = []
    for sym in TOP30_TICKERS:
        sdf = to_single_symbol_df(market_raw, sym)
        if sdf.empty:
            continue
        all_trades.extend(backtest_symbol(sym, sdf, regime_map))

    trades_df = pd.DataFrame(all_trades)
    if trades_df.empty:
        return trades_df, pd.DataFrame()

    trades_df["cum_pnl"] = trades_df["pnl"].cumsum()

    ts = now_stamp()
    trades_df.to_csv(REPORT_DIR / f"backtest_trades_{ts}.csv", index=False)

    stats = {
        "trades": int(len(trades_df)),
        "win_rate_pct": round((trades_df["pnl"] > 0).mean() * 100, 2),
        "gross_profit": round(trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum(), 2),
        "gross_loss": round(trades_df.loc[trades_df["pnl"] <= 0, "pnl"].sum(), 2),
        "net_profit": round(trades_df["pnl"].sum(), 2),
        "avg_trade": round(trades_df["pnl"].mean(), 2),
        "best_trade": round(trades_df["pnl"].max(), 2),
        "worst_trade": round(trades_df["pnl"].min(), 2),
    }
    gp = max(stats["gross_profit"], 0.01)
    gl = abs(min(stats["gross_loss"], -0.01))
    stats["profit_factor"] = round(gp / gl, 2)

    eq = trades_df["cum_pnl"]
    dd = eq - eq.cummax()
    stats["max_drawdown"] = round(dd.min(), 2)

    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(REPORT_DIR / f"backtest_stats_{ts}.csv", index=False)

    mc = monte_carlo(trades_df["pnl"].values, MC_RUNS)
    if not mc.empty:
        mc.to_csv(REPORT_DIR / f"monte_carlo_{ts}.csv", index=False)

    return trades_df, stats_df


# =========================================================
# OUTPUT / DASHBOARD
# =========================================================
def print_scan_console(scan_df: pd.DataFrame, regime: MarketRegime) -> None:
    print("\n=== BOT V3.3 ELITE PATTERN + NEWS | DAILY SCAN ===")
    print(f"Timestamp: {utc_now_str()}")
    print(f"Market regime: {regime.direction} | score={regime.score} | {regime.comment}")
    print("\nTop opportunities:")
    cols = [
        "symbol", "signal", "score", "confidence", "elite_rank",
        "pattern_name", "pattern_quality", "news_bias",
        "entry", "stop_loss", "take_profit"
    ]
    print(scan_df[cols].head(TOP_N_SIGNALS).to_string(index=False))


def build_dashboard_html(scan_df: pd.DataFrame, regime: MarketRegime, backtest_stats: Optional[pd.DataFrame] = None) -> Path:
    open_df = load_csv_or_empty(OPEN_TRADES_CSV, [])
    closed_df = load_csv_or_empty(CLOSED_TRADES_CSV, [])

    top5 = scan_df.head(5).copy() if not scan_df.empty else pd.DataFrame()
    long_count = int((scan_df["signal"] == "LONG").sum()) if not scan_df.empty else 0
    short_count = int((scan_df["signal"] == "SHORT").sum()) if not scan_df.empty else 0
    watch_count = int((scan_df["signal"] == "WATCH").sum()) if not scan_df.empty else 0

    bt_html = "<p>No backtest stats in this run.</p>"
    if backtest_stats is not None and not backtest_stats.empty:
        bt_html = backtest_stats.to_html(index=False, border=0)

    top5_html = top5.to_html(index=False, border=0) if not top5.empty else "<p>No scan data.</p>"
    open_html = open_df.to_html(index=False, border=0) if not open_df.empty else "<p>No open trades.</p>"
    closed_html = closed_df.tail(20).to_html(index=False, border=0) if not closed_df.empty else "<p>No closed trades.</p>"

    html_text = f"""
    <html>
    <head>
        <meta charset='utf-8'>
        <title>BOT V3.3 ELITE PATTERN + NEWS Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 24px; background:#0f172a; color:#e2e8f0; }}
            .card {{ background:#111827; border-radius:16px; padding:20px; margin-bottom:18px; }}
            .grid {{ display:grid; grid-template-columns:repeat(4, 1fr); gap:16px; }}
            .kpi {{ font-size:28px; font-weight:bold; }}
            table {{ width:100%; border-collapse:collapse; background:#111827; }}
            th, td {{ border-bottom:1px solid #243042; padding:10px; text-align:left; }}
            th {{ background:#172033; }}
            .small {{ color:#94a3b8; font-size:13px; }}
        </style>
    </head>
    <body>
        <h1>BOT V3.3 ELITE PATTERN + NEWS</h1>
        <p class='small'>Generated: {utc_now_str()}</p>

        <div class='grid'>
            <div class='card'><div class='small'>Market regime</div><div class='kpi'>{regime.direction}</div><div>{regime.comment}</div></div>
            <div class='card'><div class='small'>Long signals</div><div class='kpi'>{long_count}</div></div>
            <div class='card'><div class='small'>Short signals</div><div class='kpi'>{short_count}</div></div>
            <div class='card'><div class='small'>Watch signals</div><div class='kpi'>{watch_count}</div></div>
        </div>

        <div class='card'>
            <h2>Top 5 setups</h2>
            {top5_html}
        </div>

        <div class='card'>
            <h2>Open trades</h2>
            {open_html}
        </div>

        <div class='card'>
            <h2>Closed trades</h2>
            {closed_html}
        </div>

        <div class='card'>
            <h2>Last backtest stats</h2>
            {bt_html}
        </div>
    </body>
    </html>
    """

    path = DASHBOARD_DIR / f"dashboard_{now_stamp()}.html"
    path.write_text(html_text, encoding="utf-8")

    latest = DASHBOARD_DIR / "dashboard_latest.html"
    latest.write_text(html_text, encoding="utf-8")
    return latest


def print_review() -> None:
    open_df = load_csv_or_empty(OPEN_TRADES_CSV, [])
    closed_df = load_csv_or_empty(CLOSED_TRADES_CSV, [])
    watchlist_df = load_csv_or_empty(WATCHLIST_CSV, [])

    print("\n=== REVIEW V3.3 ELITE PATTERN + NEWS ===")
    print("\nWatchlist top 10:")
    print(watchlist_df.to_string(index=False) if not watchlist_df.empty else "No watchlist yet.")
    print("\nOpen trades:")
    print(open_df.to_string(index=False) if not open_df.empty else "No open trades.")
    print("\nLast closed trades:")
    print(closed_df.tail(10).to_string(index=False) if not closed_df.empty else "No closed trades.")


# =========================================================
# AUTO LOOP
# =========================================================
def run_auto_loop(interval_minutes: int, with_backtest: bool = False) -> None:
    interval_seconds = max(interval_minutes, 1) * 60
    print(f"\n=== AVVIO MODALITÀ AUTO V3.3 ===")
    print(f"Intervallo: {interval_minutes} minuti")
    print("Premi CTRL+C per interrompere.\n")

    try:
        while True:
            print(f"\n--- NUOVO CICLO AUTO | {utc_now_str()} ---")

            scan_df, regime = run_scan(send_alerts=True)

            stats_df = None
            if with_backtest:
                _, stats_df = run_backtest()

            if not scan_df.empty:
                print_scan_console(scan_df, regime)
                send_telegram_html(build_telegram_summary(scan_df, regime))
            else:
                print("Nessun risultato scan.")

            dash_path = build_dashboard_html(
                scan_df if not scan_df.empty else pd.DataFrame(),
                regime,
                stats_df
            )
            print(f"Dashboard aggiornata: {dash_path}")
            print(f"Attendo {interval_minutes} minuti...\n")
            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\nModalità AUTO interrotta manualmente.")


# =========================================================
# MENU / MAIN
# =========================================================
def run_menu() -> None:
    while True:
        print("\n=== BOT V3.3 ELITE PATTERN + NEWS MENU ===")
        print("1) Daily scan")
        print("2) Backtest + Monte Carlo")
        print("3) Review watchlist and trades")
        print("4) Build HTML dashboard")
        print("5) Scan + dashboard + Telegram")
        print("6) Test Telegram")
        print("7) AUTO loop 15 min")
        print("8) AUTO loop personalizzato")
        print("0) Exit")

        choice = input("Select option: ").strip()

        if choice == "1":
            scan_df, regime = run_scan(send_alerts=False)
            if scan_df.empty:
                print("No scan results.")
            else:
                print_scan_console(scan_df, regime)

        elif choice == "2":
            _, stats_df = run_backtest()
            print("\nBacktest completed.")
            print(stats_df.to_string(index=False) if not stats_df.empty else "No stats.")

        elif choice == "3":
            print_review()

        elif choice == "4":
            scan_df, regime = run_scan(send_alerts=False)
            _, stats_df = run_backtest()
            path = build_dashboard_html(scan_df, regime, stats_df)
            print(f"Dashboard created: {path}")

        elif choice == "5":
            scan_df, regime = run_scan(send_alerts=True)
            _, stats_df = run_backtest()
            path = build_dashboard_html(scan_df, regime, stats_df)
            if scan_df.empty:
                print("No scan results.")
            else:
                print_scan_console(scan_df, regime)
                send_telegram_html(build_telegram_summary(scan_df, regime))
            print(f"Dashboard created: {path}")

        elif choice == "6":
            telegram_test()

        elif choice == "7":
            run_auto_loop(interval_minutes=15, with_backtest=False)

        elif choice == "8":
            raw = input("Intervallo minuti (es. 5 / 15 / 30): ").strip()
            try:
                interval = max(int(raw), 1)
            except Exception:
                interval = AUTO_INTERVAL_MINUTES_DEFAULT
            run_auto_loop(interval_minutes=interval, with_backtest=False)

        elif choice == "0":
            print("Exit.")
            return

        else:
            print("Invalid option.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BOT V3.3 ELITE PATTERN + NEWS")
    p.add_argument("--mode", choices=["scan", "backtest", "review", "dashboard", "telegram_test", "auto"], default=None)
    p.add_argument("--menu", action="store_true")
    p.add_argument("--interval-minutes", type=int, default=AUTO_INTERVAL_MINUTES_DEFAULT)
    p.add_argument("--auto-with-backtest", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.menu or args.mode is None:
        run_menu()
        return

    if args.mode == "scan":
        scan_df, regime = run_scan(send_alerts=False)
        if scan_df.empty:
            print("No scan results.")
        else:
            print_scan_console(scan_df, regime)

    elif args.mode == "backtest":
        _, stats_df = run_backtest()
        print(stats_df.to_string(index=False) if not stats_df.empty else "No backtest stats.")

    elif args.mode == "review":
        print_review()

    elif args.mode == "dashboard":
        scan_df, regime = run_scan(send_alerts=False)
        _, stats_df = run_backtest()
        path = build_dashboard_html(scan_df, regime, stats_df)
        print(f"Dashboard created: {path}")

    elif args.mode == "telegram_test":
        telegram_test()

    elif args.mode == "auto":
        run_auto_loop(interval_minutes=args.interval_minutes, with_backtest=args.auto_with_backtest)


if __name__ == "__main__":
    main()
