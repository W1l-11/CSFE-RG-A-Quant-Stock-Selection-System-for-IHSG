import os, json, hashlib
from datetime import datetime, timedelta
import feedparser, pandas as pd
from anthropic import Anthropic
from loguru import logger

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

FEEDS = {
    "kontan":   "https://investasi.kontan.co.id/rss",
    "bisnis":   "https://market.bisnis.com/rss",
    "cnbc_id":  "https://www.cnbcindonesia.com/market/rss",
    "idx_news": "https://www.idx.co.id/rss/news",
}

SYSTEM = """You are a sell-side equity analyst covering Indonesian equities.
Score news for its likely effect on the stock over the NEXT 20 TRADING DAYS.

Return ONLY a JSON object, no preamble, no markdown fences:
{
  "sentiment": ,
  "materiality": ,
  "category": "",
  "red_flag": ,
  "reason": ""
}

Scoring rules:
- materiality < 0.3 for routine coverage, analyst chatter, price reports.
- red_flag = true ONLY for: going-concern doubt, disclaimer/adverse audit
  opinion, default or debt restructuring, regulatory sanction, bourse
  suspension, negative equity, fraud allegation, delisting risk,
  heavily dilutive rights issue or debt-to-equity conversion.
- Rumour, pump promotion, or unverified "insider" claims:
  category = "speculation", materiality <= 0.2, sentiment = 0.0.
  Do not reward hype."""


def fetch_news(tickers: list[str], company_names: dict,
               lookback_days: int = 7) -> pd.DataFrame:
    cutoff = datetime.now() - timedelta(days=lookback_days)
    rows = []
    for src, url in FEEDS.items():
        try:
            feed = feedparser.parse(url)
        except Exception as e:
            logger.warning(f"{src} failed: {e}")
            continue
        for e in feed.entries:
            pub = datetime(*e.published_parsed[:6]) if \
                  getattr(e, "published_parsed", None) else datetime.now()
            if pub < cutoff:
                continue
            text = f"{e.title} {getattr(e,'summary','')}"
            for t in tickers:
                name = company_names.get(t, "")
                if t in text.upper() or (name and name.lower() in text.lower()):
                    rows.append({
                        "ticker": t, "date": pub.date(), "source": src,
                        "title": e.title,
                        "text": text[:1500],
                        "id": hashlib.md5(
                            (t + e.title).encode()).hexdigest()[:12],
                    })
    return pd.DataFrame(rows).drop_duplicates("id")


def score_article(ticker: str, text: str) -> dict:
    try:
        r = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            system=SYSTEM,
            messages=[{"role": "user",
                       "content": f"Ticker: {ticker}\n\nArticle:\n{text}"}],
        )
        raw = "".join(b.text for b in r.content if b.type == "text")
        return json.loads(raw.replace("```json", "").replace("```", "").strip())
    except Exception as e:
        logger.warning(f"scoring failed for {ticker}: {e}")
        return {"sentiment": 0.0, "materiality": 0.0,
                "category": "routine", "red_flag": False, "reason": "error"}


def aggregate_sentiment(scored: pd.DataFrame, halflife_days: int = 5
                        ) -> pd.DataFrame:
    """Materiality-weighted, time-decayed sentiment per ticker per day."""
    scored = scored.copy()
    scored["date"] = pd.to_datetime(scored["date"])
    asof = scored["date"].max()
    age  = (asof - scored["date"]).dt.days
    scored["decay"] = 0.5 ** (age / halflife_days)
    scored["w"] = scored["materiality"] * scored["decay"]

    g = scored.groupby("ticker")
    out = pd.DataFrame({
        "news_sentiment": g.apply(
            lambda d: np.average(d.sentiment, weights=d.w)
            if d.w.sum() > 0 else 0.0),
        "news_volume":    g.size(),
        "news_max_mat":   g.materiality.max(),
        # THE VETO: any red flag in the window blocks the position entirely
        "red_flag":       g.red_flag.max().astype(int),
        "speculation_ratio": g.apply(
            lambda d: (d.category == "speculation").mean()),
    })
    return out