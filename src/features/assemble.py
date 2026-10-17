import pandas as pd
import numpy as np


def cross_sectional_normalize(df: pd.DataFrame,
                              winsor: float = 0.02) -> pd.DataFrame:
    def _norm(g):
        g = g.clip(lower=g.quantile(winsor), upper=g.quantile(1 - winsor),
                   axis=1)
        # rank → uniform → inverse-normal → approximately N(0,1)
        r = g.rank(axis=0, pct=True)
        from scipy.stats import norm
        return pd.DataFrame(norm.ppf(r.clip(1e-6, 1 - 1e-6)),
                            index=g.index, columns=g.columns)

    return df.groupby(level="date", group_keys=False).apply(_norm)


def assemble(tech, fund, news, sectors, cfg):
    """Join all three families into one [date x ticker] feature panel."""
    X = tech.join(fund, how="left").join(news, how="left")

    # forward-fill fundamentals (quarterly → daily), cap the staleness
    fund_cols = fund.columns
    X[fund_cols] = (X[fund_cols]
                    .groupby(level="ticker")
                    .ffill(limit=120))

    X["news_sentiment"] = X["news_sentiment"].fillna(0.0)
    X["red_flag"]       = X["red_flag"].fillna(0)

    # sector as a categorical LightGBM handles this natively
    X["sector"] = X.index.get_level_values("ticker").map(sectors)
    X["sector"] = X["sector"].astype("category")

    num_cols = [c for c in X.columns
                if c not in ("sector", "red_flag")]
    X[num_cols] = cross_sectional_normalize(
        X[num_cols], cfg["features"]["winsorize_pct"])

    # sector-neutralise the numeric features: removes sector beta from
    # the signal so the model picks stocks, not sectors
    X[num_cols] = (X[num_cols]
                   - X.groupby(["date", "sector"], observed=True)[num_cols]
                      .transform("mean"))

    return X.dropna(thresh=int(len(num_cols) * 0.7))