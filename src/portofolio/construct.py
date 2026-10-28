import numpy as np, pandas as pd, cvxpy as cp
from sklearn.covariance import LedoitWolf
from src.data.universe import relative_tick_cost


def build_portfolio(alpha: pd.Series, returns: pd.DataFrame,
                    prices: pd.Series, sectors: pd.Series,
                    red_flags: pd.Series, gate: float,
                    prev_w: pd.Series, cfg: dict) -> pd.Series:
    P = cfg["portfolio"]

    # NEWS VETO 
    alpha = alpha[red_flags.reindex(alpha.index).fillna(0) == 0]
    if alpha.empty:
        return pd.Series(dtype=float)

    #  select top-k 
    names = alpha.nlargest(P["top_k"]).index.tolist()
    a = alpha[names].values

    # shrunk covariance
    R = returns[names].dropna(how="any").tail(252)
    if len(R) < 120:
        return pd.Series(dtype=float)
    Sigma = LedoitWolf().fit(R.values).covariance_ * 252

    # constrained mean-variance 
    n = len(names)
    w = cp.Variable(n)
    prev = prev_w.reindex(names).fillna(0).values

    # linear turnover cost, tick-size aware
    tc = np.array([relative_tick_cost(prices[t], cfg["costs"]["slippage_ticks"])
                   + cfg["costs"]["broker_fee_buy"] for t in names])

    risk_aversion = 3.0
    obj = cp.Maximize(a @ w
                      - risk_aversion * cp.quad_form(w, cp.psd_wrap(Sigma))
                      - tc @ cp.abs(w - prev))

    cons = [cp.sum(w) == 1,
            w >= 0, # long-only: no retail shorts
            w <= P["max_weight"]]

    for s in sectors[names].dropna().unique():
        mask = (sectors[names] == s).values.astype(float)
        cons.append(mask @ w <= P["max_sector_weight"])

    cp.Problem(obj, cons).solve(solver=cp.ECOS)
    if w.value is None:
        weights = pd.Series(1.0 / n, index=names) # fallback: equal weight
    else:
        weights = pd.Series(np.clip(w.value, 0, None), index=names)
        weights /= weights.sum()

    # volatility targeting
    port_vol = float(np.sqrt(weights.values @ Sigma @ weights.values))
    scale = min(1.0, P["target_vol_annual"] / max(port_vol, 1e-6))

    # regime gate 
    return weights * scale * gate  # remainder sits in cash