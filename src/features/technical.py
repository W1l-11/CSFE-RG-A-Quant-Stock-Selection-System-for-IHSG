import numpy as np
import pandas as pd


def yang_zhang_vol(o, h, l, c, window=60):
    """Drift-independent, gap-aware volatility. Annualised."""
    log_ho, log_lo, log_co = np.log(h/o), np.log(l/o), np.log(c/o)
    log_oc = np.log(o / c.shift(1))
    log_cc = np.log(c / c.shift(1))

    rs = log_ho*(log_ho - log_co) + log_lo*(log_lo - log_co)

    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    v_o  = log_oc.rolling(n).var()
    v_c  = log_co.rolling(n).var()
    v_rs = rs.rolling(n).mean()

    return np.sqrt(v_o + k*v_c + (1 - k)*v_rs) * np.sqrt(252)


def build_technical(ohlcv: dict, cfg: dict) -> pd.DataFrame:
    """ohlcv: {'open','high','low','close','volume'} of [date x ticker]."""
    o, h, l = ohlcv["open"], ohlcv["high"], ohlcv["low"]
    c, v    = ohlcv["close"], ohlcv["volume"]
    out = {}

    # momentum: skip the most recent month (short-term reversal)
    for w in cfg["features"]["momentum_windows"]:
        out[f"mom_{w}"] = c.pct_change(w)
    out["mom_12_1"] = c.shift(21) / c.shift(252) - 1 

    # short-term reversal (strong and persistent on IDX retail flow) 
    out["reversal_5"] = -c.pct_change(5)

    # volatility 
    vw = cfg["features"]["vol_window"]
    out["vol_yz"] = yang_zhang_vol(o, h, l, c, vw)
    out["vol_ratio"] = (yang_zhang_vol(o,h,l,c,20) / out["vol_yz"])

    # risk-adjusted momentum 
    out["sharpe_63"] = (c.pct_change(63) / (out["vol_yz"] + 1e-9))

    # liquidity
    lw = cfg["features"]["liquidity_window"]
    value = c * v
    out["log_adv"] = np.log1p(value.rolling(lw).median())
    out["vol_surge"] = v.rolling(5).mean() / (v.rolling(60).mean() + 1)

    # Amihud illiquidity: price impact per unit of traded value
    ret = c.pct_change().abs()
    out["amihud"] = (ret / (value + 1)).rolling(lw).mean() * 1e12

    # distance from 52w high (strong on IDX; proxies overhead supply)
    out["dist_52w_high"] = c / c.rolling(252).max() - 1

    # trend structure 
    out["ma_ratio"] = c.rolling(20).mean() / c.rolling(100).mean() - 1

    # IDX-specific: auto-rejection proximity
    daily_ret = c.pct_change()
    out["ar_hit_freq"] = (daily_ret.abs() > 0.19).rolling(60).mean()

    return pd.concat(
        {k: df.stack(dropna=False) for k, df in out.items()}, axis=1
    ).rename_axis(["date", "ticker"])