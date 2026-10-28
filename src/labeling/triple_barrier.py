import pandas as pd
import numpy as np

def ewma_vol(close: pd.Series, span: int = 30) -> pd.Series:
    return close.pct_change().ewm(span=span).std()

def triple_barrier(close: pd.Series, cfg: dict) -> pd.DataFrame:
    L = cfg["labeling"]
    h, u, d = L["horizon_days"], L["upper_mult"], L["lower_mult"]
    sigma = ewma_vol(close, L["vol_span"])
    out = []

    for i in range(len(close) - h):
        t0, p0, s = close.index[i], close.iloc[i], sigma.iloc[i]
        if not np.isfinite(s) or s == 0:
            continue
        hi, lo = p0 * (1 + u*s), p0 * (1 - d*s)
        path = close.iloc[i+1 : i+1+h]

        up   = path[path >= hi].index.min()
        down = path[path <= lo].index.min()
        vert = path.index[-1]

        touches = [t for t in (up, down, vert) if pd.notna(t)]
        t1 = min(touches)
        r  = close.loc[t1] / p0 - 1
        lbl = 1 if t1 == up else (-1 if t1 == down else int(np.sign(r)))
        out.append({"t0": t0, "t1": t1, "ret": r, "label": lbl})

    return pd.DataFrame(out).set_index("t0")

def sample_weights(events: pd.DataFrame,
                   close: pd.Series) -> pd.Series:
    idx = close.index
    conc = pd.Series(0.0, index=idx)
    for t0, t1 in events.t1.items():
        conc.loc[t0:t1] += 1.0
    conc = conc.replace(0, np.nan)

    logret = np.log(close).diff()
    w, uniq = {}, {}
    for t0, t1 in events.t1.items():
        seg = slice(t0, t1)
        w[t0]    = abs((logret[seg] / conc[seg]).sum())
        uniq[t0] = (1.0 / conc[seg]).mean()

    w = pd.Series(w) * pd.Series(uniq)
    return w / w.mean()

