import pandas as pd
import numpy as np
from loguru import logger

def tick_size(price : float) -> int:
    """IDX fractional tick size regime (Rp)."""
    if price < 200:    return 1
    if price < 500:    return 2
    if price < 2000:   return 5
    if price < 5000:   return 10
    return 25

def relative_tick_cost(price: float, n_ticks: int = 2) -> float:
    """Slippage as a fraction of price. This is the number that kills low-priced strategies: at Rp 100 it is 2%, at Rp 5000 it is 0.5%."""
    return (n_ticks * tick_size(price)) / price

def build_universe(prices: pd.DataFrame,
                   volumes: pd.DataFrame,
                   shares_out: pd.Series,
                   board_flags: pd.DataFrame,
                   cfg: dict,
                   asof: pd.Timestamp) -> list[str]:

    u = cfg["universe"]
    px  = prices.loc[:asof]
    vol = volumes.loc[:asof]

    if len(px) < u["min_listing_days"]:
        logger.warning(f"insufficient history at {asof}")

    last_px = px.iloc[-1]

    # liquidity: median daily traded VALUE, not volume 
    w = u.get("liquidity_window", 60)
    daily_value = (px * vol).iloc[-w:]
    adv = daily_value.median()
    liquid = adv >= u["min_adv_idr"]

    # price floor: avoids gocap distortion and tick-size drag 
    priced = last_px >= u["min_price_idr"]

    # market cap 
    mcap = last_px * shares_out
    big_enough = mcap >= u["min_market_cap_idr"]

    # special monitoring board (FCA)
    if u["exclude_special_monitoring"]:
        on_fca = board_flags.loc[:asof].iloc[-1].fillna(False)
        not_fca = ~on_fca.astype(bool)
    else:
        not_fca = pd.Series(True, index=last_px.index)

    # trading continuity: suspension / illiquidity
    zero_vol_days = (vol.iloc[-w:] == 0).sum()
    trades = zero_vol_days <= 10

    # sufficient history for the longest feature
    has_history = px.notna().sum() >= u["min_listing_days"]

    mask = liquid & priced & big_enough & not_fca & trades & has_history
    selected = mask[mask].index.tolist()

    # cap by liquidity if too wide
    if len(selected) > u["max_universe_size"]:
        selected = adv[selected].nlargest(u["max_universe_size"]).index.tolist()

    logger.info(f"{asof.date()}: universe = {len(selected)} names")
    return sorted(selected)