import pandas as pd
import numpy as np


def piotroski_f_score(fin: pd.DataFrame) -> pd.Series:
    """
    fin: annual financials indexed by ticker, columns include
    net_income, cfo, total_assets, total_assets_prev, lt_debt,
    lt_debt_prev, current_assets, current_liab, (prev), shares,
    shares_prev, gross_profit, revenue, (prev).
    Returns 0-9. Higher = financially stronger.
    """
    s = pd.DataFrame(index=fin.index)

    roa      = fin.net_income / fin.total_assets
    roa_prev = fin.net_income_prev / fin.total_assets_prev

    # Profitability
    s["f1"] = (roa > 0).astype(int)
    s["f2"] = (fin.cfo > 0).astype(int)
    s["f3"] = (roa > roa_prev).astype(int)
    s["f4"] = (fin.cfo > fin.net_income).astype(int)  

    # Leverage / liquidity / dilution
    s["f5"] = (fin.lt_debt / fin.total_assets <
               fin.lt_debt_prev / fin.total_assets_prev).astype(int)
    cr      = fin.current_assets / fin.current_liab
    cr_prev = fin.current_assets_prev / fin.current_liab_prev
    s["f6"] = (cr > cr_prev).astype(int)
    s["f7"] = (fin.shares <= fin.shares_prev * 1.01).astype(int) 

    # Operating efficiency
    gm      = fin.gross_profit / fin.revenue
    gm_prev = fin.gross_profit_prev / fin.revenue_prev
    s["f8"] = (gm > gm_prev).astype(int)
    at      = fin.revenue / fin.total_assets
    at_prev = fin.revenue_prev / fin.total_assets_prev
    s["f9"] = (at > at_prev).astype(int)

    return s.sum(axis=1)


def build_fundamental(fin: pd.DataFrame, mcap: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=fin.index)

    out["f_score"] = piotroski_f_score(fin)

    # Valuation use earnings YIELD not P/E, so negative earnings rank
    # correctly at the bottom instead of blowing up to +inf.
    out["earnings_yield"] = fin.net_income / mcap
    out["book_to_price"]  = fin.total_equity / mcap
    out["fcf_yield"]      = (fin.cfo - fin.capex) / mcap
    out["sales_to_price"] = fin.revenue / mcap

    # Quality
    out["roe"] = fin.net_income / fin.total_equity.replace(0, np.nan)
    out["roic"] = fin.ebit * (1 - 0.22) / (
        fin.total_equity + fin.total_debt - fin.cash)
    out["gross_margin"] = fin.gross_profit / fin.revenue

    # Leverage DER above ~2 is a real default signal on IDX
    out["der"] = fin.total_debt / fin.total_equity.replace(0, np.nan)
    out["net_debt_ebitda"] = (fin.total_debt - fin.cash) / fin.ebitda

    # Accruals (Sloan): high accruals predict future underperformance
    out["accruals"] = (fin.net_income - fin.cfo) / fin.total_assets

    # Growth
    out["rev_growth"]  = fin.revenue / fin.revenue_prev - 1
    out["ni_growth"]   = fin.net_income / fin.net_income_prev.abs() - 1

    # IDX-critical: share count inflation. Serial diluters via
    # debt-to-equity conversion destroy per-share value silently.
    out["share_dilution"] = fin.shares / fin.shares_prev - 1

    # Negative equity = special monitoring board criterion #5
    out["negative_equity"] = (fin.total_equity < 0).astype(int)

    return out