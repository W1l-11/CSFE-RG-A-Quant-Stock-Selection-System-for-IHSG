import numpy as np
import pandas as pd


class PurgedWalkForward:
    def __init__(self, n_splits=6, embargo_pct=0.01, train_min_days=750):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.train_min = train_min_days

    def split(self, X: pd.DataFrame, t1: pd.Series):
        #X indexed by (date, ticker); t1 = label end date per sample.
        dates = X.index.get_level_values("date").unique().sort_values()
        n = len(dates)
        embargo = int(n * self.embargo_pct)

        test_size = (n - self.train_min) // self.n_splits
        if test_size <= 0:
            raise ValueError("not enough history for this n_splits")

        for k in range(self.n_splits):
            te_start = self.train_min + k * test_size
            te_end   = min(te_start + test_size, n)
            if te_start >= n:
                break

            t_lo, t_hi = dates[te_start], dates[te_end - 1]
            emb_hi = dates[min(te_end - 1 + embargo, n - 1)]

            sample_dates = X.index.get_level_values("date")
            test_mask = (sample_dates >= t_lo) & (sample_dates <= t_hi)

            # PURGE: drop training samples whose LABEL WINDOW overlaps
            # [test_start, test_end + embargo]
            overlap = (t1.values >= t_lo) & (sample_dates <= emb_hi)
            train_mask = (sample_dates < t_lo) & (~overlap)

            yield np.where(train_mask)[0], np.where(test_mask)[0]