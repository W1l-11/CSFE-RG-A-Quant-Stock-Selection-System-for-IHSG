import lightgbm as lgb
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from loguru import logger
from src.models.cv import PurgedWalkForward


def make_target(events: pd.DataFrame) -> pd.Series:
    """Cross-sectional percentile rank of triple-barrier return → [-1, 1]."""
    r = events.groupby(level="date")["ret"]
    return (2 * r.rank(pct=True) - 1).rename("y")


class AlphaModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.models, self.feature_names = [], None

    def fit(self, X, y, w, t1):
        cv = PurgedWalkForward(**self.cfg["cv"])
        self.feature_names = list(X.columns)
        ics = []

        for i, (tr, te) in enumerate(cv.split(X, t1)):
            m = lgb.LGBMRegressor(**self.cfg["model"], verbose=-1)
            m.fit(X.iloc[tr], y.iloc[tr],
                  sample_weight=w.iloc[tr],
                  eval_set=[(X.iloc[te], y.iloc[te])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

            pred = m.predict(X.iloc[te])
            # Information Coefficient, computed WITHIN each date
            fold = pd.DataFrame(
                {"p": pred, "y": y.iloc[te].values},
                index=X.index[te])
            ic = fold.groupby(level="date").apply(
                lambda d: spearmanr(d.p, d.y)[0] if len(d) > 5 else np.nan)
            ics.append(ic.mean())
            logger.info(f"fold {i}: IC = {ic.mean():.4f}  "
                        f"ICIR = {ic.mean()/ic.std():.2f}")
            self.models.append(m)

        self.cv_ic = float(np.nanmean(ics))
        logger.info(f"mean out-of-sample IC: {self.cv_ic:.4f}")
        return self

    def predict(self, X):
        """Average across folds — a natural ensemble, reduces variance."""
        p = np.mean([m.predict(X[self.feature_names])
                     for m in self.models], axis=0)
        s = pd.Series(p, index=X.index, name="alpha")
        # re-rank cross-sectionally so scores are comparable across dates
        return s.groupby(level="date").rank(pct=True) * 2 - 1

    def importance(self):
        imp = np.mean([m.feature_importances_ for m in self.models], axis=0)
        return (pd.Series(imp, index=self.feature_names)
                  .sort_values(ascending=False))