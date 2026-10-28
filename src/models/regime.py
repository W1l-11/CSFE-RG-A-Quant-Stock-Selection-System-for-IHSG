import numpy as np, pandas as pd
from hmmlearn.hmm import GaussianHMM
from loguru import logger


class RegimeModel:
    """3-state Gaussian HMM on IHSG. Produces the exposure gate g_t."""

    def __init__(self, cfg):
        self.n = cfg["regime"]["n_states"]
        self.exposure = np.array(cfg["regime"]["exposure_map"])
        self.hmm = None

    @staticmethod
    def _obs(idx_close: pd.Series) -> pd.DataFrame:
        r = np.log(idx_close).diff()
        return pd.DataFrame({
            "ret": r,
            "vol": r.rolling(20).std() * np.sqrt(252),
        }).dropna()

    def fit(self, idx_close: pd.Series):
        O = self._obs(idx_close)
        self.hmm = GaussianHMM(n_components=self.n, covariance_type="full",
                               n_iter=500, random_state=42)
        self.hmm.fit(O.values)

        # state 0 = best, state n-1 = worst
        order = np.argsort(-self.hmm.means_[:, 0])
        self._order = order
        for k, i in enumerate(order):
            logger.info(f"state {k}: mu_ret={self.hmm.means_[i,0]:+.5f} "
                        f"vol={self.hmm.means_[i,1]:.3f}")
        return self

    def gate(self, idx_close: pd.Series) -> pd.Series:
        O = self._obs(idx_close)
        _, post = self.hmm.score_samples(O.values)   # forward-backward

        gates = []
        for t in range(len(O)):
            # re-run forward pass on the prefix only
            _, p = self.hmm.score_samples(O.values[: t + 1])
            gamma = p[-1][self._order]
            gates.append(float(np.dot(gamma, self.exposure)))

        return pd.Series(gates, index=O.index, name="gate")