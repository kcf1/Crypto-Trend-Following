import numpy as np
from sklearn.model_selection import BaseCrossValidator


class BlockBootstrapCV(BaseCrossValidator):
    """
    Block-bootstrap cross-validation.

    Parameters
    ----------
    block_size : int
        Length of each contiguous block.
    n_bootstraps : int, default=100
        Number of bootstrap samples (i.e. number of train/val splits).
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the block sampling.

    Notes
    -----
    * The data are first split into ``T - block_size`` overlapping blocks:
          block 0 : 0 … block_size-1
          block 1 : 1 … block_size
          …
          block T-block_size : T-block_size … T-1
    * For each bootstrap iteration we sample ``n_blocks_needed`` blocks
      **with replacement** and concatenate them to form a training set.
    * The *validation* set is the **original full data** (out-of-bootstrap
      evaluation).  This is the standard way block-bootstrap is used for
      variance reduction in hyper-parameter search.
    """

    def __init__(self, block_size, n_bootstraps=100, random_state=None):
        self.block_size = block_size
        self.n_bootstraps = n_bootstraps
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_bootstraps

    def split(self, X, y=None, groups=None):
        rng = np.random.default_rng(self.random_state)

        n_samples = X.shape[0]
        n_blocks = n_samples - self.block_size + 1          # T - block_size + 1
        if n_blocks <= 0:
            raise ValueError("block_size must be <= len(X)")

        # Pre-compute start indices of every possible block
        block_starts = np.arange(n_blocks)                 # 0,1,…,n_blocks-1

        # How many blocks do we need to approximately recover the original length?
        n_blocks_needed = int(np.ceil(n_samples / self.block_size))

        for _ in range(self.n_bootstraps):
            # ---- TRAIN indices -------------------------------------------------
            chosen_starts = rng.choice(
                block_starts, size=n_blocks_needed, replace=True
            )
            train_idx = np.concatenate([
                np.arange(start, start + self.block_size)
                for start in chosen_starts
            ])
            # trim / wrap if we overshoot (very rare)
            train_idx = train_idx[train_idx < n_samples]

            # ---- VALIDATION indices (full original data) -----------------------
            val_idx = np.arange(n_samples)

            yield train_idx, val_idx