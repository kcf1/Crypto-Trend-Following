from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import make_scorer
import numpy as np

def pearson_corr(y_true, y_pred):
    corr, p_val = pearsonr(y_pred, y_true)
    loss = -corr if not np.isnan(corr) else -1
    return float(loss)

def pearson_scorer():
    return make_scorer(pearson_corr, greater_is_better=False)