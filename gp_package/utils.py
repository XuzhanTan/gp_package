import numpy as np
from scipy.stats import rankdata

def standardize(A):
    mean = np.nanmean(A, axis=0, keepdims=True)
    std = np.nanstd(A, axis=0, keepdims=True)
    return np.where(std != 0, (A - mean) / std, np.nan)


def ic(compiled_data, future_returns):
    try:
        assert compiled_data.shape == future_returns.shape
        IC = ts_spearman(compiled_data, future_returns)
    except Exception:
        IC = -np.inf
    return IC


def ts_spearman(mat_x, mat_y):
    """Calculate Spearman correlation between two matrix"""
    assert len(mat_x.shape) == len(mat_y.shape) == 2
    num_rows = mat_x.shape[0]
    spearman_values = []
    for i in range(num_rows):
        valid_indices = np.logical_and.reduce(
            (~np.isnan(mat_x[i]), ~np.isnan(mat_y[i]), ~np.isinf(mat_x[i]), ~np.isinf(mat_y[i]), ~np.isneginf(mat_x[i]), ~np.isneginf(mat_y[i])))
        if np.sum(valid_indices) < 2:
            continue
        mat_x_row = mat_x[i][valid_indices]
        mat_y_row = mat_y[i][valid_indices]
        mat_x_ranked = rankdata(mat_x_row)
        mat_y_ranked = rankdata(mat_y_row)
        mean_x = np.nanmean(mat_x_ranked)
        mean_y = np.nanmean(mat_y_ranked)
        std_x = np.nanstd(mat_x_ranked)
        std_y = np.nanstd(mat_y_ranked)
        if std_x == 0 or std_y == 0:
            continue
        cov = np.nanmean((mat_x_ranked - mean_x) * (mat_y_ranked - mean_y))
        spearman = cov / (std_x * std_y)
        spearman_values.append(spearman)
    return np.nanmean(spearman_values)
