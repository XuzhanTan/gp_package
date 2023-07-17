import numpy as np
from scipy.stats import rankdata
import bottleneck as bn
from multiprocessing import Pool
from .gp_definition import toolbox


# Define new function
def div(x, y):
    return np.where(y == 0, x, x / y)

def mpn(n):
    return int(np.ceil(n * .7))

def abs(x):
    return np.abs(x)

def power(x, a):
    x = np.clip(x, -10, 10)
    a = np.clip(a, -3, 3)
    return np.power(x, a)


# 截面
def mad_wash_outliers(mat):
    """n倍mad清除outlier"""
    med_arr_cp = np.expand_dims(np.nanmedian(mat, axis=1), axis=1)
    mad_arr_cp = np.expand_dims(np.nanmedian(np.abs(mat - med_arr_cp), axis=1), axis=1)
    mat = np.clip(mat, med_arr_cp - mad_arr_cp * 8, med_arr_cp + mad_arr_cp * 8)
    return mat


def rank(x):
    return bn.nanrankdata(x, axis=1)

def ts_sum(x, n):
    assert n > 1
    return bn.move_sum(x.T, n, mpn(n)).T

def ts_mean(x, n):
    assert n > 1
    return bn.move_mean(x.T, n, mpn(n)).T


def ts_std(x, n):
    assert n > 1
    return bn.move_std(x.T, n, mpn(n)).T


def ts_cov(mat_x, mat_y, n):
    assert len(mat_x.shape) == len(mat_y.shape) == 2 and n > 1
    min_periods = mpn(n)
    cov = bn.move_mean(mat_x * mat_y, n, min_periods, axis=0) \
          - (bn.move_mean(mat_x, n, min_periods, axis=0) * bn.move_mean(mat_y, n, min_periods, axis=0))
    return cov * n / n


def ts_min(x, n):
    assert n > 1
    return bn.move_min(x.T, n, mpn(n)).T


def ts_max(x, n):
    assert n > 1
    return bn.move_max(x.T, n, mpn(n)).T


def ts_spearman(mat_x, mat_y):
    assert len(mat_x.shape) == len(mat_y.shape) == 2
    num_rows = mat_x.shape[0]
    spearman_values = []
    for i in range(num_rows):
        valid_indices = np.logical_and(~np.isnan(mat_x[i]), ~np.isnan(mat_y[i]))
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


def calculate_ic(compile_func, stock_data, future_returns, features):
    stock_data = [stock_data[feature].values for feature in features]
    compiled_data = compile_func(*stock_data)
    future_returns = future_returns.values
    assert compiled_data.shape == future_returns.shape
    IC = ts_spearman(compiled_data, future_returns)
    return IC

def evaluate(individual, stock_data, future_returns):
    func = toolbox.compile(expr=individual)
    try:
        IC = calculate_ic(func, stock_data, future_returns)
    except Exception as e:
        IC = -np.inf
    return IC,

def evaluate_population(population):
    with Pool(2) as pool:
        fitnesses = pool.map(toolbox.evaluate, population)
    return fitnesses
