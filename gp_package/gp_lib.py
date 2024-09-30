import numpy as np
import bottleneck as bn

# 数
def self_int(a):
    return int(a)

def self_float(a):
    return float(a)
def mpn(n):
    return int(np.ceil(n * .7))


def if_then_else(A, B, C, D):
    return np.where(A > B, C, D)

#######################################################
# 点

def abs(x):
    return np.abs(x)

def if_then_else(A, B, C, D):
    return np.where(A > B, C, D)

def log(x):
    result = np.sign(x)*np.log(abs(x))
    result[np.isinf(result)] = np.nan
    return result

def power(x, a):
    # x = np.clip(x, -10, 10)
    # a = np.clip(a, -3, 3)
    result = np.power(np.abs(x), a)
    result[np.isinf(result)] = np.nan
    return result

def div(x, y):
    return np.where(y == 0, x, np.where(np.isnan(x) | np.isnan(y), np.nan, x / y))


# 截面
def mad_wash_outliers(mat):
    """n倍mad清除outlier"""
    if len(mat.shape) != 2:
        return np.nan
    else:
        med_arr_cp = np.expand_dims(np.nanmedian(mat, axis=1), axis=1)
        mad_arr_cp = np.expand_dims(np.nanmedian(np.abs(mat - med_arr_cp), axis=1), axis=1)
        mat = np.clip(mat, med_arr_cp - mad_arr_cp * 8, med_arr_cp + mad_arr_cp * 8)
        return mat

def rank(x):
    return bn.nanrankdata(x, axis=1)

def rank_div(x, y):
    rank1 = rank(x)
    rank2 = rank(y)
    return rank1/rank2

# 时序
def delay(x, n):
    if n >= 1 and x.ndim > 0:
        result = np.roll(x, n, axis=0)
        result[:n] = np.nan
        return result
    else:
        return np.full_like(x, np.nan)

def delta(x, n):
    if n >= 1 and x.ndim > 0:
        return x - delay(x, n)
    else:
        return np.full_like(x, np.nan)

def ts_corr(mat_x, mat_y, n):
    if len(mat_x.shape) == len(mat_y.shape) == 2 and n > 1:
        std_x = ts_std(mat_x, n)
        std_y = ts_std(mat_y, n)
        std_x[std_x == 0] = np.nan
        std_y[std_y == 0] = np.nan
        result = ts_cov(mat_x, mat_y, n) / (std_x * std_y)
        return result
    else:
        return np.full_like(mat_x, np.nan)

def ts_sum(x, n):
    if n > 1 and x.ndim > 0:
        return bn.move_sum(x.T, n, mpn(n)).T
    else:
        return np.full_like(x, np.nan)

def ts_mean(x, n):
    if n > 1 and x.ndim > 0:
        return bn.move_mean(x.T, n, mpn(n)).T
    else:
        return np.full_like(x, np.nan)

def ts_std(x, n):
    if n > 1 and x.ndim > 0:
        return bn.move_std(x.T, n, mpn(n)).T
    else:
        return np.full_like(x, np.nan)

def ts_cov(mat_x, mat_y, n):
    if len(mat_x.shape) == len(mat_y.shape) == 2 and n > 1:
        min_periods = mpn(n)
        cov = bn.move_mean(mat_x * mat_y, n, min_periods, axis=0) \
          - (bn.move_mean(mat_x, n, min_periods, axis=0) * bn.move_mean(mat_y, n, min_periods, axis=0))
        return cov
    else:
        return np.full_like(mat_x, np.nan)

def ts_min(x, n):
    if n > 1 and x.ndim > 0:
        return bn.move_min(x.T, n, mpn(n)).T
    else:
        return np.full_like(x, np.nan)

def ts_max(x, n):
    if n > 1 and x.ndim > 0:
        return bn.move_max(x.T, n, mpn(n)).T
    else:
        return np.full_like(x, np.nan)

def ts_rank(x, n):
    """ 序列x过去n天末尾值的排序"""
    if n > 1 and x.ndim > 0:
        return bn.move_rank(x.T, n, mpn(n)).T
    else:
        return np.full_like(x, np.nan)


