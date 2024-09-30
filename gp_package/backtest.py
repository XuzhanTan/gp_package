import operator
import bottleneck as bn
from scipy.stats import rankdata
import pandas as pd
import numpy as np
from deap import base, creator, tools, gp

# 处理数据
df = pd.read_parquet('/Users/xuzhantan/Desktop/中信建投/GP/ashareeodprices.parquet')
df['future_ret'] = df.groupby('stockcode')['ret'].shift(-2)
df['pvt'] = (df['change'] / df['close']) * df['volume']
df['avg_price'] = (df['close'] + df['high'] + df['low'])/3 * df['volume']
df['weighted_close'] = (2*df['adjclose'] + df['adjhigh'] + df['adjlow'])/4
df['volatility'] = (df.groupby('stockcode')['adjclose'].transform(lambda x: x.rolling(10).max()) -
                     df.groupby('stockcode')['adjclose'].transform(lambda x: x.rolling(10).min())) / df['adjclose']
df['swing'] = df['adjhigh'] - df['adjlow']
df['apb'] = np.log((df['high'] + df['low'] + df['close'] + df['open'])/(4*df['vwap']))
df_reset = df.reset_index()
df = df_reset.pivot(index='trade_dt', columns='stockcode')
mask = (df['tradestatus'] == '停牌')
df[mask] = np.nan
df = df.drop(columns = ['tradestatus', 'adjfactor'])
features = ['preclose', 'open', 'high', 'low', 'close', 'change', 'pctchange',
            'volume', 'amount', 'adjpreclose', 'adjopen', 'adjhigh', 'adjlow', 'adjclose', 'vwap', 'ret',  'pvt',
            'avg_price', 'weighted_close', 'volatility', 'swing', 'apb']
future = ['future_ret']
stock_data = df.loc[:, df.columns.get_level_values(0).isin(features)]
stock_data = stock_data[:-2]

def add(x, y):
    return operator.add(x,y)

def sub(x, y):
    return operator.sub(x, y)

def mul(x, y):
    return operator.mul(x, y)

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

factors=[]
alpha1 = pd.read_csv('/Users/xuzhantan/Desktop/实习/GP/因子/factor1.csv')
alpha2 = pd.read_csv('/Users/xuzhantan/Desktop/实习/GP/因子/factor2.csv')
alpha3 = pd.read_csv('/Users/xuzhantan/Desktop/实习/GP/因子/factor3.csv')
alpha4 = pd.read_csv('/Users/xuzhantan/Desktop/实习/GP/因子/factor4.csv')
alpha5 = pd.read_csv('/Users/xuzhantan/Desktop/实习/GP/因子/factor5.csv')
alpha1 = alpha1['Factor'].tolist()
alpha2 = alpha2['Factor'].tolist()
alpha3 = alpha3['Factor'].tolist()
alpha4 = alpha4['Factor'].tolist()
alpha5 = alpha5['Factor'].tolist()
stock_data = [stock_data[feature].values for feature in features]
preclose = stock_data[0]
open = stock_data[1]
high = stock_data[2]
low = stock_data[3]
close = stock_data[4]
change = stock_data[5]
pctchange = stock_data[6]
volume = stock_data[7]
amount = stock_data[8]
adjpreclose = stock_data[9]
adjopen = stock_data[10]
adjhigh = stock_data[11]
adjlow = stock_data[12]
adjclose = stock_data[13]
vwap = stock_data[14]
ret = stock_data[15]
pvt = stock_data[16]
avg_price = stock_data[17]
weighted_close = stock_data[18]
volatility = stock_data[19]
swing = stock_data[20]
apb = stock_data[21]

alpha_list = [alpha1, alpha2, alpha3, alpha4, alpha5]
data_dict = {}
for i, alpha in enumerate(alpha_list, start=1):
    data_combined = np.zeros(volume.shape)
    for factor in alpha:
        data_combined += standardize(eval(factor))
    data_dict[f'alpha{i}'] = data_combined

alpha_names = list((data_dict.keys()))
correlation = {}
for i in range(len(alpha_names)):
    for j in range(i+1, len(alpha_names)):
        data1 = data_dict[alpha_names[i]]
        data2 = data_dict[alpha_names[j]]
        corr = ts_spearman(data1, data2)
        correlation[(alpha_names[i], alpha_names[j])] = corr


future_returns = df.loc[:, df.columns.get_level_values(0).isin(future)]
future_returns = future_returns[:-2]
data1 = pd.DataFrame(data_dict['alpha1'], index=future_returns.index, columns=future_returns.columns.droplevel())
data2 = pd.DataFrame(data_dict['alpha2'], index=future_returns.index, columns=future_returns.columns.droplevel())

data1.to_parquet('/Users/xuzhantan/Desktop/实习/GP/stock_data1.parquet')
data2.to_parquet('/Users/xuzhantan/Desktop/实习/GP/stock_data2.parquet')



