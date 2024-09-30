from gp_lib import *
import operator
from engine import GP_pv
from utils import *
import pandas as pd
import csv

# 处理数据
df = pd.read_parquet('/Users/xuzhantan/Desktop/实习/GP/ashareeodprices.parquet')
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
future_returns = df.loc[:, df.columns.get_level_values(0).isin(future)]
stock_data = stock_data[:-2]
future_returns = future_returns[:-2]


primitives = {
    'self_int': (self_int, [int], int),
    'add': (operator.add, [np.ndarray, np.ndarray], np.ndarray),
    'sub': (operator.sub, [np.ndarray, np.ndarray], np.ndarray),
    'mul': (operator.mul, [np.ndarray, np.ndarray], np.ndarray),
    'div': (div, [np.ndarray, np.ndarray], np.ndarray),
    'if_then_else': (if_then_else, [np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray),
    'abs': (abs, [np.ndarray], np.ndarray),
    'power': (power, [np.ndarray, int], np.ndarray),
    'log': (log, [np.ndarray], np.ndarray),
    'mad_wash_outliers': (mad_wash_outliers, [np.ndarray], np.ndarray),
    'rank': (rank, [np.ndarray], np.ndarray),
    'rank_div': (rank_div, [np.ndarray, np.ndarray], np.ndarray),
    'delay': (delay, [np.ndarray, int], np.ndarray),
    'delta': (delta, [np.ndarray, int], np.ndarray),
    'ts_corr': (ts_corr, [np.ndarray, np.ndarray, int], np.ndarray),
    'ts_sum': (ts_sum, [np.ndarray, int], np.ndarray),
    'ts_mean': (ts_mean, [np.ndarray, int], np.ndarray),
    'ts_std': (ts_std, [np.ndarray, int], np.ndarray),
    'ts_cov': (ts_cov, [np.ndarray, np.ndarray, int], np.ndarray),
    'ts_min': (ts_min, [np.ndarray, int], np.ndarray),
    'ts_max': (ts_max, [np.ndarray, int], np.ndarray),
    'ts_rank': (ts_rank, [np.ndarray, int], np.ndarray),
}

terminals = {
    str(i): (i, int) for i in range(1, 11)
}
terminals.update({
    str(i): (i, float) for i in np.arange(-1, 10.0, 0.2)
})


toolbox_dict = {
    'weight': 1.0,
    'expr_min': 2,
    'expr_max': 10,
    'tournsize': 25,
    'mut_min': 0,
    'mut_max': 2,
    'mate_max': 10,
}

run_dict = {
    'gen': 4,
    'pop': 300,
    'select': 50,
    'crossover_prob': 0.6,
    'mutate_prob': 0.2,
}

factor1 = 'log(rank_div(amount, apb))'
factor2 = 'ts_mean(mul(sub(ts_sum(ts_mean(ts_cov(change, high, 11), 7), 4), add(div(power(abs(amount), 3), abs(amount)), ts_mean(abs(high), 3))), rank(power(div(ts_sum(amount, 2), add(low, vwap)), 13))), 19)'
factor3 = '(rank((vwap - close)) / rank((vwap + close)))'
factor4 = '(-1 * rank(ts_cov(rank(adjclose), rank(volume), 5)))'
factor5 = '(adjclose-ts_mean(adjclose,12))/ts_mean(adjclose,12)*100'
factor6 = 'div(mul(power(adjpreclose, 16), ts_std(adjclose, 12)), abs(abs(ts_sum(adjopen, 17))))'
factor7 = 'div((-1 * ((adjlow - adjclose) * power(adjopen,5))) , ((adjclose - adjhigh) * power(adjclose,5)))'
factor8 = 'rank(delta(((((high + low) / 2) * 0.2) + (vwap * 0.8)), 4) * -1)'
factors = [factor1, factor2, factor3, factor4, factor5, factor6, factor7, factor8]


if __name__ == '__main__':
    g1 = GP_pv()
    g1.init(len(features))
    g1.load_data(features, stock_data, future_returns, 20170104, 20190104)
    g1.init_terminals(terminals)
    g1.init_primitives(primitives)
    g1.init_tool_box(toolbox_dict)
    g1.set_eval_func(ic)
    g1.run(factors, run_dict)


    # factors, IC, ic_improvements = g1.run(factors, run_dict)

    # for i in range(len(ic_improvements)):
    #     formatted_ic_improvement = format(ic_improvements[i], ".5f")
    #     filename = str(factors[8+i]) + '_' + formatted_ic_improvement + '.txt'
    #     full_path = '/Users/xuzhantan/Desktop/实习/GP/alphas/' + filename
    #     with open(full_path, 'w', encoding='utf-8') as file:
    #         pass

    # rows = zip(factors, ic_improvements)
    #
    # with open('/Users/xuzhantan/Desktop/实习/GP/因子/因子5.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Factor", "IC Improvement"])
    #     writer.writerows(rows)

    # print("New Best factors: ", [str(factor) for factor in factors])
    # print("IC:", IC)
    # print("ic_improvements", ic_improvements)