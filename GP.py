import operator
import bottleneck as bn
import random
from multiprocessing import Pool
from scipy.stats import spearmanr, rankdata
import pandas as pd
import numpy as np
from deap import base, creator, tools, gp, algorithms

df = pd.read_parquet('/Users/xuzhantan/Desktop/实习/GP/ashareeodprices.parquet')
df_reset = df.reset_index()
df = df_reset.pivot(index='trade_dt', columns='stockcode')
new_columns = {}
for stock in df.columns.get_level_values(1).unique():
    new_columns[('future_ret', stock)] = df[('ret', stock)].shift(-2)
df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
features = ['preclose', 'open', 'high', 'low', 'close', 'change', 'pctchange',
            'volume', 'amount', 'adjfactor', 'vwap', 'ret']
future = ['future_ret']
stock_data = df.loc[:, df.columns.get_level_values(0).isin(features)]
future_returns = df.loc[:, df.columns.get_level_values(0).isin(future)]
stock_data = stock_data[:-2]
future_returns = future_returns[:-2]


# Define new function
def div(x, y):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(y == 0, x, np.where(np.isnan(x) | np.isnan(y), np.nan, x / y))
    return result


def mpn(n):
    return int(np.ceil(n * .7))


#######################################################
# 点
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


# 时序
def delay(A, n):
    return np.concatenate([np.full(n, np.nan), A[:-n]])

def delta(A, n):
    return A - delay(A, n)


def reg_beta(A, B, n):
    """Calculate rolling linear regression beta of A against B over window n"""
    A = pd.Series(A)
    B = pd.Series(B)

    def single_regression(x, y):
        slope, _ = np.polyfit(x, y, 1)
        return slope

    betas = A.rolling(n).apply(lambda x: single_regression(x, B.loc[x.index]), raw=False)
    return betas.values


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
    """Calculate Spearman correlation between two matrix"""
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


# Creating a primitive set named "MAIN" for functions that take 12 matrix as input and return a matrix.
pset = gp.PrimitiveSetTyped("MAIN", [np.ndarray] * 12, np.ndarray)
# addPrimitive will add functions, and addTerminal will add constant values
pset.addPrimitive(operator.add, [np.ndarray, np.ndarray], np.ndarray, name='add')
pset.addPrimitive(operator.sub, [np.ndarray, np.ndarray], np.ndarray, name='sub')
pset.addPrimitive(operator.mul, [np.ndarray, np.ndarray], np.ndarray, name='mul')
pset.addPrimitive(div, [np.ndarray, np.ndarray], np.ndarray, name='div')
pset.addPrimitive(mpn, [int], int, name='mpn')
pset.addPrimitive(abs, [np.ndarray], np.ndarray, name='abs')
pset.addPrimitive(power, [np.ndarray, int], np.ndarray, name='power')
pset.addPrimitive(mad_wash_outliers, [np.ndarray], np.ndarray, name='mad_wash_outliers')
pset.addPrimitive(rank, [np.ndarray], np.ndarray, name='rank')
pset.addPrimitive(delay, [np.ndarray, int], np.ndarray, name='delay')
pset.addPrimitive(delta, [np.ndarray, int], np.ndarray, name='delta')
pset.addPrimitive(reg_beta, [np.ndarray, np.ndarray, int], np.ndarray, name='reg_beta')
pset.addPrimitive(ts_sum, [np.ndarray, int], np.ndarray, name="ts_sum")
pset.addPrimitive(ts_mean, [np.ndarray, int], np.ndarray, name='ts_mean')
pset.addPrimitive(ts_std, [np.ndarray, int], np.ndarray, name='ts_std')
pset.addPrimitive(ts_cov, [np.ndarray, np.ndarray, int], np.ndarray, name='ts_cov')
for i in range(1, 11):
    pset.addTerminal(i, int)
for i in np.arange(0.5, 10.0, 0.5):
    pset.addTerminal(i, float)
pset.renameArguments(ARG0='preclose')
pset.renameArguments(ARG1='open')
pset.renameArguments(ARG2='high')
pset.renameArguments(ARG3='low')
pset.renameArguments(ARG4='close')
pset.renameArguments(ARG5='change')
pset.renameArguments(ARG6='pctchange')
pset.renameArguments(ARG7='volume')
pset.renameArguments(ARG8='amount')
pset.renameArguments(ARG9='adjfactor')
pset.renameArguments(ARG10='vwap')
pset.renameArguments(ARG11='ret')

# Define individual and max/min
creator.create("FitnessMax", base.Fitness,
               weights=(1.0,))  # 定义如何存储和处理适应度值positive is maximization, and negative is minimization
creator.create("Individual", gp.PrimitiveTree,
               fitness=creator.FitnessMax)  # define an example of max individual in population
# Create functions using toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2,
                 max_=5)  # Create a function called expr that generates a single tree with depth of min2 and max8
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.expr)  # Create a function called individual that applied expr(creating trees) and generate an individual
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # Create a lists of individuals
toolbox.register("compile", gp.compile, pset=pset)  # Transfrom tree to function


def calculate_ic(compile_func, stock_data, future_returns):
    stock_data = [stock_data[feature].values for feature in features]
    compiled_data = compile_func(*stock_data)
    future_returns = future_returns.values
    assert compiled_data.shape == future_returns.shape
    IC = ts_spearman(compiled_data, future_returns)
    return IC


def evaluate(individual):
    func = toolbox.compile(expr=individual)
    try:
        IC = calculate_ic(func, stock_data, future_returns)
    except Exception as e:
        # print("An error occurred with individual:", individual)
        # print("Error:", str(e))
        IC = -np.inf
    return IC,


def evaluate_population(population):
    with Pool(2) as pool:
        fitnesses = pool.map(toolbox.evaluate, population)
    return fitnesses


# 定义进化，突变，交叉，选择
toolbox.register("evaluate", evaluate)
toolbox.register("evaluate_population", evaluate_population)
toolbox.register("select", tools.selTournament,
                 tournsize=2)  # Randomly select three individuals and choose the best one
toolbox.register("mate", gp.cxOnePoint)  # Crossover/mating, swap a subtree of two trees
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)  # Randomly generate new trees for mutation
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut,
                 pset=pset)  # Randomly select a point in the individual and replaces the sub-tree at that point with a new randomly generated tree
# adding constraint of height to the trees
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))

if __name__ == "__main__":
    population = toolbox.population(n=30)
    fitnesses = evaluate_population(population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    halloffame = tools.HallOfFame(20)  # 定义halloffame的大小
    halloffame.update(population)

    for gen in range(5):
        offspring = toolbox.select(population, len(population) - 20)  # 运用select，选择300个population里ic最高的
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:  # crossover probability
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        # Mutation
        for mutant in offspring:
            if random.random() < 0.1:  # mutation probability
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = evaluate_population(invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        offspring.extend(toolbox.clone(ind) for ind in halloffame[:20])
        population[:] = offspring
        halloffame.update(population)

    for i in range(len(halloffame)):
        print("Hall of Fame", i + 1)
        print(halloffame[i])
        print("Fitness:", halloffame[i].fitness.values)

    # 建立IC值最高的五个因子组合
    creator.create("IndividualGA", list, fitness=creator.FitnessMax)
    toolbox_GA = base.Toolbox()
    N = len(halloffame)
    toolbox_GA.register("individual", tools.initRepeat, creator.Individual, random.randrange, len(halloffame), n=5)
    toolbox_GA.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate_GA(individual):
        combined_alpha = sum(halloffame[i] for i in individual)
        return calculate_ic(combined_alpha, future_returns),

    def evaluate_GA_population(population):
        with Pool(2) as pool:
            fitnesses = pool.map(toolbox_GA.evaluate, population)
        return fitnesses


    toolbox_GA.register("evaluate_GA", evaluate_GA)
    toolbox_GA.register("evaluate_GA_population", evaluate_GA_population)
    toolbox_GA.register("select", tools.selTournament, tournsize=2)
    toolbox_GA.register("mate", tools.cxTwoPoint)
    toolbox_GA.register("mutate", tools.mutUniformInt, low=0, up=N - 1, indpb=0.2)


    pop = toolbox_GA.population(n=100)
    hof = tools.HallOfFame(1)
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, halloffame=hof, verbose=True)
    best_alpha = hof[0]
    print("The best combination is: ", best_alpha)
    print("The fitness value is: ", best_alpha.fitness.values)

    # fit = evaluate_population(population)
    # for ind, fit in zip(population, fit):
    #     ind.fitness.values = fit
    # hof = tools.HallOfFame(1)
    # hof.update(population)
    #
    # for gen in range(5):
    #     offspring = toolbox.select(pop, len(pop) - 1)
    #     offspring = list(map(toolbox.clone, offspring))
    #
    #     # Crossover
    #     for child1, child2 in zip(offspring[::2], offspring[1::2]):
    #         if random.random() < 0.5:  # crossover probability
    #             toolbox.mate(child1, child2)
    #             del child1.fitness.values
    #             del child2.fitness.values
    #     # Mutation
    #     for mutant in offspring:
    #         if random.random() < 0.1:  # mutation probability
    #             toolbox.mutate(mutant)
    #             del mutant.fitness.values
    #
    #     invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    #     fit = evaluate_population(invalid_ind)
    #     for ind, fit in zip(invalid_ind, fit):
    #         ind.fitness.values = fit
    #
    #     offspring.extend(toolbox.clone(ind) for ind in hof[:20])
    #     pop[:] = offspring
    #     hof.update(pop)

    # 创建ic值最高的组合
    # pset_combination = gp.PrimitiveSetTyped("COMBINATION", len(halloffame))
    # pset_combination.addPrimitive(operator.add, [np.ndarray, np.ndarray], np.ndarray, name='add')
    # pset_combination.addPrimitive(operator.sub, [np.ndarray, np.ndarray], np.ndarray, name='sub')
    # pset_combination.addPrimitive(operator.mul, [np.ndarray, np.ndarray], np.ndarray, name='mul')
    # pset_combination.addPrimitive(div, [np.ndarray, np.ndarray], np.ndarray, name='div')
    # for i in range(len(halloffame)):
    #     alpha = halloffame[i]
    #     pset_combination.renameArguments(**{f'ARG{i}': f'alpha{i + 1}'})
    # toolbox_combination = base.Toolbox()
    # toolbox_combination.register("expr", gp.genHalfAndHalf, pset=pset_combination, min_=2, max_=5)
    # toolbox_combination.register("individual", tools.initIterate, creator.Individual, toolbox_combination.expr)
    # toolbox_combination.register("population", tools.initRepeat, list, toolbox_combination.individual)
    # toolbox_combination.register("compile", gp.compile, pset=pset_combination)
    #
    #
    # def evaluate_combination(individual):
    #     func = toolbox_combination.compile(expr=individual)
    #     factors = [halloffame[i] for i in range(len(halloffame))]
    #     combined_factor = func(*factors)
    #     IC = calculate_ic(combined_factor, future_returns)
    #     return IC,
    #
    #
    # def evaluate_combination_population(population):
    #     with Pool(2) as pool:
    #         fitnesses_combination = pool.map(toolbox_combination.evaluate_combination, population)
    #     return fitnesses_combination
    #
    # toolbox_combination.register("evaluate_combination", evaluate_combination)
    # toolbox_combination.register("evaluate_combination_population", evaluate_combination_population)
    # toolbox_combination.register("select", tools.selTournament, tournsize=2)
    # toolbox_combination.register("mate", gp.cxOnePoint)
    # toolbox_combination.register("expr_mut", gp.genFull, min_=0, max_=2)
    # toolbox_combination.register("mutate", gp.mutUniform, expr=toolbox_combination.expr_mut, pset=pset_combination)
    # toolbox_combination.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))
    # toolbox_combination.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))
    #
    # population_combination = toolbox_combination.population(n=20)
    # fitnesses_combination = evaluate_combination_population(population_combination)
    # for ind, fit in zip(population_combination, fitnesses_combination):
    #     ind.fitness.values = fit
    # hof = tools.HallOfFame(5)
    # hof.update(population_combination)
    #
    # for gen in range(5):
    #     offspring = toolbox_combination.select(population_combination, len(population_combination) - 5)
    #     offspring = list(map(toolbox.clone, offspring))
    #
    #     for child1, child2 in zip(offspring[::2], offspring[1::2]):
    #         if random.random() < 0.5:
    #             toolbox_combination.mate(child1, child2)
    #             del child1.fitness.values
    #             del child2.fitness.values
    #     for mutant in offspring:
    #         if random.random() < 0.1:
    #             toolbox_combination.mutate(mutant)
    #             del mutant.fitness.values
    #
    #     invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    #     fitnesses_combination = evaluate_combination_population(invalid_ind)
    #     for ind, fit in zip(invalid_ind, fitnesses_combination):
    #         ind.fitness.values = fit
    #
    #     offspring.extend(toolbox_combination.clone(ind) for ind in hof[:5])
    #     population_combination[:] = offspring
    #     hof.update(population_combination)
    #
    #     for i in range(len(hof)):
    #         print("Hall of Fame", i + 1)
    #         print(hof[i])
    #         print("Fitness:", hof[i].fitness.values)
