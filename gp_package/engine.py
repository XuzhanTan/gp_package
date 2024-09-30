import random
from deap import base, creator, tools, gp
import operator
from utils import *
import logging
import csv
logging.basicConfig(level=logging.INFO)


class GP_pv:
    def __init__(self):
        self.pset = None
        self.stock_data = None
        self.future_returns = None
        self.eval_func = None
        self.toolbox = base.Toolbox()
        self.data_train = None
        self.data_validate = None
        self.return_train = None
        self.return_validate = None

    def init(self, n):
        '''
        init一个PrimitiveSetTyped， 输入为input n 的长度
        '''
        self.pset = gp.PrimitiveSetTyped("MAIN", [np.ndarray] * n, np.ndarray)

    def load_data(self, f_list, stock_data, future_returns, validation_date, test_date):
        data_train = stock_data[stock_data.index < validation_date]
        data_validate = stock_data[(stock_data.index >= validation_date) & (stock_data.index <= test_date)]
        data_test = stock_data[stock_data.index > test_date]
        return_train = future_returns[future_returns.index < validation_date]
        return_validate = future_returns[(future_returns.index >= validation_date) & (future_returns.index <= test_date)]
        return_test = future_returns[future_returns.index > test_date]
        self.data_train = [data_train[feature].values for feature in f_list]
        self.data_validate = [data_validate[feature].values for feature in f_list]
        self.stock_data = [stock_data[feature].values for feature in f_list]
        self.data_test = [data_test[feature].values for feature in f_list]
        self.return_train = return_train.values
        self.return_validate = return_validate.values
        self.return_test = return_test.values
        self.future_returns = future_returns.values
        arg_dict = {f'ARG{i}': name for i, name in enumerate(f_list)}
        self.pset.renameArguments(**arg_dict)

    def init_primitives(self, func_dict):
        for name, (func, in_types, out_type) in func_dict.items():
            self.pset.addPrimitive(func, in_types, out_type, name=name)

    def init_terminals(self, terminal_dict):
        for name, (value, type_) in terminal_dict.items():
            self.pset.addTerminal(value, type_, name=name)

    def set_eval_func(self, eval_func):
        self.eval_func = eval_func

    def init_tool_box(self, param):
        creator.create("FitnessMax", base.Fitness, weights=(param['weight'],))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=param['expr_min'], max_=param['expr_max'])
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.toolbox.register("select", tools.selTournament, tournsize=param['tournsize'])
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=param['mut_min'], max_=param['mut_max'])
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"),  max_value=param['expr_max']))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=param['expr_max']))



    def run(self, factors, run_dict):
        for i, factor in enumerate(factors, start=1):
            factor_list = []
            ic_improvements = []
            compiled_func = self.toolbox.compile(expr=factor)
            compiled_data_train = standardize(compiled_func(*self.data_train))
            compiled_data_val = standardize(compiled_func(*self.data_validate))
            IC_val = ic(compiled_data_val, self.return_validate)
            ic_improvements.append(IC_val)
            logging.info(f"Initial IC_val: {IC_val}")
            population = self.toolbox.population(n=run_dict['pop'])
            for individual in population:
                try:
                    compiled_func = self.toolbox.compile(expr=individual)
                    compiled_data = standardize(compiled_func(*self.data_train))
                    sum_data = compiled_data + compiled_data_train
                    individual.fitness.values = (ic(sum_data, self.return_train),)

                except Exception:
                    continue

            for gen in range(run_dict['gen']):
                logging.info(f"Starting generation {gen}")
                # 赋予每一个新生成因子一个IC值
                new_factors = self.toolbox.select(population, run_dict['select'])
                for new_factor in new_factors.copy():
                    if new_factor in factor_list:
                        new_factors.remove(new_factor)
                best_new_factor = None
                best_IC = IC_val
                for new_factor in new_factors:
                    new_func = self.toolbox.compile(expr=new_factor)
                    try:
                        new_data = standardize(new_func(*self.data_validate))
                    except Exception:
                        continue
                    new_total_data = compiled_data_val + new_data
                    new_IC = ic(new_total_data, self.return_validate)
                    if new_IC > best_IC:
                        best_new_factor = new_factor
                        best_IC = new_IC
                        logging.info(f"Found new best factor in generation {gen} with IC {best_IC}")
                if best_new_factor is not None:
                    best_new_func = self.toolbox.compile(expr=best_new_factor)
                    compiled_data_val += standardize(best_new_func(*self.data_validate))
                    ic_improvement = best_IC - IC_val
                    logging.info(f"IC improvement in generation {gen}: {ic_improvement}")
                    ic_improvements.append(ic_improvement)
                    IC_val = best_IC
                    factor_list.append(best_new_factor)
                    population.remove(best_new_factor)

                for child1, child2 in zip(new_factors[::2], new_factors[1::2]):
                    if random.random() < run_dict['crossover_prob']:
                        try:
                            self.toolbox.mate(child1, child2)
                        except IndexError:
                            continue
                        del child1.fitness.values
                        del child2.fitness.values
                for mutant in new_factors:
                    if random.random() < run_dict['mutate_prob']:
                        try:
                            self.toolbox.mutate(mutant)
                        except IndexError:
                            continue
                        del mutant.fitness.values

                invalid_ind = [ind for ind in new_factors if not ind.fitness.valid]
                for individual in invalid_ind:
                    try:
                        compiled_func = self.toolbox.compile(expr=individual)
                        compiled_data = standardize(compiled_func(*self.data_train))
                        sum_data = compiled_data + compiled_data_train
                        individual.fitness.values = (ic(sum_data, self.return_train),)
                    except Exception:
                        continue
                population.extend(invalid_ind)

            rows = zip(factor_list, ic_improvements)
            with open(f'/Users/xuzhantan/Desktop/实习/GP/因子/因子{i}.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Factor", "IC Improvement"])
                writer.writerows(rows)



    # def run(self, factors, run_dict):
    #     ic_improvements = []
    #     compiled_funcs = [self.toolbox.compile(expr=factor) for factor in factors]
    #     compiled_data_train = sum([standardize(func(*self.data_train)) for func in compiled_funcs])
    #     compiled_data_val = np.zeros(self.return_validate.shape)
    #     IC_val = 0
    #     for func in compiled_funcs:
    #         compiled_data_val += standardize(func(*self.data_validate))
    #         previous_IC = IC_val
    #         IC_val = ic(compiled_data_val, self.return_validate)
    #         ic_improvements.append(IC_val-previous_IC)
    #     logging.info(f"Initial IC_val: {IC_val}")
    #
    #     population = self.toolbox.population(n=run_dict['pop'])
    #     for individual in population:
    #         try:
    #             compiled_func = self.toolbox.compile(expr=individual)
    #             compiled_data = standardize(compiled_func(*self.data_train))
    #             sum_data = compiled_data + compiled_data_train
    #             individual.fitness.values = (ic(sum_data, self.return_train),)
    #         except Exception:
    #             continue
    #
    #     for gen in range(run_dict['gen']):
    #         logging.info(f"Starting generation {gen}")
    #         # 赋予每一个新生成因子一个IC值
    #         new_factors = self.toolbox.select(population, run_dict['select']) #锦标赛法
    #         for new_factor in new_factors.copy():  # 使用copy()方法来避免在循环过程中修改集合
    #             if new_factor in factors:
    #                 new_factors.remove(new_factor)
    #         best_new_factor = None
    #         best_IC = IC_val
    #         for new_factor in new_factors:
    #             new_func = self.toolbox.compile(expr=new_factor)
    #             try:
    #                 new_data = standardize(new_func(*self.data_validate))
    #             except Exception:
    #                 continue
    #             new_total_data = compiled_data_val + new_data
    #             new_IC = ic(new_total_data, self.return_validate)
    #             if new_IC > best_IC:
    #                 best_new_factor = new_factor
    #                 best_IC = new_IC
    #                 logging.info(f"Found new best factor in generation {gen} with IC {best_IC}")
    #         if best_new_factor is not None:
    #             best_new_func = self.toolbox.compile(expr=best_new_factor)
    #             compiled_data_val += standardize(best_new_func(*self.data_validate))
    #             ic_improvement = best_IC - IC_val
    #             logging.info(f"IC improvement in generation {gen}: {ic_improvement}")
    #             ic_improvements.append(ic_improvement)
    #             IC_val = best_IC
    #             factors.append(best_new_factor)
    #             population.remove(best_new_factor)
    #
    #         #进行crossover，mutation，将生成出的新新因子加入population，作为下一代的父代
    #         for child1, child2 in zip(new_factors[::2], new_factors[1::2]):
    #             if random.random() < run_dict['crossover_prob']:
    #                 self.toolbox.mate(child1, child2)
    #                 del child1.fitness.values
    #                 del child2.fitness.values
    #         for mutant in new_factors:
    #             if random.random() < run_dict['mutate_prob']:
    #                 self.toolbox.mutate(mutant)
    #                 del mutant.fitness.values
    #
    #         invalid_ind = [ind for ind in new_factors if not ind.fitness.valid]
    #         for individual in invalid_ind:
    #             try:
    #                 compiled_func = self.toolbox.compile(expr=individual)
    #                 compiled_data = standardize(compiled_func(*self.data_train))
    #                 sum_data = compiled_data + compiled_data_train
    #                 individual.fitness.values = (ic(sum_data, self.return_train),)
    #             except Exception:
    #                 continue
    #         population.extend(invalid_ind)
    #
    #     return factors, best_IC, ic_improvements




