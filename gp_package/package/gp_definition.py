from deap import base, creator, tools, gp
import operator
import numpy as np
from .utils import *


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
toolbox.register("compile", gp.compile, pset=pset)  # Transfrom tree

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
