import random
from .gp_definition import toolbox, evaluate_population
from deap import tools

def run_gp():
    population = toolbox.population(n=30)
    fitnesses = evaluate_population(population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    halloffame = tools.HallOfFame(20)
    halloffame.update(population)

    for gen in range(5):
        offspring = toolbox.select(population, len(population) - 20)
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < 0.1:
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

if __name__ == "__main__":
    run_gp()
