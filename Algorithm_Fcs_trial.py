import numpy as np
from statistics import median
import math
import time
import random
import statistics

def sigmoid1(gamma):
    # print(gamma)
    if gamma < 0:
        return 1 - 1 / (1 + math.exp(gamma))
    else:
        return 1 / (1 + math.exp(-gamma))

def fitness(total_chi, pop_size, dimension, search_agent_matrix,sum_fitness_SA, fitness_SA, chi_val):

    number_of_classes = len(chi_val)
    all_fitness = np.zeros([pop_size, number_of_classes])
    for n in range(0, pop_size):
        for i in range(0, dimension):
            if search_agent_matrix[n][i] == 1:
                fitness_SA.append(chi_val[0:number_of_classes,i])
        # put all search agents fitnesses in one array and sort it
        #all_fitness.append(statistics.harmonic_mean(fitness_SA))
        print("Shape of fitness matrix for an agent", np.shape(fitness_SA))
        #    SUM THE CHI VALUES FOR EVERY CLASS
        for i in range(0, number_of_classes):
            sum_fitness_SA.append(sum(fitness_SA[:][i]))
        all_fitness[n] = sum_fitness_SA
        print("Sumed fitness for each class", sum_fitness_SA)
        fitness_SA.clear()
        sum_fitness_SA.clear()
    return all_fitness


def algorithm_fcs(total_chi, iter_nr, pop_size, chi_val, dimension):
    # initialize position of search agents/population
    """random matrix of (3-search agents,1144-words in the corpus/features)"""
    search_agent_matrix = np.zeros([pop_size, dimension])

    for i in range(0, pop_size):
        for j in range(0, dimension):
           search_agent_matrix[i][j] = int(random.uniform(0,2))
    #print("First agent \n", np.shape(search_agent_matrix[0:1, :]), "\n")
    iteration = 1
    best_ever = 0
    best_position = []
    sum_fitness_SA = []
    fitness_SA = []
    #all_fitness = []
    while iteration < iter_nr + 1:

        print("Current Iteration: ", iteration)
        # calculate the fitness of each individual
        all_fitness = fitness(total_chi, pop_size, dimension, search_agent_matrix,sum_fitness_SA, fitness_SA, chi_val)
        # put all search agents fitnesses in one array and sort it
        print("Search agent fitness unordered", all_fitness)
        all_fitness_sorted = all_fitness
        for n in range(0, pop_size-1):
            for m in range(0, pop_size-n-1):
                if(sum(np.subtract(all_fitness_sorted[m], all_fitness_sorted[m+1])) > 0):
                    #all_fitness_sorted[m], all_fitness_sorted[m+1] = all_fitness_sorted[m + 1], all_fitness_sorted[m]
                    all_fitness_sorted[m] = all_fitness_sorted[m] * all_fitness_sorted[m + 1]
                    all_fitness_sorted[m + 1] = all_fitness_sorted[m] / all_fitness_sorted[m + 1]
                    all_fitness_sorted[m] = all_fitness_sorted[m] / all_fitness_sorted[m + 1]
        print("Ascending order of search agents by fitness:", all_fitness_sorted)
        #print("Ascending order of search agents by fitness:", all_fitness_sorted)
        # OBTAIN AND STORE THE BEST ALL TIME INDIVIDUAL
        best_fitness = all_fitness[pop_size-1]
        print("Best fitness", best_fitness)
        print("All fitness length",len(all_fitness))
        if sum(np.subtract(best_ever, best_fitness)) < 0:
            best_ever = best_fitness
        print("Best ever", best_ever)
        for i in range(0, len(all_fitness)):
            if sum(np.subtract(all_fitness[i], best_fitness)) == 0:
                if iteration == 1:
                    best_position = search_agent_matrix[i][:]
                    break
                elif sum(np.subtract(best_fitness, best_ever)) > 0:
                    best_position = search_agent_matrix[i][:]
                    break
        print("Best position", best_position)
        worst_fitness = all_fitness[0]
        print("Worst fitness", worst_fitness)

        #print("all fitness:", all_fitness)
        # calculate the parameter "a"
        a = np.arctanh(-(iteration / iter_nr) + 1)
        b = 1 - iteration / iter_nr
        c = 0 + iteration / (4*iter_nr)
        # calculate the fitness weight of each slime mold
        weight = np.zeros([pop_size])
        for i in range(0, pop_size):
            """pow(2,-52) to avoid denominator 0"""
            fitness_formula_replacer = sum(np.subtract(best_fitness, all_fitness_sorted[i])) / (
                        sum(np.subtract(best_fitness, worst_fitness)) + pow(2, -52)) + 1
            print("Fitness formula shortcut", fitness_formula_replacer)

            if i >= (1/2*pop_size):
                weight[i] = (1 + np.random.uniform() * np.log10(fitness_formula_replacer))
                print("Strong weight", weight[i])
            else:
                weight[i] = (1 - np.random.uniform() * np.log10(fitness_formula_replacer))
                print("Week weight", weight[i])
            #print("fitness formula", fitness_formula_replacer)
            # UPDATE THE POSITIONS OF THE SEARCH AGENTS
            # calculate p parameter
        for i in range(0,pop_size):
            p = np.tanh(abs(median(np.subtract(all_fitness_sorted[i], best_ever))))
            # calculate vb and vc
            vb = random.uniform(-a, a)
            vc = random.uniform(-b, b)
            #print(len(search_agent_matrix[0][:]))
            A = random.randint(0, pop_size-1)
            B = random.randint(0, pop_size-1)
            r = random.random()
            if r < p:
                for j in range(0, dimension):
                    search_agent_matrix[i][j] = best_position[j] + vb * (int(weight[i]) * search_agent_matrix[A][j]
                                                                  - search_agent_matrix[B][j])
            else:
                for j in range(0, dimension):
                    search_agent_matrix[i][j] = vc * search_agent_matrix[i][j]


            for j in range(0, dimension):
                random.seed(time.time() * 200 + 999)
                r1 = random.random()
                if sigmoid1(search_agent_matrix[i][j]) > r1:
                    search_agent_matrix[i][j] = 1
                else:
                    search_agent_matrix[i][j] = 0
            print("Random value in sigmoid function:", r1)
        #print("best", best_position)
        for i in range(0, len(all_fitness)):
            if sum(np.subtract(all_fitness[i], best_fitness)) == 0:
                if iteration == 1:
                    best_position = search_agent_matrix[i][:]
                    break
                elif sum(np.subtract(best_fitness, best_ever)) > 0:
                    best_position = search_agent_matrix[i][:]
                    break
        print("Best position", best_position)

        iteration += 1
        """
        print("First fitness \n", fitness_SA1, "\n")
        print("Second fitness \n", fitness_SA2, "\n")
        print("Third fitness \n", fitness_SA3, "\n")
        """
        #print("These are the weights: ", weight)
    #print("last position", search_agent_matrix)
        counter = 0
        anti_counter = 0
        for i in range(0, dimension):
            if best_position[i] == 1:
                counter += 1
            elif best_position[i] == 0:
                anti_counter += 1
        print("Solution 1: ", counter, "and 0: ", anti_counter)


    return best_position
