import numpy as np
import copy
import time
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

def ga_linear_model(predictors_dataset, Y, bools):
    '''implements the exhaustive search for the best feature combination.'''
    estimators_LR = []
    estimators_LR.append(('rescale', MinMaxScaler()))
    estimators_LR.append(('LR', LogisticRegression(max_iter = 10000)))
    pipe = Pipeline(estimators_LR)
    kfold = KFold(n_splits=10, shuffle = True, random_state=222)
    mask = [i==1 for i in bools]
    data = predictors_dataset[predictors_dataset.columns[mask]]
    scoring = 'accuracy'
    X = data.values
    X = np.nan_to_num(X)
    results = cross_val_score(pipe, X, Y, cv=kfold, scoring=scoring)
    results = np.array(results)
    return results.mean()

def Roulette_wheel_selection(alpha_population, fitness):
    N = fitness.sum()
    parents = []
    for i in range(2):
        n = np.random.uniform(0, N)
        wheel_position = 0
        if n == 0:
            parents.append(alpha_population[0])
        else:
            for gene_index in range(len(fitness)):
                wheel_position += fitness[gene_index]
                if wheel_position >= n:
                    parents.append(alpha_population[gene_index])
                    break
    return parents

def genetic_operator_crossover(parent_1, parent_2):
    if np.random.choice([False, True], p=[0.1, 0.9]):
        cross_over_point = np.random.choice(len(parent_1))
        child_1 = np.append(parent_1[:cross_over_point], parent_2[cross_over_point:])
        child_2 = np.append(parent_2[:cross_over_point], parent_1[cross_over_point:])
        return child_1, child_2
    return parent_1, parent_2

def genetic_operator_mutation(child, p = 0.1):
    if np.random.choice([False, True], p=[1-p, p]):
        mutation_point = np.random.choice(len(child))
        if child[mutation_point] == 0:
            child[mutation_point] = 1
        else:
            child[mutation_point] = 0
    return child

def genetic_algorithm(predictors_dataset, Y):
    start_time = time.time()
    highest_accuracy = 0
    best = []
    N_in_generation = 10
    current_generation = 0
    N_of_generations = 10
    no_of_columns = len(predictors_dataset.columns)
    alpha_population = [np.random.choice([0, 1], size=no_of_columns) for i in range(N_in_generation)]

    while current_generation < N_of_generations:
        fitness = np.array([ga_linear_model(predictors_dataset, Y, i) for i in alpha_population])
        print('Generation ', current_generation)
        #print('Average Accuracy:', fitness.mean(), 'Best Accuracy:', fitness.max())
        with open('GeneticAlgorithm.csv', 'a') as generational_history:
            generational_history.write(str(fitness.min()) + ", " + str(fitness.mean()) + ", " + str(fitness.max()) +'\n')

        #get results
        if fitness.max() > highest_accuracy:
            time_diff = time.time() - start_time
            highest_accuracy = fitness.max()
            best.append((list(copy.deepcopy(alpha_population[np.argmax(fitness)])), alpha_population[np.argmax(fitness)].sum(), fitness.max(), time_diff))

        next_gen = []
        #ensuring 2 mutants of best are always in the next generation
        best_mut1 = copy.deepcopy(alpha_population[np.argmax(fitness)])
        best_mut2 = copy.deepcopy(alpha_population[np.argmax(fitness)])
        best_mut1 = genetic_operator_mutation(best_mut1, 1)
        best_mut2 = genetic_operator_mutation(best_mut2, 1)
        while best_mut1.sum() == 0:
            genetic_operator_mutation(best_mut1, 1)
        while best_mut2.sum() == 0:
            genetic_operator_mutation(best_mut2, 1)
        next_gen.append(best_mut1)
        next_gen.append(best_mut2)
        while len(next_gen) != N_in_generation:
            #intermediate_population
            parent_1, parent_2 = Roulette_wheel_selection(alpha_population, fitness)
            child_1, child_2 = genetic_operator_crossover(parent_1, parent_2)
            child_1, child_2 = genetic_operator_mutation(child_1), genetic_operator_mutation(child_2)            
            #ensure no chromosome with all zero (no feature) and slow down convergence
            if child_1.sum() != 0 and child_2.sum() != 0 and list(child_1) not in [list(i) for i in next_gen]:
                next_gen.extend((child_1, child_2))
        alpha_population = next_gen
        current_generation += 1
    return best
