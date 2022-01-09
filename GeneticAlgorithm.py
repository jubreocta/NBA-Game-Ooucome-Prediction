import numpy as np
import pandas as pd
import time
import copy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

def ga_linear_model(predictors_dataset, Y, bools):
    '''implements the exhaustive search for the best feature combination.'''
    estimators_LR = []
    estimators_LR.append(('rescale', MinMaxScaler()))
    estimators_LR.append(('LR', LinearRegression()))
    pipe = Pipeline(estimators_LR)
    kfold = KFold(n_splits=5)
    mask = [i==1 for i in bools]
    data = predictors_dataset[predictors_dataset.columns[mask]]
    scoring = 'r2'
    X = data.values
    X = np.nan_to_num(X)
    results = cross_val_score(pipe, X, Y, cv=kfold, scoring=scoring)
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
    best=0.314950449239347
    best_results = []
    N_in_generation = 100
    current_generation = 0
    N_of_generations = 10000
    no_of_columns = len(predictors_dataset.columns)
    #alpha_population = [np.random.choice([0, 1], size=no_of_columns) for i in range(N_in_generation)]
    last_population = pd.read_csv('GAMostRecentPopulation.csv', index_col=0)
    alpha_population = last_population.to_numpy()
    while current_generation < N_of_generations:
        fitness = np.array([ga_linear_model(predictors_dataset, Y, i) for i in alpha_population])
        print('Generation', current_generation)
        print('Average Accuracy:', fitness.mean(), 'Best Accuracy:', fitness.max())
        #get results
        if fitness.max() > best:
            new_time = time.time()
            best = fitness.max()
            max_index = fitness.tolist().index(best)
            best_results.append([alpha_population[max_index].tolist(), alpha_population[max_index].sum(), best, new_time - start_time])
            start_time = new_time
            best_results = sorted(best_results, key = lambda x: x[2], reverse = True)
            dataset = pd.DataFrame.from_records(best_results, columns = ['col_ind', 'k', 'rsquared', 'time'])
            dataset.to_csv('FeatureSelection\GA.csv', index = False)
            pd.DataFrame(alpha_population).to_csv('GAMostRecentPopulation.csv')

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
            #ensure no chromosome with all zero (no feature) and slow down convergence
            if child_1.sum() != 0 and child_2.sum() != 0 and list(child_1) not in [list(i) for i in next_gen]:
                next_gen.extend((child_1, child_2))
        alpha_population = next_gen
        current_generation += 1