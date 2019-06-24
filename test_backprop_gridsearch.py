# Libraries
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from copy import deepcopy
import multiprocessing as mp
import progressbar
import os
from sklearn.model_selection import KFold
import sys

# Internal imports
from simplegp.Nodes.BaseNode import Node
from simplegp.Nodes.SymbolicRegressionNodes import *
from simplegp.Nodes.Backpropagation import Backpropagation
from simplegp.Fitness.FitnessFunction import SymbolicRegressionFitness
from simplegp.Evolution.Evolution import SimpleGP


# Load regression dataset
X, y = sklearn.datasets.load_diabetes( return_X_y=True )

# Set functions and terminals
functions = [
        AddNode(),
        SubNode(),
        MulNode(),
        DivNode()
        ]

# chosen function nodes
terminals = [ EphemeralRandomConstantNode() ]	# use one ephemeral random constant node
for i in range(X.shape[1]):
    terminals.append(FeatureNode(i))	# add a feature node for each feature

dir_name = "experiments_backprop_gridsearch"

def createExperiments(X, kf):
    experiments = []

    # set up experiements
    population = 512
    mutation_rate = 0.001
    crossover_rate = 1
    max_height = 2
    t_size = 8
    max_time = 20
    main_ga_parameters = (population, mutation_rate, crossover_rate, max_height,
                          t_size, max_time)

    backpropGeneration = [1, 2, 5, 10] # every how many generations to perfrom backprop
    initial = [0, 1] # whether to perform backprop on initial population
    bIterations = [1, 5, 10, 15] # how many iterations of backprop to perform
    learningRates = [0.1, 0.01, 0.001, 0.0001]
    uniformK = [0, 0.1, 0.25, 0.5, 0.75, 1]
    first_generations = [1, 3, 5, sys.maxsize]

    for fG in first_generations:
        for lr in learningRates:
            for initB in initial:
                for g in backpropGeneration:
                    for bIter in bIterations:
                        for u in uniformK:
                            extra_parameters = (lr, initB, u, g, bIter, fG)

                            i = 0 # index for logs
                            for train_index, test_index in kf.split(X):
                                X_train, X_test = X[train_index], X[test_index]
                                y_train, y_test = y[train_index], y[test_index]
                                experiments.append((i, main_ga_parameters, extra_parameters, X_train,
                                                    X_test, y_train, y_test))
                                i += 1
   
    return experiments


def do_experiment(experiment):
    i, (p, m, cr, mH, tSize, tim), (lr, initB, u, g, bIter, fG), X_train, X_test, y_train, y_test = experiment
    # Set fitness function
    fitness_function = SymbolicRegressionFitness( X_train, y_train )
    # Run GP
    backprop_function = Backpropagation( X_train, y_train, iters=bIter, learning_rate=lr,
                                        decayFunction = Backpropagation.NoDecay )
    sgp = SimpleGP(fitness_function, backprop_function, functions, terminals, pop_size = p,
                   mutation_rate=m, crossover_rate=cr, initialization_max_tree_height = mH,
                   tournament_size = tSize, max_time = tim, uniform_k = u,
                   backprop_selection_ratio = 1, backprop_every_generations = g,
                   initialBackprop = initB, first_generations = fG)
    _, _, _, runtime = sgp.Run(applyBackProp=True, iterationNum = i, dirName = dir_name)

    # Print results
    with open(sgp.dirName +"/" + sgp.logName, "a") as fp:

        # Show the evolved function
        final_evolved_function = fitness_function.elite
        nodes_final_evolved_function = final_evolved_function.GetSubtree()
        fp.write('Function found (' +str(len(nodes_final_evolved_function)) + 'nodes ):\n\t' + str(nodes_final_evolved_function) + "\n")
        # Print results for training set
        fp.write('Training\n\tMSE:'+ str(np.round(final_evolved_function.fitness,3)) +
                    '\n\tRsquared:' + str(np.round(1.0 - final_evolved_function.fitness / np.var(y_train),3)) + "\n")
        # Re-evaluate the evolved function on the test set
        test_prediction = final_evolved_function.GetOutput( X_test )
        test_mse = np.mean(np.square( y_test - test_prediction ))
        fp.write('Test:\n\tMSE:' + str(np.round( test_mse, 3)) +
                    '\n\tRsquared:'+ str(np.round(1.0 - test_mse / np.var(y_test),3)) + "\n")
        fp.write(runtime)

if __name__ == '__main__':

    np.random.seed(42)

    kf = KFold(n_splits = 10, random_state=None, shuffle=True)

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    e = createExperiments(X, kf)
    pool = mp.Pool(mp.cpu_count())
    print(len(e))

    print("running on " + str(mp.cpu_count()) + " cores")

    with progressbar.ProgressBar(max_value=len(e)) as bar:
        for i, _ in enumerate(pool.imap_unordered(do_experiment, e), 1):
            bar.update(i)
