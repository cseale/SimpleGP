# Libraries
import numpy as np
import sklearn.datasets
from sklearn.model_selection import KFold
from copy import deepcopy
import multiprocessing as mp
import progressbar
import os

# Internal imports
from simplegp.Nodes.BaseNode import Node
from simplegp.Nodes.SymbolicRegressionNodes import *
from simplegp.Nodes.Backpropagation import Backpropagation
from simplegp.Fitness.FitnessFunction import SymbolicRegressionFitness
from simplegp.Evolution.Evolution import SimpleGP


np.random.seed(42)

# Set functions and terminals
functions = [
	AddNode(),
    SubNode(),
    MulNode(),
    DivNode()
]

# Load regression dataset
X, y = sklearn.datasets.load_diabetes( return_X_y=True )
# Take a dataset split
kf = KFold( n_splits=10, shuffle=True, random_state=42 )

# chosen function nodes
terminals = [ EphemeralRandomConstantNode() ]	# use one ephemeral random constant node
for i in range(X.shape[1]):
	terminals.append(FeatureNode(i))	# add a feature node for each feature

def createExperiments():
    experiments = []
    
    # number of runs
    number_of_runs = 30
    
    # set up experiements
    population = 512
    mutation_rate = 0.001
    crossover_rate = 1
    max_height = 2
    t_size = 8
    max_time = 20
    numRep = 30 # number of repetitions
    main_ga_parameters = (population, mutation_rate, crossover_rate, max_height, t_size, max_time)
    
    # define parameters for other experiments here
    backprop_every_generations = [1, 5, 10, 20, 50, 100]
    
    i = 0
    for train_index, test_index in kf.split(X):
        i += 1
        indices = (i, train_index, test_index)
        for every_gen in backprop_every_generations:
            experiments.append((indices, main_ga_parameters, every_gen))
   
    return experiments


def do_experiment(experiment):
    (i, train_index, test_index), (p, m, cr, mH, tSize, tim), backprop_every_generations = experiment
    # Cross validation
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # Set fitness function
    fitness_function = SymbolicRegressionFitness( X_train, y_train )
    # Run GP
    backprop_function = Backpropagation( X_train, y_train, iters=10, learning_rate=0.01, decayFunction = Backpropagation.NoDecay )
    sgp = SimpleGP(fitness_function, backprop_function, functions, terminals, pop_size = p, mutation_rate=m, crossover_rate=cr, initialization_max_tree_height = mH, tournament_size = tSize, max_time = tim, backprop_every_generations = backprop_every_generations)	# other parameters are optional
    _, _, _, runtime = sgp.Run(applyBackProp=True, iterationNum = i)

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
    dir_name = "experiments"

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    
    e = createExperiments()
    pool = mp.Pool(mp.cpu_count())

    print("running on " + str(mp.cpu_count()) + " cores")
    
    with progressbar.ProgressBar(max_value=len(e)) as bar:
        for i, _ in enumerate(pool.imap_unordered(do_experiment, e), 1):
            bar.update(i)