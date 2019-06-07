# Libraries
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from copy import deepcopy
import multiprocessing as mp

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
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.5, random_state=42 )

# chosen function nodes
terminals = [ EphemeralRandomConstantNode() ]	# use one ephemeral random constant node
for i in range(X.shape[1]):
	terminals.append(FeatureNode(i))	# add a feature node for each feature


def createExperiments():
    # set up experiements
    populationSizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    mutationRates = [0, 0.001, 0.01, 0.1]
    crossoverRates = [0.1, 0.25, 0.5, 0.75, 1]
    maxHeights = [2, 4, 8]
    tourSize = [2, 4, 8]
    #maxNumEval = [5000, 10000]
    maxTime = [5, 10, 15, 20, 25, 30]
    numRep = 10 # number of repetitions

    experiments = []
    for i in range(numRep):
        for p in populationSizes:
            for m in mutationRates:
                for cr in crossoverRates:
                    for mH in maxHeights:
                        for tSize in tourSize:
                            for tim in maxTime:
                                experiments.append((i, p, m, cr, mH, tSize, tim))
    return experiments


def doExperiment(experiment):
    (i, p, m, cr, mH, tSize, tim) = experiment
    # Set fitness function
    fitness_function = SymbolicRegressionFitness( X_train, y_train )
    # Run GP
    backprop_function = Backpropagation( X_train, y_train, iters=5, learning_rate=0.001, decayFunction = Backpropagation.NoDecay )
    sgp = SimpleGP(fitness_function, backprop_function, functions, terminals, pop_size = p, mutation_rate=m, crossover_rate=cr, initialization_max_tree_height = mH, tournament_size = tSize, max_time = tim)	# other parameters are optional
    _, _, _, runtime = sgp.Run(applyBackProp=False, iterationNum = i)

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
    e = createExperiments()
    pool = mp.Pool(mp.cpu_count())
    pool.map(doExperiment, e)