# Libraries
import math
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from copy import deepcopy

# Internal imports
from simplegp.Nodes.BaseNode import Node
from simplegp.Nodes.SymbolicRegressionNodes import *
from simplegp.Nodes.Backpropagation import Backpropagation
from simplegp.Fitness.FitnessFunction import SymbolicRegressionFitness
from simplegp.Evolution.Evolution import SimpleGP

np.random.seed(42)

# Load regression dataset 
X, y = sklearn.datasets.load_diabetes( return_X_y=True )
# Take a dataset split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.5, random_state=42 )
# Set fitness function
fitness_function = SymbolicRegressionFitness( X_train, y_train )

# Set functions and terminals
functions = [
	AddNode()
	, SubNode()
	, MulNode()
]
# chosen function nodes
terminals = [ EphemeralRandomConstantNode() ]	# use one ephemeral random constant node
for i in range(X.shape[1]):
	terminals.append(FeatureNode(i))	# add a feature node for each feature

# Find the best amount of iterations for backpropagation
steps_vals = range(20)
learning_rates = [math.pow(10, i) for i in range(-5, 1)]
best_steps_val = -1
best_fitness = float("inf")

log_file = open("./logs/learning_rate_and_iterations_experiments.txt", "w")
log_file.write("learning_rate iterations train_mse test_mse runtime evals\n")

# Optimal parameters for simpleGP symbolic regression without backprop
tour_size = 8
max_height = 2
cross_rate = 1.0
mut_rate = 0.001
pop_size = 512
max_time = 20

for lr in learning_rates:
    for steps in steps_vals:
        avg_fitness = 0
        for i in range(10):  # Run each experiment 10 times, because of stochasticity
            backprop_function = Backpropagation(X_train, y_train, iters=steps, learning_rate=lr)
            sgp = SimpleGP(fitness_function,
                           backprop_function,
                           functions,
                           terminals,
                           pop_size=pop_size,
                           max_generations=100,
                           mutation_rate=mut_rate,
                           crossover_rate=cross_rate,
                           initialization_max_tree_height=max_height,
                           max_time=max_time,
                           tournament_size=tour_size)
            _, _, _, runtime = sgp.Run(applyBackProp=True)

            # Log results
            final_evolved_function = fitness_function.elite
            nodes_final_evolved_function = final_evolved_function.GetSubtree()
            test_prediction = final_evolved_function.GetOutput( X_test )

            train_mse = final_evolved_function.fitness
            test_mse = np.mean(np.square( y_test - test_prediction ))
            evals = fitness_function.evaluations

            log_file.write(f"{lr} {steps} {train_mse} {final_evolved_function} {test_mse} {runtime} {evals}\n")

log_file.close()
