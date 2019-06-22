# Libraries
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import KFold
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
# X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.5, random_state=42 )

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
# Learning rate 0.01 works well in our unit test, so we freeze it at that and
# try multiple values of iters for that.
steps_vals = range(1, 20)
learning_rates = [math.pow(10, i) for i in range(-6, 0)]

log_file = open("./logs/learning_rate_and_iterations_experiments.txt", "w")
_ = log_file.write("learning_rate iterations train_mse test_mse runtime evals gens nodes_amnt\n")

# Optimal parameters for simpleGP symbolic regression without backprop
tour_size = 8
max_height = 2
cross_rate = 1.0
mut_rate = 0.001
pop_size = 512
max_time = 20

# Optimal backprop params
uni_k = 0.5

# Use 10-fold cross validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

total_experiments = 10*len(learning_rates)*len(steps_vals)
counter = 0

for lr in learning_rates:
    for steps in steps_vals:
        # Run the experiments with 10-fold cross validation
        i = 0
        for (train_index, test_index) in kf.split(X):
            i += 1
            counter += 1
            print(f"Running experiment {counter}/{total_experiments}, lr={lr}, iters={steps}, crossval iter {i}/10")

            # Reset fitness function object with new train/test sets
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            fitness_function = SymbolicRegressionFitness( X_train, y_train )

            backprop_function = Backpropagation(X_train, y_train, iters=steps, learning_rate=lr)
            sgp = SimpleGP(
                fitness_function,
                backprop_function,
                functions,
                terminals,
                pop_size=pop_size,
                mutation_rate=mut_rate,
                crossover_rate=cross_rate,
                initialization_max_tree_height=max_height,
                tournament_size=tour_size,
                max_time=max_time,
                uniform_k=uni_k
            )

            _, _, _, runtime = sgp.Run(applyBackProp=True, iterationNum = i)

            # Log results
            final_evolved_function = fitness_function.elite
            nodes_final_evolved_function = final_evolved_function.GetSubtree()
            test_prediction = final_evolved_function.GetOutput( X_test )

            train_mse = final_evolved_function.fitness
            test_mse = np.mean(np.square( y_test - test_prediction ))
            evals = fitness_function.evaluations
            gens = sgp.generations
            nodes_amnt = len(nodes_final_evolved_function)

            _ = log_file.write(f"{lr} {steps} {train_mse} {test_mse} {runtime} {evals} {gens} {nodes_amnt}\n")

log_file.close()

exit()
# Extract the log file's contents into a data frame for easy processing
filepath = "./logs/learning_rate_and_iterations_experiments.txt"
df = pd.read_csv(filepath, sep=" ")

# Separately plot the influence of the learning rate and the amount of
# iterations on the MSEs, the amount of generations, the amount of nodes, and
# the runtime.
for var in ["learning_rate", "iterations"]:
    var_str = var.replace("_", " ").capitalize()

    # MSEs
    plt.plot(df[var], df.train_mse, label="Train MSE")
    plt.plot(df[var], df.test_mse, label="Test MSE")
    plt.legend()
    plt.xlabel(var_str)
    plt.ylabel("Mean Square Error (MSE)")
    plt.title(f"{var_str} ~ Train and test MSE")
    plt.show()

    # Gens
    plt.plot(df[var], df.gens)
    plt.xlabel(var_str)
    plt.ylabel("Generations")
    plt.title(f"{var_str} ~ Generations")
    plt.show()

    # Amount of nodes in the final evolved function
    plt.plot(df[var], df.nodes_amnt)
    plt.xlabel(var_str)
    plt.ylabel("Amount of nodes in the final function")
    plt.title(f"{var_str} ~ ")
    plt.show()

    # Runtime
    plt.plot(df[var], df.runtime)
    plt.xlabel(var_str)
    plt.ylabel("Runtime (s)")
    plt.title(f"{var_str} ~ Runtime")
    plt.show()
