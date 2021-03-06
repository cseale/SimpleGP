# Libraries
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

# Extract the log file's contents into a data frame for easy processing
filepath = "./logs/learning_rate_and_iterations_experiments.txt"
df = pd.read_csv(filepath, sep=" ")

cv_means_df = pd.DataFrame(index=range(114), columns=df.columns)
for i in range(114):
    j = 10*i
    means = df[j:(j + 10)].mean()
    for col in cv_means_df.columns.values:
        cv_means_df.loc[i, col] = means[col]


def create_means_df(var_name, variables):
    means_df = pd.DataFrame(index=range(len(variables)), columns=df.columns)
    for (i, var) in enumerate(variables):
        rounded_floats = [np.around(num, decimals=6) for num in cv_means_df[var_name]]
        selection_cond = np.isclose(rounded_floats, var)
        means = cv_means_df[selection_cond].mean()
        for col in cv_means_df.columns.values:
            means_df.loc[i, col] = means[col]
    
    return means_df

var_means_dfs = {
    "learning_rate": create_means_df("learning_rate", learning_rates),
    "iterations": create_means_df("iterations", steps_vals)
}

# Find mean difference between train and test MSE for each learning rate
lr_dfs = [df[(df.learning_rate == lr) & (df.iterations == 15)] for lr in learning_rates]
lr_mse_diffs = [(lr_df.test_mse - lr_df.train_mse).abs().mean() for lr_df in lr_dfs]

# Separately plot the influence of the learning rate and the amount of
# iterations on the MSEs, the amount of generations, the amount of nodes, and
# the runtime.
for var in ["learning_rate", "iterations"]:
    var_str = var.replace("_", " ").capitalize()

    # When evaluating one variable, keep the other constant
    sub_df = df[df.iterations == 15] if var == "learning_rate" else df[df.learning_rate == 0.001]

    # MSEs in box plots. Force the whiskers to extend to min/max values.
    data = [sub_df[sub_df[var] == v].test_mse for v in sub_df[var].unique()]
    plt.boxplot(data, labels=sub_df[var].unique(), whis=float("inf"))
    plt.xlabel(var_str)
    plt.ylabel("Mean Square Error (MSE)")
    plt.title(f"{var_str} ~ Test MSE")
    # plt.savefig(f"./figs/{var}-vs-mse-box.png")
    plt.show()

    # Amount of nodes in the final evolved function
    fig1, ax = plt.subplots()
    ax.scatter(sub_df[var], sub_df.nodes_amnt)
    ax.set_xlabel(var_str)
    if var == "learning_rate":
        ax.set_xscale("log")
    unique_vars = sub_df[var].unique()
    prepend_var = 1e-07 if var == "learning_rate" else 0
    unique_vars = np.insert(unique_vars, 0, prepend_var)
    ax.set_xticks(unique_vars)
    ax.set_ylabel("Amount of nodes in the final function")
    plt.title(f"{var_str} ~ Amount of nodes")
    # plt.savefig(f"./figs/{var}-vs-nodes-scatter.png")
    plt.show()

# Plot gens vs. MSE to display the effect of backprop with each parameter comb
for lr in [1e-05, 1e-03, 1e-01]:
    for iters in [1, 5, 10, 15]:
        rounded_floats = [np.around(num, decimals=6) for num in df["learning_rate"]]
        selection_cond = (np.isclose(rounded_floats, lr)) & (df.iterations == iters)
        sub_df = df[selection_cond]
        plt.scatter(sub_df.gens, sub_df.test_mse, label=f"lr={lr}, iters={iters}")
plt.legend()
plt.title("Generations ~ Test MSE")
plt.xlabel("Generations")
plt.ylabel("Mean Square Error (MSE)")
# plt.savefig(f"./figs/lr-and-iters-gens-vs-mse-scatter.png")
plt.show()
