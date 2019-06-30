# Libraries
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
steps_vals = range(1, 20, 2)
learning_rates = [math.pow(10, i) for i in range(-6, 0)]
decay_func_params = \
    [(Backpropagation.StepDecay, k) for k in [5, 10, 15]] + \
    [(Backpropagation.ExpDecay, k) for k in [0.05, 0.1, 0.15]]

log_file = open("./logs/lr-iters-lr_decay-_experiments.txt", "w")
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

total_experiments = 10*len(learning_rates)*len(steps_vals)*len(decay_func_params)
counter = 0

for lr in learning_rates:
    for steps in steps_vals:
        for decay_func_param_pair in decay_func_params:
            # Run the experiments with 10-fold cross validation
            i = 0
            for (train_index, test_index) in kf.split(X):
                i += 1
                counter += 1
                decay_func_str = str(decay_func_param_pair[0]).split()[1]
                print(f"Running experiment {counter}/{total_experiments}, lr={lr}, iters={steps}, decay_f={decay_func_str}, k={decay_func_param_pair[1]} crossval iter {i}/10")

                # Reset fitness function object with new train/test sets
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                fitness_function = SymbolicRegressionFitness( X_train, y_train )

                # Extract decay func params, then set all backprop params
                (decay_func, k) = decay_func_param_pair
                backprop_function = Backpropagation(
                    X_train, y_train,
                    iters=steps, learning_rate=lr,
                    decayFunction=decay_func, decay_k=k
                )
                
                # Set SGP params
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
filepath = "./logs/lr-iters-lr_decay-_experiments.txt"
df = pd.read_csv(filepath, sep=" ")

# Add the decay-k parameter manually
decay_func_params = \
    [(Backpropagation.StepDecay, k) for k in [5, 10, 15]] + \
    [(Backpropagation.ExpDecay, k) for k in [0.05, 0.1, 0.15]]
decay_params_times_ten = []
for params_list in [[decay_func_params[i]] * 10 for i in range(len(decay_func_params))]:
    decay_params_times_ten.extend(params_list)
decay_f_series = pd.Series([str(pair[0]).split()[1].strip("Backpropagation.") for pair in (decay_params_times_ten * 60)])
decay_k_series = pd.Series([pair[1] for pair in (decay_params_times_ten * 60)])
df["decay_f"] = decay_f_series
df["decay_k"] = decay_k_series
full_df = df.copy()

decay_func_param_tups = [("StepDecay", k) for k in [5, 10, 15]] + \
                        [("ExpDecay", k) for k in [0.05, 0.1, 0.15]]
abbrev_f = {
    "StepDecay": "SD",
    "ExpDecay": "ED"
}

# Set params to optimal ones
(opt_lr, opt_iters) = (0.001, 13)
freeze_params = (df.learning_rate == opt_lr) & (df.iterations == opt_iters)
df = df[freeze_params]

# Find mean difference between train and test MSE for each decay function
decay_f_dfs = [df[(df.decay_f == f) & (df.decay_k == k)] for (f, k) in decay_func_param_tups]
decay_f_mse_diffs = [(dec_df.test_mse - dec_df.train_mse).abs().mean() for dec_df in decay_f_dfs]

# Construct a box plot to describe the test MSE for all decay functions
mse_data = [df[(df.decay_f == f) & (df.decay_k == k)].test_mse for (f, k) in decay_func_param_tups]
boxplot_labels = [f"({abbrev_f[pair[0]]}, {pair[1]})" for pair in decay_func_param_tups]
plt.boxplot(mse_data, labels=boxplot_labels, whis=float("inf"))
plt.xlabel("Decay function & parameter")
plt.ylabel("Mean Square Error (MSE)")
plt.title(f"Decay function ~ Test MSE for optimal lr and #iters")
# plt.savefig(f"./figs/lr_decay-func-vs-mse-box.png")
plt.show()

# Plot gens vs. MSE for each decay function using the optimal lr & iters params
colors = {
    ("StepDecay", 5): "blue",
    ("StepDecay", 10): "red",
    ("StepDecay", 15): "grey",
    ("ExpDecay", 0.05): "orange",
    ("ExpDecay", 0.1): "green",
    ("ExpDecay", 0.15): "purple"
}
opt_df = df
for (f, k) in decay_func_param_tups:
    sub_df = opt_df[(opt_df.decay_f == f) & (opt_df.decay_k == k)]
    plt.scatter(sub_df.evals, sub_df.test_mse, label=f"f={abbrev_f[f]}, k={k}", color=colors[(f, k)])
    plt.axhline(sub_df.test_mse.mean(), color=colors[(f, k)])
plt.legend()
plt.xlabel("Evaluations")
plt.ylabel("Mean Square Error (MSE)")
plt.title(f"Evaluations ~ Test MSE for optimal lr and #iters")
# plt.savefig(f"./figs/evals-vs-mse-lr-decay-opt-params.png")
plt.show()

# Plot correlation heat map to show the decay functions' lack of influence.
# Set up clean dataframes for the heat maps first.
corr_df = df.copy()
corr_df = corr_df.drop(["iterations", "learning_rate", "runtime"], axis=1)
corr_df_k_step = corr_df.copy()[corr_df.decay_f == "StepDecay"]
corr_df_k_step = corr_df_k_step.drop(["decay_f"], axis=1)
corr_df_k_exp = corr_df.copy()[corr_df.decay_f == "ExpDecay"]
corr_df_k_exp = corr_df_k_exp.drop(["decay_f"], axis=1)
corr_df = corr_df.drop(["decay_k"], axis=1)
corr_df["decay_f"] = pd.Series(df.decay_f.astype("category").cat.codes)

# Plot corr heat map for decay functions
sns.heatmap(corr_df.corr(), annot=True)
plt.tight_layout()
# plt.savefig("./figs/decay_f-corr-heatmap.png")
plt.show()

# Plot corr heat map for each function's parameter impact
sns.heatmap(corr_df_k_step.corr(), annot=True)
plt.tight_layout()
# plt.savefig("./figs/step-decay_k-corr_heatmap.png")
plt.show()
sns.heatmap(corr_df_k_exp.corr(), annot=True)
plt.tight_layout()
# plt.savefig("./figs/exp-decay_k-corr_heatmap.png")
plt.show()
