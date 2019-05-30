# Libraries
import time
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
backprop_function = Backpropagation( X_train, y_train )

# Set functions and terminals
functions = [
	AddNode()
	, SubNode()
	, MulNode()
	, DivNode()
	, CosNode()
	, SinNode()
]
# chosen function nodes
terminals = [ EphemeralRandomConstantNode() ]	# use one ephemeral random constant node
for i in range(X.shape[1]):
	terminals.append(FeatureNode(i))	# add a feature node for each feature

# Open a log file of which the name is based on the current UNIX timestamp
# log_file = open(f"./logs/log-{int(time.time())}", "w")

# # TODO: Put the simple GP into a loop such that we can run all experiments on AWS etc, and collect all the data
# # Define parameters with which we will run the algorithm
# # TODO: Add values for which we would actually like to run the algorithm
# pop_sizes = {100}
# max_gen_vals = {100}
# apply_backprop_vals = {True}
# learning_rate_vals = {0.01}
# num_of_grad_descent_iters_vals = {5}

# for pop_size in pop_sizes:
#     for max_gens in  max_gen_vals:
#         for apply_backprop in apply_backprop_vals:
#             print(f"Running GP for popSize={pop_size}, maxGen={max_gen}, funcs={func_operators}, applyBP={apply_backprop}")

#             # Run GP
#             sgp = SimpleGP(fitness_function, backprop_function, functions, terminals, pop_size=100, max_generations=100)
#             sgp.Run()

# func_operators = [func_type_to_operator_str[type(func)] for func in functions]

# # TODO: Write all the below data to logs in some way.

# log_file.close()

# Print results
# Show the evolved function
final_evolved_function = fitness_function.elite
nodes_final_evolved_function = final_evolved_function.GetSubtree()
print ('Function found (',len(nodes_final_evolved_function),'nodes ):\n\t', nodes_final_evolved_function) # this is in Polish notation
# Print results for training set
print ('Training\n\tMSE:', np.round(final_evolved_function.fitness,3),
	'\n\tRsquared:', np.round(1.0 - final_evolved_function.fitness / np.var(y_train),3))
# Re-evaluate the evolved function on the test set
test_prediction = final_evolved_function.GetOutput( X_test )
test_mse = np.mean(np.square( y_test - test_prediction ))
print ('Test:\n\tMSE:', np.round( test_mse, 3),
	'\n\tRsquared:', np.round(1.0 - test_mse / np.var(y_test),3))
