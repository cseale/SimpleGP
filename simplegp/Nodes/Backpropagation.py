import numpy as np
from copy import deepcopy

class Backpropagation:

	def __init__( self, X_train, y_train ):
		self.X_train = X_train
		self.y_train = y_train
		self.iterations = 1000
		self.learning_rate = 0.01

	def Backprop( self, individual ):
		# TODO: boolean flag to turn off backprop for comparision runs

		# Track best individual over all iterations of gradient descent
		best_indiv = None
		lowest_mse = float("inf")

		# backwards prop for certain number of iterations
		# TODO: Check for convergence, and any other early stopping conditions
		for i in range(self.iterations):
			# forward prop
			output = individual.GetOutput( self.X_train )
			mse = np.mean ( np.square( self.y_train - output ) )

			# Track best individual
			if mse < lowest_mse:
				best_indiv = deepcopy(individual)
				lowest_mse = mse

			# hard coded MSE gradient
			grad_mse = -2 * (self.y_train - output)
			# do gradient descent
			individual.GradientDescent(grad_mse, self.learning_rate)
		return best_indiv
