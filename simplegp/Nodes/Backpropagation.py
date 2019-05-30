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

		# assume worst fitness possible at start
		previousFitness = float("inf")
		previousIndividual = None
		
		# backwards prop for certain number of iterations
		# TODO: Check for convergence, and any other early stopping conditions
		for i in range(self.iterations):
			# forward prop
			output = individual.GetOutput( self.X_train )
			mse = np.mean ( np.square( self.y_train - output ) )

			# if there is a degradation, return previou individual
			if mse > previousFitness:
				return previousIndividual
				
			# mse improved, so keep descending
			# keep previous value
			previousFitness = mse
			previousIndividual = deepcopy(individual)
			# hard coded MSE gradient
			grad_mse = -2 * (self.y_train - output)
			# do gradient descent
			individual.GradientDescent(grad_mse, self.learning_rate)

		return individual
