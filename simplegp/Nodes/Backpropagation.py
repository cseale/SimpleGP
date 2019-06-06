import numpy as np
from copy import deepcopy
import math

class Backpropagation:

	def __init__( self, X_train, y_train, iters, learning_rate, decayFunction ):
		self.X_train = X_train
		self.y_train = y_train
		self.iterations = iters
		self.learning_rate = learning_rate
		self.decayFunction = decayFunction

	def StepDecay(self, generation): # Halves the learning rate every 10 itertations
		newLr = self.learning_rate * math.pow(0.5, math.floor(generation / 10))
		return newLr

	def ExpDecay(self, generation): # Exponential Decay, -0.05 is hyperparam (res is 0.6 after 10 gens)
		newLr = self.learning_rate * math.exp(-0.05 * generation)
		return newLr

	def NoDecay(self, generation):
		return self.learning_rate

	def Backprop( self, individual, generation): # Generation is passed for decaying learning rate.
		# assume worst fitness possible at start
		previousFitness = float("inf")
		previousIndividual = None

		# backwards prop for certain number of iterations
		for i in range(self.iterations):
			# forward prop
			output = individual.GetOutput( self.X_train )
			mse = np.mean ( np.square( self.y_train - output ) )

			# if there is a degradation, return previous individual
			if mse > previousFitness:
				return previousIndividual

			# mse improved, so keep descending
			# keep previous value
			previousFitness = mse
			previousIndividual = deepcopy(individual)
			# hard coded MSE gradient
			grad_mse = -2 * (self.y_train - output)

			# Decay the learning rate (function is passed in constructor, in test setup when defining backprop function)
			newLr = self.decayFunction(self,generation)

			# do gradient descent
			individual.GradientDescent(grad_mse, newLr)

		return individual
