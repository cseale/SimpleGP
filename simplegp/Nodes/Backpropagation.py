import numpy as np
from copy import deepcopy
import math

class Backpropagation:

	def __init__( self, X_train, y_train, iters, learning_rate, decayFunction, override_iterations = None):
		self.X_train = X_train
		self.y_train = y_train
		self.iterations = iters
		self.learning_rate = learning_rate
		self.decayFunction = decayFunction
		self.override_iterations = self.iterations if override_iterations is None else override_iterations

	def StepDecay(self, generation): # Halves the learning rate every 10 itertations
		newLr = self.learning_rate * math.pow(0.5, math.floor(generation / 10))
		return newLr

	def ExpDecay(self, generation): # Exponential Decay, -0.05 is hyperparam (res is 0.6 after 10 gens)
		newLr = self.learning_rate * math.exp(-0.05 * generation)
		return newLr

	def NoDecay(self, generation):
		return self.learning_rate

	def Backprop( self, individual, generation, override_iterations = False): # Generation is passed for decaying learning rate.

		if override_iterations: # We want an increased number of iterations for this individual
			itersToApply = self.override_iterations
			if self.override_iterations == self.iterations:
				print("Warning: Override iterations to be applied equals default iterations.")
		else: # We don't want the above: just apply the default iterations
			itersToApply = self.iterations
		# assume worst fitness possible at start
		previousFitness = float("inf")
		previousIndividual = None
		# backwards prop for certain number of iterations
		for i in range(itersToApply):

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
