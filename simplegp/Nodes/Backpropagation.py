import numpy as np
from copy import deepcopy
import math


class Backpropagation:

    def StepDecay(self, iteration): # Halves the learning rate every k iterations
        k = self.decay_k
        newLr = self.learning_rate * math.pow(0.5, math.floor((1+iteration)/k))
        return newLr

    def ExpDecay(self, iteration): # Exponential Decay, -0.05 is hyperparam (res is 0.6 after 10 gens)
        k = self.decay_k
        newLr = self.learning_rate * math.exp(-k * iteration)
        return newLr

    def NoDecay(self, iteration):
        return self.learning_rate

    def __init__( self, X_train, y_train, iters=5, learning_rate=0.01, decayFunction=NoDecay, decay_k=1, override_iterations = None ):
        self.X_train = X_train
        self.y_train = y_train
        self.iterations = iters
        self.learning_rate = learning_rate
        self.decayFunction = decayFunction
        self.decay_k = decay_k
        self.override_iterations = self.iterations if override_iterations is None else override_iterations

    def Backprop( self, individual, override_iterations = False ): # Generation is passed for decaying learning rate.
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
            newLr = self.decayFunction(self, i)
            # do gradient descent
            individual.GradientDescent(grad_mse, newLr)

        return individual
