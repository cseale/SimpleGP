import numpy as np
import os
from numpy.random import random
import numpy.random
import time
from copy import deepcopy
import sys

from simplegp.Variation import Variation
from simplegp.Selection import Selection


class SimpleGP:

    def __init__(self,
                 fitness_function,
                 backprop_function,
                 functions,
                 terminals,
                 pop_size=500,
                 crossover_rate=0.5,
                 mutation_rate=0.5,
                 max_evaluations=-1,
                 max_generations=-1,
                 max_time = -1,
                 initialization_max_tree_height = 4,
                 max_tree_size=100,
                 tournament_size = 4,
                 uniform_k = 1,
                 backprop_every_generations = 1,
                 backprop_selection_ratio = 1,
                 initialBackprop = 1,
                 first_generations = sys.maxsize,
                 reset_weights_on_variation = False
            ):
        self.pop_size = pop_size
        self.backprop_function = backprop_function
        self.fitness_function = fitness_function
        self.functions = functions
        self.terminals = terminals
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_evaluations = max_evaluations
        self.max_generations = max_generations
        self.max_time = max_time
        self.initialization_max_tree_height = initialization_max_tree_height
        self.max_tree_size = max_tree_size
        self.tournament_size = tournament_size
        self.generations = 0
        # for grid search
        self.dirName = ""
        self.logName = ''
        # gradient descent params
        self.first_generations = first_generations
        self.initialBackprop = initialBackprop
        self.uniform_k = uniform_k
        self.backprop_every_generations = backprop_every_generations
        self.backprop_selection_ratio = backprop_selection_ratio
        self.reset_weights_on_variation = reset_weights_on_variation
        assert backprop_selection_ratio <= 1, "backprop_selection_ratio should be <= 1."

    def __ShouldTerminate(self):
        must_terminate = False
        elapsed_time = time.time() - self.start_time
        if self.max_evaluations > 0 and self.fitness_function.evaluations >= self.max_evaluations:
            must_terminate = True
        elif self.max_generations > 0 and self.generations >= self.max_generations:
            must_terminate = True
        elif self.max_time > 0 and elapsed_time >= self.max_time:
            must_terminate = True

        # if must_terminate:
        #     print('Terminating at\n\t',
		# 		self.generations, 'generations\n\t', self.fitness_function.evaluations,
        #         'evaluations\n\t', np.round(elapsed_time,2), 'seconds')
        return must_terminate

    def getFilename(self, backprop = False, iterationNum = 0):
        basename = "maxtime" + str(self.max_time) + "_pop" + str(self.pop_size) + "_mr" + str(self.mutation_rate) + "_tour" + str(self.tournament_size) + "_maxHeight" + str(self.initialization_max_tree_height) + "_cr" + str(self.crossover_rate) + "_reset" + str(self.reset_weights_on_variation)
        log = ".txt"
        if backprop:
            extension = "__topK" + str(self.backprop_selection_ratio) + "_unK" + str(self.uniform_k) + "_gen" + str(self.backprop_every_generations) + "_lr" + str(self.backprop_function.learning_rate) + "_it" + str(self.backprop_function.iterations) + "_oIt" + str(self.backprop_function.override_iterations) + "_initial" + str(self.initialBackprop) + "_firstGen" + str(self.first_generations)
        else:
            extension = ""
        self.logName = basename + extension + "_" + str(iterationNum) + log
        return self.logName

    def Run(self, applyBackProp = True, iterationNum = 0, dirName = "experiments"):
        self.dirName = dirName

        self.start_time = time.time()

        population = []

        with open(self.dirName + "/" + self.getFilename(applyBackProp, iterationNum), "w+") as fp:

            for i in range( self.pop_size):

                population.append(Variation.GenerateRandomTree( self.functions, self.terminals,
                                                  self.initialization_max_tree_height ) )

                if self.initialBackprop:
                    population[i] = self.backprop_function.Backprop(population[i]) if applyBackProp else population[i]
                self.fitness_function.Evaluate(population[i])

            fp.write("generations_elite-fitness_number-of-evaluations_time\r\n")
            # print ('g:',self.generations,'elite fitness:', np.round(self.fitness_function.elite.fitness,3), ', size:', len(self.fitness_function.elite.GetSubtree()))
            fp.write(str(self.generations) + "_" + str(np.round(self.fitness_function.elite.fitness,3)) + "_" + str(self.fitness_function.evaluations) + "_" + str(time.time() - self.start_time) + "\r\n")

            while not self.__ShouldTerminate():

                O = []

                for i in range(len(population)):
                    variated = False
                    o = deepcopy(population[i])
                    if ( random() < self.crossover_rate ):
                        o = Variation.SubtreeCrossover( o, population[numpy.random.randint(len(population))] )
                        variated = True
                    if ( random() < self.mutation_rate ):
                        o = Variation.SubtreeMutation( o, self.functions, self.terminals, max_height=self.initialization_max_tree_height )
                        variated = True
                    if variated and self.reset_weights_on_variation:
                        for node in o.GetSubtree():
                            node.weights = np.random.normal(size = node.arity * 2)
                    if len(o.GetSubtree()) > self.max_tree_size:
                        del o
                        o = deepcopy( population[i])
                    else:
                        '''
                        Apply backprop to all if uniform_k was not passed, otherwise apply to uniform_k percent.'
                        Apply backprop every generation if backprop_every_generations is not passed, otherwise only do it every x gens
                        '''
                        doBackprop = False
                        if applyBackProp and self.generations % self.backprop_every_generations == 0 and self.generations < self.first_generations:
                            if self.uniform_k == 1 or random() <= self.uniform_k:
                                doBackprop = True
                        o = self.backprop_function.Backprop(o) if doBackprop else o
                        self.fitness_function.Evaluate(o)

                    O.append(o)

                if self.backprop_selection_ratio != 1: # Non-Default: Apparently, we want to select the top k%.
                    if applyBackProp and self.generations % self.backprop_every_generations == 0:
                        population_fitness = np.array([population[curr].fitness for curr in range(len(population))])
                        to_select = int(self.backprop_selection_ratio*len(population)) # Get the top k% fitnessboys
                        # Unsorted, lowest toSelect fitness individuals, in linear time :)
                        top_k_percent = np.argpartition(population_fitness, 3)[:to_select]
                        for curr_top_k in top_k_percent:
                            O[curr_top_k] = self.backprop_function.Backprop(O[curr_top_k], override_iterations = True)
                            self.fitness_function.Evaluate(O[curr_top_k]) # Re-evaluate fitness for coming tournament

                PO = population+O
                population = Selection.TournamentSelect( PO, len(population), tournament_size=self.tournament_size )

                self.generations = self.generations + 1

                # print ('g:',self.generations,'elite fitness:', np.round(self.fitness_function.elite.fitness,3), ', size:', len(self.fitness_function.elite.GetSubtree()))

                fp.write(str(self.generations) + "_" + str(np.round(self.fitness_function.elite.fitness,3)) + "_" + str(self.fitness_function.evaluations) + "_" + str(time.time() - self.start_time) + "\r\n")

        return self.generations, np.round(self.fitness_function.elite.fitness,3), self.fitness_function.evaluations, str(time.time() - self.start_time)
