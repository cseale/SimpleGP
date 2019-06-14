import numpy as np
import os
from numpy.random import random
import numpy.random
import time
from copy import deepcopy

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
                 max_time=-1,
                 initialization_max_tree_height=4,
                 max_tree_size=100,
                 tournament_size=4,
                 uniform_k=1,
                 backprop_every_generations = 1,
                 backprop_selection_ratio = 1,
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
        self.uniform_k = uniform_k
        self.backprop_every_generations = backprop_every_generations
        self.backprop_selection_ratio = backprop_selection_ratio
        assert backprop_selection_ratio <= 1, "backprop_selection_ratio should be leq 1."

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

    def getFilename(self, run, backprop = False, iterationNum = 0):
        basename = "maxtime" + str(run.max_time) + "_pop" + str(run.pop_size) + "_mr" + str(run.mutation_rate) + "_tour" + str(run.tournament_size) + "_maxHeight" + str(run.initialization_max_tree_height) + "_cr" + str(run.crossover_rate)
        log = ".txt"
        extension = ""
		
        if self.backprop_every_generations != 1:
            extension = extension + "_bpeverygen" + str(self.backprop_every_generations) 
            
        if self.uniform_k != 1:
            extension = extension + "_uniform" + str(self.uniform_k)
            
        if self.backprop_selection_ratio != 1:
            extension = extension + "_bpratio" + str(self.backprop_selection_ratio)
        
        self.logName = basename + extension + "_" + str(iterationNum) + log
        return self.logName

    def Run(self, applyBackProp = True, iterationNum = 0):
		# Create target Directory if don't exist
        self.dirName = "experiments"

        self.start_time = time.time()

        population = []

        with open(self.dirName + "/" + self.getFilename(self, applyBackProp, iterationNum), "w+") as fp:

            for i in range( self.pop_size):

                population.append(Variation.GenerateRandomTree( self.functions, self.terminals,
                                                  self.initialization_max_tree_height ) )

                population[i] = self.backprop_function.Backprop(population[i]) if applyBackProp else population[i]
                self.fitness_function.Evaluate(population[i])

            fp.write("generation, individual, diff_after_backprop\r\n")

            while not self.__ShouldTerminate():

                O = []

                for i in range(len(population)):

                    o = deepcopy(population[i])
                    if ( random() < self.crossover_rate ):
                        o = Variation.SubtreeCrossover( o, population[numpy.random.randint(len(population))] )
                    if ( random() < self.mutation_rate ):
                        o = Variation.SubtreeMutation( o, self.functions, self.terminals, max_height=self.initialization_max_tree_height )
                    if len(o.GetSubtree()) > self.max_tree_size:
                        del o
                        o = deepcopy( population[i])
                    else:
                        '''
                        Apply backprop to all if uniform_k was not passed, otherwise apply to uniform_k percent.'
                        Apply backprop every generation if backprop_every_generations is not passed, otherwise only do it every x gens
                        '''
                        doBackprop = False
                        before = self.fitness_function.Evaluate(o)            
                        if applyBackProp and self.generations % self.backprop_every_generations == 0:
                            if self.uniform_k == 1 or random() <= self.uniform_k:
                                doBackprop = True
                        o = self.backprop_function.Backprop(o) if doBackprop else o
                        after = self.fitness_function.Evaluate(o)
                        fp.write(str(self.generations) + "," + str(i) + "," + str(before - after) + "\r\n")

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

        return self.generations, np.round(self.fitness_function.elite.fitness,3), self.fitness_function.evaluations, str(time.time() - self.start_time)
