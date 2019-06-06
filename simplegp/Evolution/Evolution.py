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
                 backprop_every_generations = 1
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
        self.uniform_k = 1
        self.backprop_every_generations = 1
        
    def __ShouldTerminate(self):
        must_terminate = False
        elapsed_time = time.time() - self.start_time
        if self.max_evaluations > 0 and self.fitness_function.evaluations >= self.max_evaluations:
            must_terminate = True
        elif self.max_generations > 0 and self.generations >= self.max_generations:
            must_terminate = True
        elif self.max_time > 0 and elapsed_time >= self.max_time:
            must_terminate = True
        
        if must_terminate:
            print('Terminating at\n\t',
				self.generations, 'generations\n\t', self.fitness_function.evaluations, 
                'evaluations\n\t', np.round(elapsed_time,2), 'seconds')
        return must_terminate

    def getFilename(self, run, backprop = False, iterationNum = 0):
        basename = "maxtime" + str(run.max_time) + "_pop" + str(run.pop_size) + "_mr" + str(run.mutation_rate) + "_tour" + str(run.tournament_size) + "_maxHeight" + str(run.initialization_max_tree_height) + "_cr" + str(run.crossover_rate) 
        log = ".txt"
		# if backprop:
		# 	# extension = "random" + str(run.random_k) + "top" + str(run.top_k) + "bpeverygen" + str(run.backprop_every_generations) + "lr" + str(run.learning_rate) + "toplr" + str(run.top_k_learning_rate)
		# 	# return basename + extension + log
		# else:
        self.logName = basename + "_" + str(iterationNum) + log
        return self.logName

    def Run(self, applyBackProp = True, iterationNum = 0):
		# Create target Directory if don't exist
        self.dirName = "experiments"

        if not os.path.exists(self.dirName):
            os.mkdir(self.dirName)
            print("Directory " , self.dirName ,  " Created ")
		
        self.start_time = time.time()

        population = []
        
        with open(self.dirName + "/" + self.getFilename(self, applyBackProp, iterationNum), "w+") as fp:
            
            for i in range( self.pop_size):
            
                population.append(Variation.GenerateRandomTree( self.functions, self.terminals, 
                                                  self.initialization_max_tree_height ) )
                '''
                population[i] = self.backprop_function.Backprop(population[i]) if applyBackProp else population[i]
                '''
                self.fitness_function.Evaluate(population[i])
            
            fp.write("generations_elite-fitness_number-of-evaluations_time\r\n")
            print ('g:',self.generations,'elite fitness:', np.round(self.fitness_function.elite.fitness,3), ', size:', len(self.fitness_function.elite.GetSubtree()))
            fp.write(str(self.generations) + "_" + str(np.round(self.fitness_function.elite.fitness,3)) + "_" + str(self.fitness_function.evaluations) + "_" + str(time.time() - self.start_time) + "\r\n")
                
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
						doBackprop = False
						if applyBackProp and self.generations % self.backprop_every_generations == 0:
							# uniformly randomly choose individuals to backprop
							if self.uniform_k == 1 or random() <= self.uniform_k:
								doBackprop = True
						
						o = self.backprop_function.Backprop(o) if doBackprop else o
                        '''
                        self.fitness_function.Evaluate(o)

                    O.append(o)

                PO = population+O
                population = Selection.TournamentSelect( PO, len(population), tournament_size=self.tournament_size )

                self.generations = self.generations + 1

                print ('g:',self.generations,'elite fitness:', np.round(self.fitness_function.elite.fitness,3), ', size:', len(self.fitness_function.elite.GetSubtree()))

                fp.write(str(self.generations) + "_" + str(np.round(self.fitness_function.elite.fitness,3)) + "_" + str(self.fitness_function.evaluations) + "_" + str(time.time() - self.start_time) + "\r\n")

        return self.generations, np.round(self.fitness_function.elite.fitness,3), self.fitness_function.evaluations, str(time.time() - self.start_time)