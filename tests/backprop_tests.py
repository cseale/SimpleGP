import unittest
import numpy as np
from sklearn.model_selection import train_test_split
from simplegp.Nodes.Backpropagation import Backpropagation
from simplegp.Nodes.SymbolicRegressionNodes import AddNode, FeatureNode
from simplegp.Fitness.FitnessFunction import SymbolicRegressionFitness

class BackpropTests(unittest.TestCase):
    ''' Tests Backpropagation for a simple dataset: y = (2x+3) + (4x+6)
        Tree consists of 3 nodes: A plus function with two children, x1 and x2
    '''
    def test_backprop(self):
        X = np.random.randint(0,10,size=(2000,2)) #Generate random 2-dimensional data
        y = [2 * i[0] + 3 + 4*i[1] + 6 for i in X] #Apply function
        backprop = Backpropagation(X, y, learning_rate=0.01, iters=500)
        fitness_evaluator = SymbolicRegressionFitness(X, y)

        add_node = AddNode() # Root node
        x0_node = FeatureNode(0) # Left child
        x1_node = FeatureNode(1) # Right child
        add_node.AppendChild(x0_node) # Set as children
        add_node.AppendChild(x1_node)

        output = add_node.GetOutput(X) # Get output for current weights
        fitnessPrev = np.mean(np.square(y - output)) # Get current fitness
        print(f"Weights before: {add_node.weights}")
        print(f"Fitness before: {fitnessPrev}")
        node_after_prop = backprop.Backprop(add_node) # Apply backprop and repeat
        output = node_after_prop.GetOutput(X)
        fitnessNew = np.mean(np.square(y - output))
        print(f"Weights after: {node_after_prop.weights}")
        print(f"Node fitness after: {fitnessNew}")
        self.assertTrue(fitnessNew < 0.1) # Check if best fitness smaller than 0.1

if __name__ == '__main__':
    unittest.main()
