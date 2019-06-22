import numpy as np


class Node:	# Base class with general functionalities

	def __init__(self, arity = 0):
		self.fitness = np.inf
		self.parent = None
		self.arity = arity	# arity is the number of expected inputs
		self._children = []
		self.weights = np.random.normal(size = self.arity * 2)
		self.dw = [None] * (self.arity * 2)
		self.X0 = None
		self.X1 = None
		self.backprop_iterations = 0
		self.backprop_improvement = 0
		self.generations_alive = 0
		self.created_by = "init"


	def GetSubtree( self ):
		result = []
		self.__GetSubtreeRecursive(result)
		return result

	def AppendChild( self, N ):
		self._children.append(N)
		N.parent = self

	def DetachChild( self, N ):
		assert(N in self._children)
		for i, c in enumerate(self._children):
			if c == N:
				self._children.pop(i)
				N.parent = None
				break
		return i

	def InsertChildAtPosition( self, i, N ):
		self._children.insert( i, N )
		N.parent = self

	def GetOutput( self, X ):
		return None

	def GradientDescent(self, input_grad, learning_rate):
		return None

	def GetDepth(self):
		n = self
		d = 0
		while (n.parent):
			d = d+1
			n = n.parent
		return d

	def __GetSubtreeRecursive( self, result ):
		result.append(self)
		for c in self._children:
			c.__GetSubtreeRecursive( result )
		return result

