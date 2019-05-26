import numpy as np

from simplegp.Nodes.BaseNode import Node

# numerical stability, ensure no division by zero
epsilon = 0.00001

class AddNode(Node):
	
	def __init__(self):
		super(AddNode,self).__init__(arity=2)

	def __repr__(self):
		return '+'

	def GetOutput( self, X ):
        # Recursively compute output of the children
		X0 = self._children[0].GetOutput( X )
		X1 = self._children[1].GetOutput( X )
        self.X0 = X0
        self.X1 = X1

		# derivatives of node output w.r.t weights
		self.dw[0] = X0
		self.dw[1] = 1
		self.dw[2] = X1
		self.dw[3] = 1
		return X0 + X1
		
	# input_grad is deriative of loss w.r.t output
	# see: https://cdn-images-1.medium.com/max/1600/1*FceBJSJ7j8jHjb4TmLV0Ew.png
	def GradientDescent(self, input_grad, learning_rate):
		self._children[0].GradientDescent(self.weights[0] * input_grad, learning_rate)
		self._children[1].GradientDescent(self.weights[2] * input_grad, learning_rate)
		
		# derivatives of Loss w.r.t input
		dw0 = np.mean(self.dw[0] * input_grad)
		dw1 = np.mean(self.dw[1] * input_grad)
		dw2 = np.mean(self.dw[2] * input_grad)
		dw3 = np.mean(self.dw[3] * input_grad)

		# update weights in direction opposite to gradients
		self.weights[0] = self.weights[0] - learning_rate * dw0
		self.weights[1] = self.weights[1] - learning_rate * dw1
		self.weights[2] = self.weights[2] - learning_rate * dw2
		self.weights[3] = self.weights[3] - learning_rate * dw3
		


class SubNode(Node):
	def __init__(self):
		super(SubNode,self).__init__(arity=2)

	def __repr__(self):
		return '-'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		X1 = self._children[1].GetOutput( X )
		self.dw[0] = X0
		self.dw[1] = 1
		self.dw[2] = -X1
		self.dw[3] = -1
		return X0 - X1

	def GradientDescent(self, input_grad, learning_rate):
		self._children[0].GradientDescent(self.weights[0] * input_grad, learning_rate)
		self._children[1].GradientDescent(-self.weights[2] * input_grad, learning_rate)
		
		
		dw0 = np.mean(self.dw[0] * input_grad)
		dw1 = np.mean(self.dw[1] * input_grad)
		dw2 = np.mean(self.dw[2] * input_grad)
		dw3 = np.mean(self.dw[3] * input_grad)

		self.weights[0] = self.weights[0] - learning_rate * dw0
		self.weights[1] = self.weights[1] - learning_rate * dw1
		self.weights[2] = self.weights[2] - learning_rate * dw2
		self.weights[3] = self.weights[3] - learning_rate * dw3

class MulNode(Node):
	def __init__(self):
		super(MulNode,self).__init__(arity=2)

	def __repr__(self):
		return '*'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		X1 = self._children[1].GetOutput( X )
		self.dw[0] = X0*self.weights[2]*X1 + X0*self.weights[3]
		self.dw[1] = self.weights[2]*X1 + self.weights[3]
		self.dw[2] = self.weights[1]*X0*X1 + X1*self.weights[1]
		self.dw[3] = self.weights[0]*X0 + self.weights[1]
		self.X0 = X0
		self.X1 = X1
		return np.multiply(X0 , X1)

	def GradientDescent(self, input_grad, learning_rate):
		self._children[0].GradientDescent((self.weights[0]*self.weights[2]*self.X1 + self.weights[0]*self.weights[3]) * input_grad, learning_rate)
		self._children[1].GradientDescent((self.weights[0]*self.weights[2]*self.X0 + self.weights[1]*self.weights[2]) * input_grad, learning_rate)
		
		
		dw0 = np.mean(self.dw[0] * input_grad)
		dw1 = np.mean(self.dw[1] * input_grad)
		dw2 = np.mean(self.dw[2] * input_grad)
		dw3 = np.mean(self.dw[3] * input_grad)

		self.weights[0] = self.weights[0] - learning_rate * dw0
		self.weights[1] = self.weights[1] - learning_rate * dw1
		self.weights[2] = self.weights[2] - learning_rate * dw2
		self.weights[3] = self.weights[3] - learning_rate * dw3
	
class DivNode(Node):
	def __init__(self):
		super(DivNode,self).__init__(arity=2)

	def __repr__(self):
		return '/'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		X1 = self._children[1].GetOutput( X )
		self.tmp1 = 1/(self.weights[2]*X1 + self.weights[3] + epsilon)
		self.dw[0] = X0 * self.tmp1
		self.dw[1] = self.tmp1
		self.tmp2 = -(self.weights[0]*X0 + self.weights[1]) * self.tmp1**2
		self.dw[2] = self.tmp2 * X1
		self.dw[3] = self.tmp2

		return np.multiply( np.sign(X1), X0) / ( 1e-2 + np.abs(X1) )
	
	def GradientDescent(self, input_grad, learning_rate):
		self._children[0].GradientDescent(self.tmp1 * self.weights[0] * input_grad, learning_rate)
		self._children[1].GradientDescent(self.tmp2 * self.weights[2] * input_grad, learning_rate)
		
		dw0 = np.mean(self.dw[0] * input_grad)
		dw1 = np.mean(self.dw[1] * input_grad)
		dw2 = np.mean(self.dw[2] * input_grad)
		dw3 = np.mean(self.dw[3] * input_grad)

		self.weights[0] = self.weights[0] - learning_rate * dw0
		self.weights[1] = self.weights[1] - learning_rate * dw1
		self.weights[2] = self.weights[2] - learning_rate * dw2
		self.weights[3] = self.weights[3] - learning_rate * dw3

class AnalyticQuotientNode(Node):
	def __init__(self):
		super(AnalyticQuotientNode,self).__init__(arity=2)

	def __repr__(self):
		return 'aq'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		X1 = self._children[1].GetOutput( X )
		return X0 / np.sqrt( 1 + np.square(X1) )

	
class ExpNode(Node):
	def __init__(self):
		super(ExpNode,self).__init__(arity=1)

	def __repr__(self):
		return 'exp'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		return np.exp(X0)


class LogNode(Node):
	def __init__(self):
		super(LogNode,self).__init__(arity=1)

	def __repr__(self):
		return 'log'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		return np.log( np.abs(X0) + 1e-2 )


class SinNode(Node):
	def __init__(self):
		super(SinNode,self).__init__(arity=1)

	def __repr__(self):
		return 'sin'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		tmp = np.cos(self.weights[0]*X0 + self.weights[1])
		self.dw[0] = tmp*X0
		self.dw[1] = tmp
		self.X0 = X0
		return np.sin(X0)

	def GradientDescent(self, input_grad, learning_rate):
		self._children[0].GradientDescent((np.cos(self.weights[0]  * self.X0 + self.weights[1])*self.weights[0]) * input_grad, learning_rate)
		
		dw0 = np.mean(self.dw[0] * input_grad)
		dw1 = np.mean(self.dw[1] * input_grad)

		self.weights[0] = self.weights[0] - learning_rate * dw0
		self.weights[1] = self.weights[1] - learning_rate * dw1

class CosNode(Node):
	def __init__(self):
		super(CosNode,self).__init__(arity=1)

	def __repr__(self):
		return 'cos'

	def GetOutput( self, X ):
		X0 = self._children[0].GetOutput( X )
		tmp = -np.sin(self.weights[0]*X0 + self.weights[1])
		self.dw[0] = tmp*X0
		self.dw[1] = tmp
		self.X0 = X0
		return np.cos(X0)

	def GradientDescent(self, input_grad, learning_rate):
		self._children[0].GradientDescent((-np.sin(self.weights[0]  * self.X0 + self.weights[1])*self.weights[0]) * input_grad, learning_rate)
		
		dw0 = np.mean(self.dw[0] * input_grad)
		dw1 = np.mean(self.dw[1] * input_grad)

		self.weights[0] = self.weights[0] - learning_rate * dw0
		self.weights[1] = self.weights[1] - learning_rate * dw1

class FeatureNode(Node):
	def __init__(self, id):
		super(FeatureNode,self).__init__()
		self.id = id

	def __repr__(self):
		return 'x'+str(self.id)

	def GetOutput(self, X):
		return X[:,self.id]

	
class EphemeralRandomConstantNode(Node):
	def __init__(self):
		super(EphemeralRandomConstantNode,self).__init__()
		self.c = np.nan

	def __Instantiate(self):
		self.c = np.round( np.random.random() * 10 - 5, 3 )

	def __repr__(self):
		if np.isnan(self.c):
			self.__Instantiate()
		return str(self.c)

	def GetOutput(self,X):
		if np.isnan(self.c):
			self.__Instantiate()
		return np.array([self.c] * X.shape[0])
