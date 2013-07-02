import numpy as np

EPSILON = 0.01

class memObj:
    """A wrapper for objects.

        This wrapper is used to store multi-size arrays in a single list by
        backProp.
    """
    def __init__(self, object):
        self.obj = object
        
    def get(self):
        return self.obj
        
    def set(self, object):
        self.obj = object

class NN:

    def setTheta(self,theta):
        """A function to forcefully set the value of the weights instead of 
        the present randomly initialized values.
        
        Args:
            theta: the complete theta matrix as a 1-D array
        """
        self.theta = theta.reshape(-1)
        
    def getTheta(self):
        """A function to obtain the values of the weight matrix.
        
        Returns:
            the complete weight matrix as a 1-D array.
        """
        return self.theta

    def setLayers(self,layers):
        """A function to forcefully set the layers instead of 
        the present randomly initialized values.
        
        Args:
            layers: the complete theta matrix as a 2-D array
                Format: m * 3, where
                m = number of layers
                layers[m,0]=number of inputs to the layer
                layers[m,1]=number of outputs to the layer
                layers[m,3]=first position of the weight in theta
                
        """
        self.layers = layers
        
    def getLayers(self):
        """A function to obtain the values of the weight matrix.
        
        Returns:
            the complete weight matrix as a 1-D array.
        """
        return self.layers

    def getThetaLayer(self, tIn, lPos):
        t = tIn[self.layers[lPos,2]:self.layers[lPos,0]*self.layers[lPos,1]+self.layers[lPos,2]]
        return t.reshape(self.layers[lPos,0],self.layers[lPos,1])

    def __randInit(self,lIn, lOut):
        return ((np.random.rand(lOut,lIn + 1) * 2.0 * EPSILON) - EPSILON).reshape(-1)

    def __init__(self, X, y, l):
        """Initialize the neural network with inputs, outputs and 
        form the base architecture of the network.
    
        Args:
            
            inputs: A 2-dimensional array where each row is an input vector
            and each column an input variable of the vector.
            
                Format : numpy array of shape n * m, where 
                n = number of input data points
                m = number of inputs in the input vector
        
            outputs: A 2-dimensional array where each row is an output 
            vector and each column an output variable of the vector.
            
                Format : numpy array of shape k * n, where 
                k = number of features in the output vector
                n = number of outputs
        
            layers: A 1-dimensional array where each value specifies the
            number of neurons in the layer. This includes the input
            layer, the hidden layers and an output layer. The input
            layer must assume 1 neuron per input.
            
                Format : 1D numpy array, where 
                layers[0] = m = number of inputs
                layers[1...l] = hidden layers with value equal to number of 
                neurons in each layer.
                layers[l+1] = k = number of features in the output vector
    
            Returns:
            NIL.
            """
        self.X = X
        self.y = np.zeros((l[-1],y.size))
        for i in range(0,y.size):
            t = y[i]
            self.y[t,i] = 1
#        print self.y
        self.l = l
        self.layers_count = l.size
        self.theta = self.__randInit(l[0],l[1])
        self.layers = np.array((l[1],l[0]+1,0))
        for i in range(1,l.size-1):
            pos = self.theta.size
            self.theta = np.hstack((self.theta,self.__randInit(l[i], l[i + 1])))
#           print self.theta
            self.layers = np.vstack((self.layers, np.array((l[i+1],l[i]+1,pos))))
#           print self.layers
        
    def __str__(self):
        return 'Inputs (%d,%d)\nOutputs (%d,%d)\nlayers %s' %(self.X.shape[0],self.X.shape[1],self.y.shape[0],self.y.shape[1],str(self.l))
