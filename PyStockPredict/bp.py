"""
Backpropagation
===============

Provides
  1. An implementation of backpropgation algorithm to train 
  and run a neural network
  
"""


import containers as cont
import numpy as np
import scipy.optimize as sopt
import scipy.io as sio
import time 

__author__ = "Chinmay Hundekari"
__version__ = "0.0.1"
__status__ = "Development"


        

def time_end(startTime):
    return (time.clock() - startTime)
        
def sigmoid(z):
    g = 1 /(1 + np.exp(-z))
    return g

def memArray(obj,size):
    objArr = []
    for i in range(0,size):
        objArr.append(cont.memObj(obj))
    return objArr

def feedFwdBp(nn,X):
    """Executes the initialized neural network based on the existing values
    of theta
    
    Args:
        X: input vectors for which the neural network prediction 
        has to be calculated.
        
            Format : numpy array of shape n * m, where 
            n = number of inputs
            m = number of inputs in the input vector
        
    Returns:
        Values predicted by the neural network.
        
            Format: numpy array of shape n * k, where
            k = number of output features to be predicted.
    """
    Xiter = X
    mOnes = np.ones((Xiter.shape[0],1))
    for i in range(0, nn.l.size-1):
        Xiter = sigmoid(np.dot(np.hstack((mOnes,Xiter)),nn.getThetaLayer(nn.theta,i).T))
    Xiter = Xiter.argmax(axis=1)
    return Xiter
    
def costGradient(t,nn,lamda):
    z = []
    a = []
    m = nn.X.shape[0]
    l_size = nn.l.size
    nOnes = np.ones((m,1))
    m_a = cont.memObj(np.hstack((nOnes,nn.X)))
    a.append(m_a)
    z.append(0)

    for i in range(1,l_size):
        tL = nn.getThetaLayer(t,i-1)
        a_i_1 = a[i-1].get()
        m_z = cont.memObj(np.dot(a_i_1,tL.T))
        z.append(m_z)
        nOnes = np.ones((a_i_1.shape[0],1))
        m_z = z[i].get()
        sig_m_z = sigmoid(m_z)
        m_a = cont.memObj(np.hstack((nOnes,sig_m_z)))
        a.append(m_a)
    h = a[l_size-1].get()[:,1:]

    delta = memArray(0,l_size)
    D = memArray(0,l_size-1)
    m_delta = h - nn.y.T
    delta[l_size-1].set(m_delta)
    m_D = np.dot(a[l_size-2].get().T,delta[l_size-1].get())
    D[l_size-2].set(m_D)

    for i in range(l_size-3,-1,-1):
        tL = nn.getThetaLayer(t,i+1)
        m_delta = np.dot(delta[i+2].get(), tL) * (a[i+1].get() * (1 - a[i+1].get()))
        m_delta = m_delta[:,1:]
        delta[i+1].set(m_delta)
        m_D = np.dot(a[i].get().T, delta[i+1].get())
        D[i].set(m_D)
    
    tL = nn.getThetaLayer(t,0)
    treg = lamda/m * np.hstack((np.zeros((tL.shape[0],1)),tL[:,1:] ))
    grad = ((D[0].get() + treg.T)/m)
    grad = grad.reshape(-1, order='F')
    for i in range(1, l_size-1):
        tL = nn.getThetaLayer(t,i)
        treg = lamda/m * np.hstack((np.zeros((tL.shape[0],1)),tL[:,1:] ))
        grad = np.hstack((grad, ((D[i].get()/m) + treg.T).reshape(-1, order='F')))

    return grad
    
def costFunction(t,nn,lamda):
    z = []
    a = []
    m = nn.X.shape[0]
    l_size = nn.l.size
    nOnes = np.ones((m,1))
    m_a = cont.memObj(np.hstack((nOnes,nn.X)))
    a.append(m_a)
    z.append(0)
    for i in range(1,l_size):
        tL = nn.getThetaLayer(t,i-1)
        a_i_1 = a[i-1].get()
        m_z = cont.memObj(np.dot(a_i_1,tL.T))
        z.append(m_z)
        nOnes = np.ones((a_i_1.shape[0],1))
        m_z = z[i].get()
        sig_m_z = sigmoid(m_z)
        m_a = cont.memObj(np.hstack((nOnes,sig_m_z)))
        a.append(m_a)
    h = a[l_size-1].get()[:,1:]
    temp1 = (1.0 - nn.y).T * np.log(1.0 - h)
    temp2 = -1.0 * nn.y.T * np.log(h)
    temp3 = temp2 - temp1 
    J_unreg = 1.0/m * np.sum(np.sum(temp3, axis=1))
    
    J_reg = 0
    for i in range(0,l_size-1):
        theta_reg = nn.getThetaLayer(t,0) ** 2
        theta_reg[:,0] = 0
        J_reg = J_reg + (lamda / (2.0 * m)) * np.sum(np.sum(theta_reg))

    J = J_unreg+J_reg    
    return J

def trainBp(nn, lamda, maxIter):
    """Train the initilized neural network by backpropogation. 

    Args:            
        nn: A neural network of type NN.
        
        lamda: Regularisation parameter.
                    
        maxIter: Maximum iterations before stopping training.

    Returns:
        Final cost of the function.

    """
    t = nn.theta
    startTime = time.clock()
    nn.theta = sopt.fmin_cg(costFunction,t,costGradient,(nn,lamda),maxiter=maxIter)
    endTime = time_end(startTime)
    J = costFunction(nn.theta,nn,lamda)
    print "Time taken to train the network is ",endTime,"sec. with the final cost at ", J
    return J

def analytics(bpNet,inputs,outputs):
    feedFwd = np.around(feedFwdBp(bpNet,inputs)).astype(int)
    for i in range(0,np.max(outputs) + 1):
        truePos = np.sum(feedFwd[outputs == i] == i) * 1.0
        totVal  = np.sum(outputs == i) * 1.0
        print 'Class %d :%r'%(i, truePos / totVal * 100.0)
    accuracy = np.mean(feedFwd == outputs) * 100.0
    print 'Accuracy: %r' %(accuracy)

def boolAnalytics(bpNet,inputs,outputs):
    feedFwd = feedFwdBp(bpNet,inputs) > 0.5
    print np.sum(feedFwd[outputs])
    truePos = np.sum(feedFwd[outputs]) *1.0
    allpos = np.sum(outputs) *1.0
    recall = truePos/allpos * 100.0
    predpos = np.sum(feedFwd) *1.0
    precision = truePos/predpos * 100.0
    accuracy = np.mean(feedFwd == outputs) * 100.0
    
    
    print 'Accuracy : %r' %(accuracy)
    print 'Recall   : %r' %(recall)
    print 'Precision: %r' %(precision)

def bpTrain():
    data = sio.loadmat('../data/bollinger_inputs.mat')
    X = data['X']
    y = data['y'].reshape(-1).astype(int)
    L = np.array(([X.shape[1],5,5,np.max(y)+1]))
    w = cont.NN(X,y,L)
    j = trainBp(w,1,100)
    t = w.getTheta()
    sio.savemat('../data/bollinger_weights.mat',{'t':t,'L':L})

def bpExec(X,):
    data = sio.loadmat('../data/bollinger_weights.mat')
    t = data['t'].reshape(-1)
    L = data['L']
    y = np.zeros((X.shape[0]))
    w = cont.NN(X,y,L)
    w.setTheta(t)
    y = feedFwdBp(w,X)
    return y
     
