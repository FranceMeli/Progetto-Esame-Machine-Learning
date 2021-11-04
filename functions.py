import pandas as pd
import numpy as np



''' ---------------- Function ---------------- '''
class Function:

    def __init__(self, function, name):
        self.function = function
        self.name = name

        def function(self):
            return self.function




''' ---------------- DerivableFunction ---------------- '''
class DerivableFunction:

    def __init__(self, function, derivative, name):
        self.function = function
        self.derivative = derivative
        self.name = name

    def function(self):
        return self.function

    def derivative(self):
        return self.derivative




''' ---------- Activation functions ---------- '''
def idn_f(x):
    return x
def idn_deriv(x):
    return np.ones(x.shape)

identity = DerivableFunction(idn_f, idn_deriv, 'identity')


def relu_f(x):
    return np.maximum(0, x)

def relu_deriv(x):
    for i in range(len(x)):
        if(x[i]>0):
            x[i] = 1
        else:
            x[i] = 0
    return x

relu = DerivableFunction(relu_f, relu_deriv, 'relu')


def leaky_relu_f(x):
    for i in range(len(x)):
        if(x[i]>=0):
            x[i] = x[i]
        else:
            x[i] = x[i]*0.01
    return x

def leaky_relu_deriv(x):
    for i in range(len(x)):
        if(x[i]>=0):
            x[i] = 1
        else:
            x[i] = x[i]*0.01
    return x

leaky_relu = DerivableFunction(leaky_relu_f, leaky_relu_deriv, 'leaky_relu')


def sigm_f(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x):
    f = 1 / (1 + np.exp(-x))
    return f * (1-f)

sigmoid = DerivableFunction(sigm_f, sigmoid_deriv, 'sigmoid')


def tanh_f(x):
    return np.tanh(x)
def tanh_deriv(x):
    return 1 - (np.tanh(x)) **2 #(1/ (np.exp(x) + np.exp(-x))/2) **2

tanh = DerivableFunction(tanh_f, tanh_deriv, 'tanh')


def softplus_f(x):
    return np.log(1 + np.exp(x))
def softplus_deriv(x):
    return 1 / (1 + np.exp(-x))
     ## perchè mettere np.diag ???

softplus = DerivableFunction(softplus_f, softplus_deriv, 'softplus')




''' ---------- Loss functions ---------- '''

def squaredLoss_f(o, t):
    return 0.5 * np.square(np.subtract(t, o))
def squaredLoss_deriv(o, t):
    return np.subtract(o, t)

squaredloss = DerivableFunction(squaredLoss_f, squaredLoss_deriv, 'MSE')


def euclideanLoss_f(o, t):
    return np.sqrt(np.square(np.subtract(t, o)))
def euclideanLoss_deriv(o, t):
    return (o - t) / np.sqrt(np.square(np.subtract(t, o)))
    ## controllare se va bene la derivata sopratturro (o-t) / [...]

euclideanLoss = DerivableFunction(euclideanLoss_f, euclideanLoss_deriv, 'MEE')





''' ---------- Accuracy functions ---------- '''

def binaryAccuracy_f(o, t):
    if np.abs(o - t) < 0.3: # 0.3 + la nostra soglia ma vedere se ci piace
        return 1
    else:
        return 0

binaryAccuracy = Function(binaryAccuracy_f, 'binaryAccuary')

def multipleAccuracy_f(o, t):
    if np.argmax(o) == np.argmax(t):
        return 1
    else:
        return 0

multipleAccuracy = Function(multipleAccuracy_f, 'multipleAccuracy')
