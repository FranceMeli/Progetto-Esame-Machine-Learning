import pandas as pd
import numpy as np
import math
import copy
import random
from numpy.linalg import norm #forse non serve più

from functions import *




''' ---------------- Layer ---------------- '''
class Layer:

    def __init__(self, input_dimension, output_dimension, activation, regularization, initialization):
        if (initialization == 'random'):
            self.w = np.random.randn(output_dimension, input_dimension)
        elif (initialization == 'range2'):
            self.w = np.random.uniform(low=-0.2, high=0.2, size=(output_dimension, input_dimension))
        elif (initialization == 'range7'):
            self.w = np.random.uniform(low=-0.7, high=0.7, size=(output_dimension, input_dimension))
        elif (initialization == 'range6'):
            self.w = np.random.uniform(low=-0.6, high=0.6, size=(output_dimension, input_dimension))
        elif (initialization == 'range4'):
            self.w = np.random.uniform(low=-0.4, high=0.4, size=(output_dimension, input_dimension))
        elif (initialization == 'xavier'):
            self.w = np.random.uniform(low=-1/math.sqrt(input_dimension), high=1/math.sqrt(input_dimension), size=(output_dimension, input_dimension))
        elif (initialization == 'normalized_xavier'):
            self.w = np.random.uniform(low=-math.sqrt(6)/math.sqrt(input_dimension+output_dimension), high=math.sqrt(6)/math.sqrt(input_dimension+output_dimension), size=(output_dimension, input_dimension))
        elif (initialization == 'he'): #relu
            self.w = np.random.normal(loc=0, scale=math.sqrt(2/input_dimension), size=(output_dimension, input_dimension))
        self.bias = np.random.randn(output_dimension)
        self.activation = activation
        self.regularization = regularization


    def feed_forward(self, x):
        self.x = x
        self.z = np.dot(self.w, self.x) + self.bias
        return self.activation.function(self.z)


    def propagate_back(self, current_delta):
        derivative_z = self.activation.derivative(self.z)
        self.delta = np.multiply(current_delta, derivative_z)
        step = np.dot(self.delta, self.w)
        return step


    def compute_gradients(self):
        self.gradient_bias = -self.delta
        self.gradient_weights = np.outer(self.delta, self.x)
        return self.gradient_weights, self.gradient_bias




''' ---------------- Network ---------------- '''
class Network:

    def __init__(self, layers_size, activation_function, last_activation_function, regularization=0.01, initialization_w='random', learning_rate=0.1, loss_function = squaredloss, max_epochs=10, momentum=0.6):
        self.layers_size = layers_size
        self.activation_function = activation_function
        self.last_activation_function = last_activation_function
        self.initialization_w = initialization_w
        self.layers = []
        for i in range(0, len(layers_size)-2):
            layer = Layer(layers_size[i], layers_size[i+1], activation_function, regularization, initialization_w)
            self.layers.append(layer)
        layer = Layer(layers_size[-2], layers_size[-1], last_activation_function, regularization, initialization_w)
        self.layers.append(layer)
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.max_epochs = max_epochs
        self.momentum = momentum


    def get_network_description(self):
        parameters = {}
        parameters['layers_size'] =  self.layers_size
        parameters['activation_function'] = self.activation_function.name
        parameters['last_activation_function'] = self.last_activation_function.name
        parameters['initialization_w'] = self.initialization_w
        parameters['regularization'] = self.regularization
        parameters['learning_rate'] = self.learning_rate
        parameters['loss_function'] = self.loss_function.name
        parameters['momentum'] = self.momentum
        parameters['max_epochs'] = self.max_epochs
        #return a dictionary with all the parameters of the network
        return parameters

    def compile(self):
        network_params = self.get_network_description()
        net = Network(
            layers_size = network_params['layers_size'],
            activation_function = self.activation_function,
            last_activation_function = self.last_activation_function,
            initialization_w = network_params['initialization_w'],
            regularization = network_params['regularization'],
            learning_rate = network_params['learning_rate'],
            loss_function = self.loss_function,
            momentum = network_params['momentum'],
            max_epochs = network_params['max_epochs']
        )
        return net


    def get_null_copy(self):
            copied = copy.deepcopy(self)
            for layer in copied.layers:
                layer.w.fill(0)
                layer.bias.fill(0)
            return copied


    def feed_forward(self, x):
        for layer in self.layers:
            x = layer.feed_forward(x)
        return x


    def back_propagation(self, diff):
        for layer in reversed(self.layers):
            diff = layer.propagate_back(diff)


    def fit(self, train, val, batch, acc = True):
        losses_train = []
        accuracies_train = []
        losses_val = []
        accuracies_val = []

        net_copy = self.get_null_copy()

        for epoch in range(self.max_epochs):
            random.shuffle(train)
            error_at_epoch = []
            #cycle for the batch size
            for b in range (int(math.ceil(len(train)/batch))):
                grad_net = self.get_null_copy()

                for x, target in train[b*batch : (b+1)*batch]:
                    #print(x, target)
                    out = self.feed_forward(x)
                    #print(out)
                    #print()
                    if (epoch == -80):
                        print(out)
                    error_deriv = self.loss_function.derivative(out, target)

                    self.back_propagation(error_deriv)
                    # update gradients
                    for j, layer in enumerate(self.layers):
                        gradient_w, gradient_bias = layer.compute_gradients()
                        grad_net.layers[j].w += gradient_w - 2*self.regularization * net_copy.layers[j].w
                        grad_net.layers[j].bias += gradient_bias - 2*self.regularization * net_copy.layers[j].bias
                        #grad_net.layers[j].w += gradient_w - self.regularization * np.abs(net_copy.layers[j].w)
                        #grad_net.layers[j].bias += gradient_bias - self.regularization * np.abs(net_copy.layers[j].bias)
                #apply the momentum and then the delta rule
                for j, layer in enumerate(self.layers):
                    grad_net.layers[j].w /= batch
                    grad_net.layers[j].bias /= batch

                    net_copy.layers[j].w = self.momentum * net_copy.layers[j].w + (1 - self.momentum) * grad_net.layers[j].w
                    net_copy.layers[j].bias = self.momentum * net_copy.layers[j].bias + (1 - self.momentum) * grad_net.layers[j].bias

                    self.layers[j].w -= self.learning_rate * net_copy.layers[j].w
                    self.layers[j].bias -= self.learning_rate * net_copy.layers[j].bias
            losses_train.append(self.avg_loss(train))
            losses_val.append(self.avg_loss(val))
            if(acc):
                accuracies_train.append(self.avg_accuracy(train, binary= True))
                accuracies_val.append(self.avg_accuracy(val, binary= True))
        if (acc):
            return losses_train, losses_val, accuracies_train, accuracies_val
        else:
            return losses_train, losses_val


    def predict(self, data):
        prediction = []
        for i in range(len(data)):
            prediction.append(self.feed_forward(data[i])) #data[i][0]
        return prediction


    def avg_loss(self, data):
        loss = 0.
        for x, y in data:
            result = self.feed_forward(x)
            loss += self.loss_function.function(result, y)
        return loss / len(data)


    def RMSE_loss(self, data): #serve davvero ??
        loss = 0.
        for x, y in data:
            result = self.feed_forward(x)
            loss += (self.loss_function.function(result, y))**2
        return (loss / len(data))**0.5


    def avg_accuracy(self, data, binary):
        acc = 0.
        for x, y in data:
            result = self.feed_forward(x)
            if(binary):
                acc += binaryAccuracy.function(result, y)
            else:
                acc += multipleAccuracy.function(result, y)
        return acc / len(data)
