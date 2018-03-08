#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import normalize
import sklearn
import sklearn.datasets
import sklearn.linear_model

%matplotlib inline

np.random.seed(1)
#load dataset
dataset = pd.read_csv("/Users/samar/Documents/pythonwd/special-potato/AXISBANK.csv")


#Clean dataset
#adding t1 value or the next day value
dataset["t1"] = dataset["4. close"].shift(-1)


#removing last NaN - because nobody likes naan! 
dataset = dataset[0:(dataset.shape[0]-1)]


#adding p1 as change between t1 and t0
dataset["p1"] = (dataset["t1"] - dataset["4. close"]) 


#changing p1 to 1 if change was +ve and 0 if - ve 
dataset["p1"] = np.where(dataset["p1"] > 0, 1,0)


#cleaning out dates and dataframe
X = dataset[["1. open","2. high", "3. low" ,"4. close", "5. volume"]]

def normalize(df):
    G = preprocessing.StandardScaler().fit(df)
    ndf_mean = df- G.mean_
    ndf = ndf_mean/G.var_
    return ndf
    
X = normalize(X)

split = (int(X.shape[0]*(2/3)))
train_set_x = X[:split]
train_set_x = train_set_x.transpose()
train_set_x = train_set_x.values


test_set_x = X[split:]
test_set_x = test_set_x.transpose()
test_set_x = test_set_x.values

# similarly making y matrix with (1, m) size
Y = dataset[["p1"]]

train_set_y = Y[:split]
train_set_y = train_set_y.transpose()
train_set_y = train_set_y.values


test_set_y = Y[split:]
test_set_y = test_set_y.transpose()
test_set_y = test_set_y.values



np.shape(test_set_y)

train_set_y2 = column_or_1d(train_set_y, warn=True)

clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(train_set_x.T, train_set_y.T);

# Print accuracy
LR_predictions = clf.predict(test_set_x.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(test_set_y, LR_predictions) + np.dot(1 - test_set_y,1 - LR_predictions)) / float(test_set_y.size) * 100) +
       '% ' + "(percentage of correctly labelled datapoints)")

X = train_set_x
Y = train_set_y

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

#layer size function
#X -- input dataset of shape (input size, number of examples)
#Y -- labels of shape (output size, number of examples)

def layer_sizes(X,Y,H= 4):
    #size of input layer
    n_x = X.shape[0]
    #siz of output layer
    n_y = Y.shape[0]
    #size of hidden layer
    n_h= H
    return(n_x,n_h,n_y)
    
#layer_sizes(train_set_x,test_set_y)

#initializing wiehts with random

    
def initialize_parameters(n_x,n_h,n_y):
    #W1 -- weight matrix of shape (wieghts of layer, wieghts of input)
    #b1 -- bias vector of shape (n_h, 1)
    #W2 -- weight matrix of shape (n_y, n_h)
    #b2 -- bias vector of shape (n_y, 1)
    #random initialize wieghts
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros((n_y,1))
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters
    
n_x, n_h, n_y = layer_sizes(train_set_x,test_set_y)
parameters = initialize_parameters(n_x,n_h,n_y)   

    


def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    #parameter in
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    assert(A2.shape == (1, X.shape[1]))
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

A2, cache = forward_propagation(train_set_x, parameters)

def compute_cost(A2,Y, parameters):
    m = Y.shape[1]
    #number of examples
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    #wieghts in mofu!
    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
    cost = - np.sum(logprobs) / m
    #calculate cost 
    cost = np.squeeze(cost)
    assert(isinstance(cost, float))
    # make sure dimension of cost are right
    return cost
    

compute_cost(A2,train_set_y, parameters)

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    #number of examples
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    #get wieghts
    A1 = cache["A1"]
    A2 = cache["A2"]
    #get predictions
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    return grads

grads = backward_propagation(parameters, cache, train_set_x, train_set_y)


def update_parameters(parameters, grads, learning_rate=1.2):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters
    
    
    
update_parameters(parameters, grads, learning_rate=1.2)


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=True):
    np.random.seed(5)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        compute_cost(A2,Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate=1.2)
    
    if i % 100 == 0:
        print ("Cost after iteration %i: %f" % (i, cost))
    
    return parameters

    
    
    
    
parameters = nn_model(train_set_x,train_set_y,8)    

def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)
    return predictions    
    
    

predictions = predict(parameters, train_set_x)
print("predictions mean = " + str(np.mean(predictions)))    
    

# Build a model with a n_h-dimensional hidden layer
parameters = nn_model(test_set_x, train_set_y, n_h = 10, num_iterations=10000, print_cost=True)

predictions = predict(parameters, train_set_x)
print ('Accuracy: %d' % float((np.dot(train_set_y, predictions.T) + np.dot(1 - train_set_y, 1 - predictions.T)) / float(train_set_y.size) * 100) + '%')

    
    
np.size(predictions)
    
    
    
