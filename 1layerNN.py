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
    
#X = normalize(X)

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

plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")

# Print accuracy
LR_predictions = clf.predict(test_set_x.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(test_set_y, LR_predictions) + np.dot(1 - test_set_y,1 - LR_predictions)) / float(test_set_y.size) * 100) +
       '% ' + "(percentage of correctly labelled datapoints)")


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
    w1 = np.random.randn(n_h, n_x)
    b1 = np.zeros((n_h,1))
    w2 = np.random.randn(n_y,n_h)
    b2 = np.zeros((n_y,1))
    parameters = {"w1": w1,
                  "b1": b1,
                  "w2": w2,
                  "b2": b2}
    return parameters
    
n_x, n_h, n_y = layer_sizes(train_set_x,test_set_y)
parameters = initialize_parameters(n_x,n_h,n_y)   

    
#forward prop


def forward_propagation(X, parameters):
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    #parameter in
    z1 = np.dot(w1,X) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2,a1) + b2
    a2 = sigmoid(z2)
    cache = {"z1": z1, "a1": a1, "z2": z2, "a2": a2}
    return a2, cache

a2, cache = forward_propagation(train_set_x, parameters)

def compute_cost(a2,Y, parameters):
    m = Y.shape[1]
    #number of examples
    w1 = parameters["w1"]
    w2 = parameters["w2"]
    #wieghts in mofu!
    logprobs = np.multiply(np.log(a2), Y) + np.multiply((1 - Y), np.log(1 - a2))
    cost = - np.sum(logprobs) / m
    #calculate cost 
    cost = np.squeeze(cost)
    # make sure dimension of cost are right
    return cost
    

compute_cost(a2,train_set_y, parameters)

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    #number of examples
    w1 = parameters["w1"]
    w2 = parameters["w2"]
    #get wieghts
    a1 = cache["a1"]
    a2 = cache["a2"]
    #get predictions
    dz2 = a2 - Y
    dw2 = (1/m) * np.dot(dz2,a1.T)
    db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.multiply(np.dot(w2.T,dz2), 1 - np.power(a1, 2))
    dw1 = (1/m) * np.dot(dz1, X.T)
    db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)    
    grads = {"dw1": dw1,
             "db1": db1,
             "dw2": dw2,
             "db2": db2}
    return grads

grads = backward_propagation(parameters, cache, train_set_x, train_set_y)


def update_parameters(parameters, grads, learning_rate=1.2):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    dw1 = grads['dw1']
    db1 = grads['db1']
    dw2 = grads['dw2']
    db2 = grads['db2']
    w1 = w1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1
    w2 = w2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2
    parameters = {"w1": w1,
                  "b1": b1,
                  "w2": w2,
                  "b2": b2}
    return parameters
    
    
    
update_parameters(parameters, grads, learning_rate=1.2)


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=True):
    np.random.seed(5)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    parameters = initialize_parameters(n_x, n_h, n_y)
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    for i in range(0, num_iterations):
        a2, cache = forward_propagation(train_set_x, parameters)
        compute_cost(a2,train_set_y, parameters)
        grads = backward_propagation(parameters, cache, train_set_x, train_set_y)
        parameters = update_parameters(parameters, grads, learning_rate=1.2)
    
    if i % 1000 == 0:
        print ("Cost after iteration %i: %f" % (i, cost))
    
    return parameters

    
    
    
    
parameters = nn_model(train_set_x,train_set_y,8)    

def predict(parameters, X):
    a2, cache = forward_propagation(X, parameters)
    predictions = np.round(a2)
    return predictions    
    
    

predictions = predict(parameters, train_set_x)
print("predictions mean = " + str(np.mean(predictions)))    
    

# Build a model with a n_h-dimensional hidden layer
parameters = nn_model(test_set_x, train_set_y, n_h = 10, num_iterations=10000, print_cost=True)

predictions = predict(parameters, train_set_x)
print ('Accuracy: %d' % float((np.dot(train_set_y, predictions.T) + np.dot(1 - train_set_y, 1 - predictions.T)) / float(train_set_y.size) * 100) + '%')

    
    
np.size(predictions)
    
    
    
