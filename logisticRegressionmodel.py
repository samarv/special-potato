#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import normalize

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
#defining sigmoid function
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim, 1))
    b = 0
    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)  # compute activation
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # compute cost
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    cost = np.squeeze(cost)
    grads = {"dw": dw,
             "db": db}
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - (learning_rate * dw)  # need to broadcast
        b = b - (learning_rate * db)
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    return d
    
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

