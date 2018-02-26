#bois and grills starting it out easy with a logistic regression.

#importing libraries
import numpy as np
import pandas as pd

#load dataset
dataset = pd.read_csv("AXISBANK.csv")


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



#IMP - changing it to (nx,m) size as nx = number of features and m is no. of datapoints
# you have nx features in every single datapoint.
#done because it makes matrix manipulation and mangement - less cringy
X = X.transpose()
X = X.values


# similarly making y matrix with (1, m) size
Y = dataset[["p1"]]
Y = Y.transpose()
Y = Y.values

#defining sigmoid function
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

#initializing W and b. 
#w is (n,1) where n is number of features
w = np.zeros((X.shape[0],1))
b = 0 
m = X.shape[1]
learning_rate = 0.009

#for loop of gradient decent

for i in range(1000):
    #forward prop
    A = sigmoid(np.dot(w.T, X) + b)
    # a metric of our cost
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))
    
    #backward prop derivative of A
    dz = A - Y
    dw = (1/m) * np.dot(X,dz.T)
    db = np.sum(dz)/m
    
    #updating the wirghts
    w = w - (learning_rate * dw)
    b = b - (learning_rate * db)
    print(cost)

    
    
    
    

