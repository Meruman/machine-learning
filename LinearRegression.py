# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 00:57:24 2018

@author: Martin Mendez Ruiz
Implementing Gradient decent, Program assignment 1 for Machine Learning
"""
import csv
import numpy as np
import pandas as pd

#Asking for the location of the file
file = input("Please give me the location of the file including .csv: ");

# Importing the dataset
dataset = pd.read_csv(file, header = None)
#dataset = pd.read_csv(r'C:\Users\marti\Documents\mter\clases\ML\Program ass 1\random.csv', header = None)
#Separate the dataset into features(x) and taget(y)
X = dataset.iloc[:, :-1].values;
y = dataset.iloc[:, -1].values;
m = len(y);
X = np.c_[ np.ones(m), X ]; #Gives value of 1 to Xo
m,n = X.shape; #m number of training set, n number of total features including Wo

# Asking for learning rate value
#alpha = float(input ("Please give me the desired learning rate: "));
alpha = 0.0001;

#Initialization of iterations
num_iters = 0;

#Define threshold
#threshold = float(input ("Please give me the desired threshold: "));
threshold = 0.0001

# Init weights and Run Gradient Descent 
B=[];                   #Matrix with all values
C=list();               #Support list to display values in B
#i=0;
Weights = np.zeros(n);  #Initialize weights with value of zero
#threshold=0.0001;
thr=threshold + 1;      #Define Thr as the current threshold
h = np.dot(X,Weights);  #Product of all the features with the weights give us the first hypotheses --> f(x) = wT * x
v = y - h;              #Here we find the difference between the value of our hypotheses and the actual value of y to know the error (both are vectors, so doing this we can know the error for every y)
err = np.power(v,2);    
sumerrors = np.sum(err);    #we find the sum of squared errors, we sum every error value in the vector
for num in Weights:     #Just some code to separate each value of weight to be able to display it correctly
    C.append(num);
C.insert(0,num_iters);
C.append(sumerrors);
print(C);               #C now has all our required values to display (num of iteration, weight values and sum of squared errors)
B.append(C);            #We put this values into the B matrix which we will use to save it in the csv file

while (thr > threshold):    #While our error is bigger than our threshold we are going to keep iterating looking for our correct weight value
    C=list();           #Reset C
    num_iters += 1;     #number of iterations plus 1
    sum1 = sumerrors;   #We save our last sum of squared errors
    grad = X.transpose();   #Now we start implementing our gradient, first we need the transpose of X to be able to multiply it by y-f(x)
    grad2 = np.dot(grad,v); #We multiply our last value with the value of v which is y - f(x) as stated in our gradient function
    Weights_change = alpha * grad2; #Now we multiply our result by the learning rate, so we can obtain our diference to be updated in our weights
    Weights = Weights + Weights_change; #We obtain our new weight values
    h = np.dot(X,Weights);      #Now we can obtain our new set of hypotheses or f(x)
    v = y - h;
    err = np.power(v,2);
    sumerrors = np.sum(err);    #We again calculate our error
    for num in Weights:
        C.append(num);
    C.insert(0,num_iters);
    C.append(sumerrors);
    B.append(C);
    outfile = open('./solution.csv','w', newline = '');     #lines to create a csv file with all our data
    writer=csv.writer(outfile);
    writer.writerows(B);
    print(C);
    thr = abs(sum1 - sumerrors);        #We calculate our difference between sum of square errors and if it is less than our threshold, then our work is done
#input(); #To hold the result in the command line