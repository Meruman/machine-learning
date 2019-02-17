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
#Separate the dataset into features(x) and taget(y)
X = dataset.iloc[:, :-1].values;
y = dataset.iloc[:, -1].values;
m = len(y);
X = np.c_[ np.ones(m), X ]; 
m,n = X.shape; 

# Asking for learning rate value
alpha = 0.0001;

#Initialization of iterations
num_iters = 0;

#Define threshold
threshold = 0.0001

# Init weights and Run Gradient Descent 
B=[];                   
C=list();               
#i=0;
Weights = np.zeros(n);  
#threshold=0.0001;
thr=threshold + 1;      
h = np.dot(X,Weights);  
v = y - h;              
err = np.power(v,2);    
sumerrors = np.sum(err);    
for num in Weights:     
    C.append(num);
C.insert(0,num_iters);
C.append(sumerrors);
print(C);               
B.append(C);            

while (thr > threshold):    
    C=list();           
    num_iters += 1;     
    sum1 = sumerrors;   
    grad = X.transpose();   
    grad2 = np.dot(grad,v); 
    Weights_change = alpha * grad2; 
    Weights = Weights + Weights_change; 
    h = np.dot(X,Weights);      
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
