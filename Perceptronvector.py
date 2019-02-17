# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 03:02:34 2018

@author: martin mendez ruiz
"""
import csv
import numpy as np
import pandas as pd

#Asking for the input file and desired output.
file = input("Please give me the location of the file including .tsv: ")
fileout = input("Please give me the location where you want the outputfile including the last backslash like this 'C:\Desktop\ '   : ")
print("Training perceptron")

# Importing the dataset
X = pd.read_csv(file, header = None, sep='\t')
dataset = X.iloc[:, :-1].values

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
preprocess the data

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def str_to_int(dataset, column):
    class_values = pd.Series(dataset[:,column])
    unique = reversed(class_values.unique().tolist())
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

str_to_int(dataset, 0)
dataset=pd.DataFrame(dataset,dtype=np.float64)
x = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values
m = len(y);
x = np.c_[ np.ones(m), x ]; 
m,n = x.shape 

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Once we have good data, we perform the perceptron, which implements all the steps for a single perceptron
in a Batch mode   
    
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Perceptron with batch stochastic gradient descent and constant learning rate
def perceptron(Inputs,Labels, l_rate, n_epoch):
    predictions = list()
    weights = np.zeros(n)
    for epoch in range(n_epoch+1):
        prediction=[]
        activation =  np.dot(Inputs,weights)
        for s in range(len(activation)):
            if activation[s]>0:
                prediction.append(1)
            else:
                prediction.append(0)
        error = Labels - prediction
        count = np.count_nonzero(error)
        deltaTemp=Inputs.transpose()
        weights+=  np.dot(deltaTemp,error) * l_rate
        predictions.append(count)
    return predictions

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
annealing perceptron
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


# Perceptron with batch stochastic gradient descent and annealing learning rate
def annealing_perceptron(Inputs,Labels, l_rate, n_epoch):
    predictions = list()
    weights = np.zeros(n)
    for epoch in range(n_epoch+1):
        prediction=[]
        l_rate2 = l_rate/(epoch+1)
        activation =  np.dot(Inputs,weights)
        for s in range(len(activation)):
            if activation[s]>0:
                prediction.append(1)
            else:
                prediction.append(0)
        error = Labels - prediction
        count = np.count_nonzero(error)
        deltaTemp=Inputs.transpose()
        weights+=  np.dot(deltaTemp,error)* l_rate2
        predictions.append(count)
    return predictions


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 main program
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
pred=perceptron (x,y,1,100)
pred2 = annealing_perceptron(x,y,1,100)
outfile = open(fileout+"ErrorsPerceptron.tsv",'w', newline = '')     #lines to create a csv file with all our data
writer=csv.writer(outfile, delimiter='\t')
writer.writerow(np.asarray(pred))
writer.writerow(np.asarray(pred2))
