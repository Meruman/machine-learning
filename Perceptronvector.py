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
#X = pd.read_csv(r'C:\Users\marti\Documents\mter\clases\ML\Program ass 3\Example.tsv', header = None, sep='\t')
dataset = X.iloc[:, :-1].values

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
First we need to preprocess the data, for this, we need to convert the classes from string to int, this way we can process
it later, so with the str_to_int function we change the classes from A and B to 1 and 0

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
x = np.c_[ np.ones(m), x ]; #Gives value of 1 to Xo
m,n = x.shape #m number of training set, n number of total features

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Once we have good data, we perform the perceptron with this function, which implements all the steps for a single perceptron
in a Batch mode, this is, with vectors. First we create the weights and initialize them with value of 0, after this, for 
a given number of epochs (in this excercise is 100) we perform the following:
    multiply all the instances with the weights and the sum everything for each instance, this is done with the dot product
    of the 2 values= trainingInputs and weights.
    Now that we have this, we have to compare the result with our activation function which is 1 if the result is above 0
    otherwise it is 0. We do this for all the results in our activation variable, saving our prediction value in the 
    prediction list variable.
    
    after we have all our predictions, we need the error for all the predictions which is performed and saved in the error 
    variable.
    
    now, our goal is to output how many errors for each epoch we get, so we need to count all the nonzero value in our
    error list.
    
    After this, we update the weights with the given learning rate which in this case is 1.
    
    
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
In annealing perceptron, we do the same but with learning rate = 1/(number of epoch we have done)
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
Our main program, where we call the functions and write the output tsv file.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
pred=perceptron (x,y,1,100)
pred2 = annealing_perceptron(x,y,1,100)
outfile = open(fileout+"ErrorsPerceptron.tsv",'w', newline = '')     #lines to create a csv file with all our data
writer=csv.writer(outfile, delimiter='\t')
writer.writerow(np.asarray(pred))
writer.writerow(np.asarray(pred2))