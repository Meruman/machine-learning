# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 13:58:33 2018

@author: MeRu
"""

import csv
import numpy as np
import pandas as pd
import math
from math import pi
from math import e

#Asking for the input file and desired output.
file = input("Please give me the location of the file including .tsv: ")
fileout = input("Please give me the location where you want the outputfile including the last backslash like this 'C:\Desktop\ '   : ")

# Importing the dataset
X = pd.read_csv(file, header = None, sep='\t')
dataset=X.iloc[:,:].values
for i in range(len(X.columns)):
    if np.any(pd.isnull(X.values[:,-1])):    
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

def mean(numbers):
    result = sum(numbers)/float(len(numbers))
    return result

def stddev(numbers):
    avg = mean(numbers)
    square_diff_list = []
    squared_diff=numbers-avg
    square_diff_list=square(squared_diff)
    squared_diff_sum = sum(square_diff_list)
    sample_n =len(numbers)-1
    var = squared_diff_sum/sample_n
    return var

def square(lista):
    return map(lambda x: x ** 2, lista)

def priorprob(dataset, target):
    vals,counter= np.unique(dataset[target],return_counts=True)
    prior=[]
    for i in range(len(vals)):
        prior.append(counter[i]/len(dataset))
    return prior

def train(dataset,target):
    vals= pd.unique(dataset[target])
    means = []
    stands =[]
    output = []
    output2 = []
    tools=[]
    for i in range(len(vals)):
        filtered=dataset[dataset[target]==vals[i]]
        filtered = filtered.drop(filtered.columns[target],axis=1)
        temp = []
        for j in range(1,len(filtered.columns)+1):
            means.append(mean(filtered[j]))
            stands.append(stddev(filtered[j]))
            temp = temp + [mean(filtered[j]),stddev(filtered[j])]
        output.append(temp)
        tools=tools + [temp]
    filtered2 = dataset.drop(dataset.columns[target],axis=1)
    likelihood = []
    t=0
    for j in range(1,len(filtered2.columns)+1):
        for z in range(len(vals)):
            likelihood=likelihood + [normal_pdf(filtered2[j],tools[z][t],tools[z][t+1])]
        t=t+2
    output2.append(likelihood)
    return output,output2

def normal_pdf(x,mean,stdev):
    exp_squared_diff = (x - mean) ** 2
    exp_power = -exp_squared_diff / (2 * stdev)
    exponent = e ** exp_power
    denominator = ((2 * pi* stdev) ** .5) 
    normal_prob = exponent / denominator
    return normal_prob
        

def decitions(likelyhood,prior,clases):
    predictions=[]
    suma=pd.Series()
    sumatoria=0
    finalprediction=[]
    for i in range(clases):
        suma=likelyhood[i]*prior[i]
        sumatoria+=suma
    for i in range(clases):
        predictions=predictions+[(likelyhood[i]*prior[i])/sumatoria]
    for i in range (len(predictions[0])):
        if predictions[0][i] >=0.5 :
            finalprediction.append(1)
        else:
            finalprediction.append(0)
    return finalprediction
    

        
result,likelihood = train(dataset,0)
prior=priorprob(dataset,0)
vals= pd.unique(dataset[0])
prodresult =[]
t=0
for i in range(len(vals)):
    product = likelihood[0][t] * likelihood[0][t+2]
    t+=1
    prodresult=prodresult+[product]
prediction=decitions(prodresult,prior,len(vals))
error=y-prediction
count = np.count_nonzero(error)

outfile = open(fileout+"NBSolution.tsv",'w', newline = '')     #lines to create a tsv file with all our data
writer=csv.writer(outfile, delimiter='\t')
for i in range(len(result)):
    final=[]
    for j in range(len(result[0])):
        final.append(result[i][j])
    final.append(prior[i])
    writer.writerow(np.asarray(final))
writer.writerow([count])
outfile.close()
