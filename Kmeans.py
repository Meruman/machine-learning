# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 12:43:09 2019

@author: marti
"""

import csv
import numpy as np
import pandas as pd
import math

#Asking for the input file and desired output.
file = input("Please give me the location of the file including .tsv: ")
fileout = input("Please give me the location where you want the outputfile including the last backslash like this 'C:\Desktop\ '   : ")
nameFile = input("Please give me the name of the input file without the extension: ")
outfile = open(fileout+nameFile+"-Proto.tsv",'w', newline = '')     #lines to create a tsv file with all our data
errorOutfile = open(fileout+nameFile+"-Progr.tsv",'w', newline = '')     #lines to create a tsv file with all our data
# Importing the dataset
X = pd.read_csv(file, header = None, sep='\t')
#X = pd.read_csv(r'C:\Users\marti\Documents\mter\clases\ML\Program ass 6\Example.tsv', header = None, sep='\t')
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

"""""""""""""""""""""""""""
Define Clusters

"""""""""""""""""""""""""""
Cluster1 = [0,5]
Cluster2 = [0,4]
Cluster3 = [0,3]


def EuclDistance(dataset,Center):
    data=dataset-Center
    data=list(map(lambda x: x**2, data))
    data=[sum(x) for x in data]
    #data=data[:,0]+data[:,1]
    Distances= [math.sqrt(x) for x in data]
    return Distances

def ErrorDistance(dataset,Center):
    data=dataset-Center
    data=list(map(lambda x: x**2, data))
    data=sum(data)
    Distances= math.sqrt(data)
    return Distances


def updatePrototype(dataset,Cluster):
    Medianx=[0,0]
    for i in range (len(Cluster)):
        Medianx+=dataset[Cluster[i]]
    Medianx=Medianx/len(Cluster)
    return Medianx

def Kmeans(dataset,Cluster1,Cluster2,Cluster3):
    TestC1=EuclDistance(dataset,Cluster1)
    TestC2=EuclDistance(dataset,Cluster2)
    TestC3=EuclDistance(dataset,Cluster3)
    Clusters1=[]
    Clusters2=[]
    Clusters3=[]
    Errors=[]
    for i in range(len(y)):
        if TestC1[i]<TestC2[i] and TestC1[i]<TestC3[i]:
            Clusters1.append(i)
            Errors.append(ErrorDistance(dataset[i],Cluster1))
        elif TestC2[i]<TestC1[i] and TestC2[i]<TestC3[i]:
            Clusters2.append(i)
            Errors.append(ErrorDistance(dataset[i],Cluster2))
        elif TestC3[i]<=TestC2[i] and TestC3[i]<=TestC1[i]:
            Clusters3.append(i)
            Errors.append(ErrorDistance(dataset[i],Cluster3))
    Errors = list(map(lambda x: x**2, Errors))
    Errors=sum(Errors)
    return Clusters1, Clusters2, Clusters3,Errors


def main(dataset,Cluster1,Cluster2,Cluster3,preverror,writer,errorWriter):
    c1,c2,c3,error = Kmeans(dataset,Cluster1,Cluster2,Cluster3)
    print(Cluster1,Cluster2,Cluster3,error)
    Final=[str(Cluster1[0])+"," + str(Cluster1[1]),str(Cluster2[0])+"," +str(Cluster2[1]),str(Cluster3[0])+"," +str(Cluster3[1])]
    writer.writerow(Final)
    errorWriter.writerow([error])
    if (abs(preverror-error)>0.5):
        Cluster1=updatePrototype(dataset,c1)
        Cluster2=updatePrototype(dataset,c2)
        Cluster3=updatePrototype(dataset,c3)
        main(dataset,Cluster1,Cluster2,Cluster3,error,writer,errorWriter)
    
    
preverror=0
writer=csv.writer(outfile, delimiter='\t')
errorWriter=csv.writer(errorOutfile, delimiter='\t')
main(x,Cluster1,Cluster2,Cluster3,preverror,writer,errorWriter) 
outfile.close()
errorOutfile.close()
#c1,c2,c3,error= Kmeans(x,Cluster1,Cluster2,Cluster3)
