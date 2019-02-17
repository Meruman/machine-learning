# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 20:29:02 2018

@author: marti
"""

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

#Asking for the location of the file
file = input("Please give me the location of the file including .csv: ")
fileout = input("Please give me the location where you want the outputfile including the last backslash like this 'C:\Desktop\ '   : ")
print("ID3 running, please wait")

# Importing the dataset with attribute names as att
dataset = pd.read_csv(file, header = None,prefix="att")

#Separate the taget(y) to know how many classes the data set has.
y = dataset.iloc[:, -1]
Class=np.unique(y)
Class=Class.size

#Calculate the entropy of a dataset.
def entropy(columnTarget):
    
    #columnTarget parameter specifies the target column
    featureData,counter = np.unique(columnTarget,return_counts = True)
    entropy=0
    for i in range(len(featureData)):
        entropy = entropy + (-counter[i]/np.sum(counter))*(np.log2(counter[i]/np.sum(counter))/np.log2(Class))
    return entropy
    
#Calculate the gain of a dataset.
def Gain(dataSet,feature_splitted,targetFeature):
   
    #Calculate the entropy of the total dataset
    total_entropy = entropy(dataSet[targetFeature])
    
    #Calculate the values and the corresponding counter for the split feature 
    vals,counter= np.unique(dataSet[feature_splitted],return_counts=True)
    
    #Calculate the entropy of the values
    ValuesEntropy=0
    for i in range(len(vals)):
        valueTemp=dataSet.where(dataSet[feature_splitted]==vals[i]).dropna()[targetFeature]
        ValuesEntropy = ValuesEntropy + (counter[i]/len(y))*entropy(valueTemp)

    #Calculate the gain
    Gain = total_entropy - ValuesEntropy
    return Gain
       
#Perform the ID3 Algorithm
def ID3(dataSet,originaldata,features,targetFeatureName,treeNode,node_class = None):

    #Define when to return a leaf node
    
    #If all target_values have the same value, return this value and add it to the xml tree
    if len(np.unique(dataSet[targetFeatureName])) <= 1:
        val=np.unique(dataSet[targetFeatureName])[0]
        treeNode.text=str(val)
        return np.unique(dataSet[targetFeatureName])[0]
    
    #If the dataset is empty, return the most common target feature value in the original dataset and add it to the tree
    elif len(dataSet)==0:
        #Cal gives the value of the most common target feature value with the help of argmax that gives us the index of the maximum value
        cal=np.unique(originaldata[targetFeatureName])[np.argmax(np.unique(originaldata[targetFeatureName],return_counts=True)[1])]
        treeNode.text=str(cal)
        return cal
    
    #If the feature space is empty, return the most common target feature value of the direct parent node 
    #the most common target feature value is stored in the node_class variable.
    
    elif len(features) ==0:
        treeNode.text=str(node_class)
        return node_class
    
    #If none of the above is true, grow the tree
    
    else:
        #Set the most common target feature value of the current node
        node_class = np.unique(dataSet[targetFeatureName])[np.argmax(np.unique(dataSet[targetFeatureName],return_counts=True)[1])]
        
        #Select the feature which best splits the dataset

        #Return the gain values for the features in the dataset, store it in the item_values array and then choose the one with the max gain value
        item_values = [Gain(dataSet,feature,targetFeatureName) for feature in features] 
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        #Remove the feature with the best gain from the features data set
        features = [i for i in features if i != best_feature]
        
        #Code to sort the best feature values according to its entropy
        valsTemp= pd.unique(dataSet[best_feature])
        ValuesEnt=[]
        for i in range(len(valsTemp)):
            ValuesEnt.append(entropy(dataSet.where(dataSet[best_feature]==valsTemp[i]).dropna()[targetFeatureName]))
        ValuesEnt, valsTemp = zip(*sorted(zip(ValuesEnt, valsTemp)))
        i=0

        #For each value in the best feature attribute, grow more branches
        for value in valsTemp:
            value = value
            temp=ValuesEnt[i]
            
            #Add the nodes to the xml file
            x=ET.SubElement(treeNode,"node",entropy = str(temp),value = value,feature = best_feature)
            
            #Split the dataset along the value of the best feature and create sub datasets
            subData = dataSet.where(dataSet[best_feature] == value).dropna()

            #Call the ID3 algorithm for each of those sub_datasets with the new parameters
            ID3(subData,dataset,features,targetFeatureName,x,node_class)
            i=i+1
        

#Create the tree structure. The root gets the name of the feature (best_feature) with the maximum gain in the first run   
root = ET.Element("tree", entropy =  str(entropy(y)))   

#Perform the ID3 Algorithm 
tree = ID3(dataset,dataset,dataset.columns[:-1],dataset.columns[-1],root)

#Create the file with the complete tree
rootWrite = ET.ElementTree(root)
rootWrite.write(fileout + "ID3Solution2.xml")
print("Decision tree successfully complete, ID3Solution.xml file created")
