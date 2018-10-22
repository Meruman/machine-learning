# Data Preprocessing Template

# Importing the libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#np.set_printoptions(threshold=np.nan)

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0) #Can be median or most_frequent instead of mean
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder #OneHotEncoder helps creating dummy variables
labelencoder_x = LabelEncoder()
X[:,0] = labelencoder_x.fit_transform(X[:,0]) #Transform Categories into numbers
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray() #Transforma el valor de las categorias, en columnas diferentes formando una especie de binarios para diferenciarlos
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y) 

#Splitting the dataset into the Training set and Test set



# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""
