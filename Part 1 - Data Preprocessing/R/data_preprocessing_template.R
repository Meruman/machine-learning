# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Data.csv')

#Taking care of missing data
dataset$Age = ifelse(is.na(dataset$Age), 
                     ave(dataset$Age, FUN = function (X) mean(X, na.rm = TRUE)),
                     dataset$Age)   #is.na is a function that tells you if the value of the function is missing or not
dataset$Salary = ifelse(is.na(dataset$Salary), 
                     ave(dataset$Salary, FUN = function (X) mean(X, na.rm = TRUE)),
                     dataset$Salary)

#Encoding categorical data
dataset$Country = factor(dataset$Country,
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1,2,3)) #Factor hace todo el trabajo de convertir a factores las categorias
                        #En R es mas sencillo que python porque la funcion factor convierte las categorias en numeros pero
                        #R lo ve como factores y no niveles, los factores se lo especificamos nosotros

dataset$Purchased = factor(dataset$Purchased,
                         levels = c('No', 'Yes'),
                         labels = c(0,1))



# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)