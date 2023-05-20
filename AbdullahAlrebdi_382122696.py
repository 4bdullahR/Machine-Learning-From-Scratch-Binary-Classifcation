# Glucose,BloodPressure,SkinThickness,Insulin,BMI,Age,Outcome
import numpy as np
from csv import reader
from random import randrange
#import pandas as pd
#from sklearn.linear_model import LogisticRegression
#from sklearn.datasets import *
from sklearn.metrics import *
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

#load the data
def load_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row or len(row)!=7 or row[0].isalpha():
                continue
            dataset.append(row)
    return dataset

#string to float
def strToFloat(dataset):
    for row in dataset:
        for column in range(len(row)):
            row[column]=float(row[column])

#minimum and maximum
def dataset_min_max(dataset):
    minmax = []
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min,value_max])
    return minmax

#normlaztion
def norm(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-1): # -1 for skipping the class
            row[i] = (row[i] - minmax[i][0]) / ((minmax[i][1] - minmax[i][0]))

##devide dataset into three thirds which fits perfectly in our example
#def split(dataset, split=0.66, count=0): #Cross Validation Function
#    train = []
#    train_size = split*len(dataset)
#    test = list(dataset)

#    while len(train) < train_size:
#        if count == 0:
#           train.append(test.pop()) #second two thirds for traing
#        elif count == 1:
#            train.append(test.pop(0))#first two thirds for traning
#        else:
#            train.append(test.pop()) #first and last thirds for traing
#            train.append(test.pop(0))
#    return train, test

#devide dataset randomly **SOME TIMES LEADS TO PERFECT RESULTS IN MY EXPERNCE** 
def split(dataset, split=0.70, count=0):
    train = []
    train_size = split*len(dataset)
    test = list(dataset)

    while len(train) < train_size:
        index = randrange(len(test))
        train.append(test.pop(index))
    return train, test

#sigmoid function returns y
def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y

def predect(X,w):
    z = np.dot(X,w)
    return sigmoid(z)

def cost(y,y_pred):
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def train(X,y,epochs=1000, lr=0.01):
    X = np.c_[np.ones((X.shape[0],1)), X]
    w = np.random.randn(X.shape[1])

    for epoch in range(epochs):
        y_pred = predect(X,w)
        error = y_pred - y
        gradient = np.dot(X.T, error) / y.size
        w -= lr * gradient

        #if epoch % 100 == 0:
        #    c = cost(y,y_pred)
        #    print(f"Epoch {epoch}: cost = {c}")
    return w

#####################################################################

##Load Data From The File
filename = 'diabetes.csv'
dataset = load_csv(filename)
print(f'-{filename} inctances:{len(dataset)} features:{len(dataset[0])}\n') #pd.read_csv("diabetes.csv") 


##Converting Strings To Floats
strToFloat(dataset)
print(f"-An example after float strings: {dataset[0]} \n")


##Minimum/Maximum For Each Feature
min_max = dataset_min_max(dataset)
print(f"-MIN_MAX each colom: {min_max} \n")


##Dataset Normlization  
norm(dataset, min_max)
print(f"-An example after normlaztion: {dataset[0][0:4]}... \n")

for i in range(3):
    ##Spliting Dataset To  Train/Test  Sets 
    training_set, test_set = split(dataset,count=i)
    #print(f" training_set example = {training_set[0]} \n test_set example = {test_set[0]}")
    #print(f"length of test_set {len(test_set)}, train_set = {len(training_set)}")


    ##Training/Test Dataset (features values), With Thier Lables y (class 1 or 0)
    X_train = np.array([i[:-1] for i in training_set])
    y_train = np.array([i[-1] for i in training_set])
    X_test = np.array([i[:-1] for i in test_set])
    y_test = [i[-1] for i in test_set]


    ##Training Stars With Given Data
    w = train(X_train,y_train,epochs=3000,lr=0.2)


    ##Testing 
    y_pred = predect(np.c_[np.ones((X_test.shape[0], 1)), X_test],w)
    y_pred_float = [1.00 if i > 0.5 else 0.00 for i in y_pred]


    print("ACCAUL, PREDECTED CLASS: ** 3 samples **")
    for i in range(3):
        print(f'{y_test[i]} : {y_pred_float[i]}')


    accr = accuracy_score(y_test ,y_pred_float) *100
    prec = precision_score(y_test ,y_pred_float) *100
    recall = recall_score(y_test ,y_pred_float) *100
    f1 = f1_score(y_test ,y_pred_float) *100


    print(f'{accr:.02f}% accuracy_score')
    print(f'{prec:.02f}% precision_score')
    print(f'{recall:.02f}% recall_score')
    print(f'{f1:.02f}% f1_score')

    dtree = DecisionTreeClassifier()
    print(f'Classifier evaluation using cross-validation:{cross_val_score(dtree,X_test,y_test,cv=3)}\n')


