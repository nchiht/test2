# Import the libary
import sys
import scipy
import pandas
import numpy
import matplotlib
import sklearn

#Load the dataset
    #Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
    #Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names = names)

#Summarize the dataset
    #Dimensions of the dataset
print(dataset.shape)    
    #Peek at data
print(dataset.head(51))
    #Statistical Summary
print(dataset.describe())
    #Class distribution
print(dataset.groupby('class').size())
#Data Visualization
    #Univariate Plots

#dataset_list = dataset.values.tolist()
#print(dataset_list)
    
