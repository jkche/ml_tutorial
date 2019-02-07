# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# dataset dimensions -> shape
print(dataset.shape)

# peek data -> head(first # rows you want to peek)
print(dataset.head(20))

# descriptions (mathematical calculations on data, e.g. mean) -> describe
print(dataset.describe())

# class distribution -> # instances per class -> groupby(col-label).size()
print(dataset.groupby('class').size())

# create box-and-whisker plots of each input variable/attribute -> plot(options)
dataset.plot(kind='box', subplots=True, layout =(2,2), sharex = False, sharey = False)
plt.show()

# create histograms for inputs to see distribution clearer
dataset.hist()
plt.show()

# multivariate plots
# create scatterplots of all pairs of attributes; helps see structured relationships
scatter_matrix(dataset)
plt.show()
