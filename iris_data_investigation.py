import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

## I left the below there to copy and paste into ipython every time I returned to the project
## import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
## df = pd.read_csv('iris.csv')

file = 'iris.csv'
df = pd.read_csv(file)

## Code introduction
## I have broken all the elements into functions.
## I have done this for two reasons 
## 1. I feel it is more reflective of what is done in industry 
## 2. When troubleshooting the code it is more straightforward to debug.
## I could comment out the execution of that function leaving me with the remaining
## function I was focusing on at the time.
## I also feel its reads better and clearer to follow.


#################
#################
### Review the data 


def review():
    print('******* Data Types **********')
    print(df.dtypes)
    print('******* Data Count **********')
    print(df.count())
    print('******* Null Values **********')
    print(df.isnull().values.any())                # Adapted from https://stackoverflow.com/questions/29530232/how-to-check-if-any-value-is-nan-in-a-pandas-dataframe
    print('******* Count Per Species **********')
    print(pd.value_counts(df['species'].values))

#################
#################
### Describe the data grouped 

def describe():
    print("***** Data Total Describe *****")
    print(df.describe())
    print("***** Describe  Setosa *****")
    print(df[df.species == 'setosa'].describe())
    print("***** Describe Versicolor *****")
    print(df[df.species == 'versicolor'].describe())
    print("***** Describe Virginica *****")
    print(df[df.species == 'virginica'].describe())

#################
#################
### Isolate the means and export it to CSV

def mean_std():
    print('************** Means for all Species ***********')
    print("*** Setosa Mean: \n" + str(round(df[df.species == 'setosa'].mean(),3)))
    print("*** Versicolor Mean: \n" + str(round(df[df.species == 'versicolor'].mean(),3)))
    print("*** Virginica Mean: \n" + str(round(df[df.species == 'virginica'].mean(),3)))

    print('************** Standard Devation for all Species ***********')
    print("*** Setosa Std: \n" + str(round(df[df.species == 'setosa'].std(),3)))
    print("*** Versicolor Std: \n" + str(round(df[df.species == 'versicolor'].std(),3)))
    print("*** Virginica Std: \n" + str(round(df[df.species == 'virginica'].std(),3)))

    mean = round(df.groupby('species').mean(),3)
    std = round(df.groupby('species').std(),3)
    mean.to_csv('csv/species_mean.csv', mode="a")
    std.to_csv('csv/species_std.csv', mode="a")

#################
#################
### Generate a Histogram from the data

def sepal_len_hist():
    sns.distplot(df[df.species == 'setosa']['sepal_length'])
    sns.distplot(df[df.species == 'versicolor']['sepal_length'])
    sns.distplot(df[df.species == 'virginica']['sepal_length'])
    ABC = ['setosa', 'versicolor', 'virginica']
    plt.legend(ABC)
    plt.show()

def sepal_wid_hist():
    sns.distplot(df[df.species == 'setosa']['sepal_width'])
    sns.distplot(df[df.species == 'versicolor']['sepal_width'])
    sns.distplot(df[df.species == 'virginica']['sepal_width'])
    ABC = ['setosa', 'versicolor', 'virginica']
    plt.legend(ABC)
    plt.show()

def petal_len_hist():
    sns.distplot(df[df.species == 'setosa']['petal_length'])
    sns.distplot(df[df.species == 'versicolor']['petal_length'])
    sns.distplot(df[df.species == 'virginica']['petal_length'])
    ABC = ['setosa', 'versicolor', 'virginica']
    plt.legend(ABC)
    plt.show()

def petal_wid_hist():
    sns.distplot(df[df.species == 'setosa']['petal_width'])
    sns.distplot(df[df.species == 'versicolor']['petal_width'])
    sns.distplot(df[df.species == 'virginica']['petal_width'])
    ABC = ['setosa', 'versicolor', 'virginica']
    plt.legend(ABC)
    plt.show()

#################
#################
### Generate a Dot plot from the data

# Adapted from the attached https://stackoverflow.com/questions/52472757/creating-a-boxplot-facetgrid-in-seaborn-for-python
def dot_plot_four_four():
    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()
    ax = sns.swarmplot(x="species", y="petal_length", data=df, orient='v', ax=axes[0]).set_title('Petal Length')
    ax = sns.swarmplot(x="species", y="petal_width", data=df, orient='v', ax=axes[1]).set_title('Petal Width')
    ax = sns.swarmplot(x="species", y="sepal_length", data=df, orient='v', ax=axes[2]).set_title('Sepal Length')
    ax = sns.swarmplot(x="species", y="sepal_width", data=df, orient='v', ax=axes[3]).set_title('Sepal Width')
    plt.show()

def dot_petalLength():
    sns.swarmplot(x="species", y="petal_length", data=df).set_title('Petal Length')
    plt.show()

def dot_petalWidth():
    sns.swarmplot(x="species", y="petal_width", data=df).set_title('Petal Width')
    plt.show()

def dot_sepalLength():
    sns.swarmplot(x="species", y="sepal_length", data=df).set_title('Sepal Length')
    plt.show()

def dot_sepalWidth():
    sns.swarmplot(x="species", y="sepal_width", data=df).set_title('Sepal Width')
    plt.show()

################
################
### Generate a plot matrix from the data - all characteristics vs one another

def plotGraphMatrix():
    sns.pairplot(df, hue='species').add_legend().fig.suptitle('Plot Matrix')
    plt.show()

def sep_width_length():
    sns.FacetGrid(df, hue="species", height=5).map(plt.scatter, "sepal_length", "sepal_width").add_legend().fig.suptitle('Sepal Length Vs Width')
    plt.show()

def pet_width_length():
    sns.FacetGrid(df, hue="species", height=5).map(plt.scatter, "petal_length", "petal_width").add_legend().fig.suptitle('Petal Length Vs Wodth')
    plt.show()

###################
###################
### Radviz and Andrew Curves 
# https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html

def rad_viz():
    pd.plotting.radviz(df, 'species')
    plt.title('Radviz Plot')
    plt.show()
# https://github.com/pandas-dev/pandas/blob/v0.24.2/pandas/plotting/_misc.py#L272-L360

def andreCurv():
    pd.plotting.andrews_curves(df, 'species')
    plt.title('Andrews Curves')
    plt.show()

###################
###################
### Machine Learning

# Adapted from https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
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


def knear_neig():
    # At this point we split out the data - Train and Test data.
    X = df.values[:,0:4]    # X being the measurements we know
    Y = df.values[:,4]      # Y being the species/target
    
    # X_train - Measurements to train the algorithm
    # Y_train - Target to also train the algorithm
    # X_validation - To test the measurements
    # Y_validation - To test the target 
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.20, random_state=7)
    knc = KNeighborsClassifier(n_neighbors=3).fit(X_train, Y_train)
    pred = knc.predict(X_validation)
    tee = metrics.accuracy_score(Y_validation,pred)
    print(tee)






'''models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
'''














###################
###################
### Output all data and graphs

def run_iris_investigation():
    review()
    describe()

    mean_std()

    sepal_len_hist()
    sepal_wid_hist()
    petal_len_hist()
    petal_wid_hist()

    dot_plot_four_four()
    dot_petalLength()
    dot_petalWidth()
    dot_sepalLength()
    dot_sepalWidth()

    plotGraphMatrix()
    sep_width_length()
    pet_width_length()

    rad_viz()
    andreCurv()

#run_iris_investigation()
knear_neig()