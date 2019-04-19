import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
## df = pd.read_csv('iris.csv')
## https://www.kaggle.com/mjbahmani/20-ml-algorithms-15-plot-for-beginners
## https://github.com/joeyajames/Python/blob/master/Pandas/pandas_weather.py

file = 'iris.csv'
df = pd.read_csv(file)

#################
#################

def review():
    print('*******Data Types **********')
    print(df.dtypes)
    print('*******Data Count **********')
    print(df.count())
    print('*******Null Values **********')
    print(df.isnull().values.any())                # Adapted from https://stackoverflow.com/questions/29530232/how-to-check-if-any-value-is-nan-in-a-pandas-dataframe
    print('*******Count Per Species **********')
    print(pd.value_counts(df['species'].values))

#################
#################

def describe():
    print("***** Data Total Describe *****")
    print(df.describe())
    print("***** Describe  Setosa *****")
    print(df[df.species == 'setosa'].describe())
    print("***** Describe Versicolor *****")
    print(df[df.species == 'versicolor'].describe())
    print("*****  Describe Virginica *****")
    print(df[df.species == 'virginica'].describe())

#################
#################

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
#plotGraphMatrix()

def run_iris_investigation():
    review()
    describe()

    mean_std()

    dot_plot_four_four()
    dot_petalLength()
    dot_petalWidth()
    dot_sepalLength()
    dot_sepalWidth()

    plotGraphMatrix()
    sep_width_length()
    pet_width_length()



run_iris_investigation()