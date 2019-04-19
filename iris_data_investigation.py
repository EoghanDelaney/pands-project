import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
## https://www.kaggle.com/mjbahmani/20-ml-algorithms-15-plot-for-beginners
## https://github.com/joeyajames/Python/blob/master/Pandas/pandas_weather.py

file = 'iris.csv'
df = pd.read_csv(file)

def print_content():
    print("***** Data Types *****")
    print(df.dtypes)
    print("***** Data Total Describe *****")
    print(df.describe())
    print("***** Describe  Setosa *****")
    print(df[df.species == 'setosa'].describe())
    print("***** Describe Versicolor *****")
    print(df[df.species == 'versicolor'].describe())
    print("*****  Describe Virginica *****")
    print(df[df.species == 'virginica'].describe())


#### Develope Table from above data
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


# title = ('','Sepal Length','Sepal Width','Petal Length','Petal Width')
# list = round(df[df.species == 'virginica'].std(),3).tolist()

# We can then append the above into a table
#################
### Box Plots entered Here####
# Adapted from the attached https://stackoverflow.com/questions/52472757/creating-a-boxplot-facetgrid-in-seaborn-for-python
def box_plot():
    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()

    ax = sns.boxplot(x="species", y="sepal_length", data=df, orient='v', ax=axes[0])
    ax = sns.boxplot(x="species", y="sepal_width", data=df, orient='v', ax=axes[1])
    ax = sns.boxplot(x="species", y="petal_length", data=df, orient='v', ax=axes[2])
    ax = sns.boxplot(x="species", y="petal_width", data=df, orient='v', ax=axes[3])
    plt.show()

#################

def petalLength():
    sns.swarmplot(x="species", y="petal_length", data=df).set_title('Petal Length')
    plt.show()

def petalWidth():
    sns.swarmplot(x="species", y="petal_width", data=df).set_title('Petal Width')
    plt.show()

def sepalLength():
    sns.swarmplot(x="species", y="sepal_length", data=df).set_title('Sepal Length')
    plt.show()

def sepalWidth():
    sns.swarmplot(x="species", y="sepal_width", data=df).set_title('Sepal Width')
    plt.show()

def dot_plot ():
    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()
    
    ax = sns.swarmplot(x="species", y="petal_length", data=df, orient='v', ax=axes[0]).set_title('Petal Length')
    ax = sns.swarmplot(x="species", y="petal_width", data=df, orient='v', ax=axes[1]).set_title('Petal Width')
    ax = sns.swarmplot(x="species", y="sepal_length", data=df, orient='v', ax=axes[2]).set_title('Sepal Length')
    ax = sns.swarmplot(x="species", y="sepal_width", data=df, orient='v', ax=axes[3]).set_title('Sepal Width')
    
    plt.show()

################

def plotGraph():
    sns.pairplot(df, hue='species').add_legend().fig.suptitle('Plot Matrix')
    plt.show()

def sep_width_length():
    sns.FacetGrid(df, hue="species", height=5).map(plt.scatter, "sepal_length", "sepal_width").add_legend().fig.suptitle('Sepal Length Vs Width')
    plt.show()

def pet_width_length():
    sns.FacetGrid(df, hue="species", height=5).map(plt.scatter, "petal_length", "petal_width").add_legend().fig.suptitle('Petal Length Vs Wodth')
    plt.show()

###################
# This can be save info into a CSV
#df.describe().to_csv('test.csv', mode="a")

###################
mean_std()
#print_content()
#box_plot()
#dot_plot()
#petalLength()
#petalWidth()
#sepalLength()
#sepalWidth()

#plotGraph()

