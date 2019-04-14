import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

#################
#def labelPlot(title):
   # heading = lm.fig.suptitle(title, fontsize=12)
  #  plt.show()


#################

def petalLength():
    sns.swarmplot(x="species", y="petal_length", data=df)
    plt.show()

def petalWidth():
    sns.swarmplot(x="species", y="petal_width", data=df)
    plt.show()

def sepalLength():
    sns.swarmplot(x="species", y="sepal_length", data=df)
    plt.show()

def sepalWidth():
    sns.swarmplot(x="species", y="sepal_width", data=df)
    plt.show()

################

def sep_width_length():
    sns.FacetGrid(df, hue="species", height=5)\
    .map(plt.scatter, "sepal_length", "sepal_width")\
    .add_legend()
    plt.show()

def pet_width_length():
    sns.FacetGrid(df, hue="species", height=5)\
    .map(plt.scatter, "petal_length", "petal_width")\
    .add_legend()
    plt.show()

def plotGraph():
    sns.pairplot(df, hue='species')\
    .add_legend()
    plt.show()

###################


print_content()


petalLength()
petalWidth()
sepalLength()
sepalWidth()

sep_width_length()
pet_width_length()

plotGraph()

