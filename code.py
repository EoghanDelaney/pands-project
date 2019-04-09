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


def plotGraph():
    sns.pairplot(df, hue='species')
    plt.show()
    

print_content()
plotGraph()