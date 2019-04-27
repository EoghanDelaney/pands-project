## Adapted directly from
## https://chrisalbon.com/machine_learning/trees_and_forests/random_forest_classifier_example/

# Load the library with the iris dataset
from sklearn.datasets import load_iris

# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

# Load pandas
import pandas as pd

# Load numpy
import numpy as np

# Set random seed
np.random.seed(0)

# Create an object called iris with the iris data
king = load_iris()


def print_fuc(name):
    print('-'*20)
    print(name)
    print('-'*20)

# Create a dataframe with the four feature variables
df = pd.DataFrame(king.data, columns=king.feature_names)

df['species'] = pd.Categorical.from_codes(king.target, king.target_names)

df['is_train'] = np.random.uniform(0, 1, len(df)) <= .8
print(df.head())

train = df[df['is_train']==True] 
test = df[df['is_train']==False]

#target_names = df.species.unique()

print(len(train))
print(len(test))

# Create a list of the feature column's names

print_fuc('feactures')

features = df.columns[:4]

print(features)

# train['species'] contains the actual species names. Before we can use it,
# we need to convert each species name into a digit. So, in this case there
# are three species, which have been coded as 0, 1, or 2.
print_fuc('y')
y = pd.factorize(train['species'])[0]
print(y)

# Create a random forest Classifier. By convention, clf means 'Classifier'
print_fuc('clf')
clf = RandomForestClassifier(n_jobs=2, random_state=0,n_estimators=10)


# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(train[features],y)

# Apply the Classifier we trained to the test data (which, remember, it has never seen before)
clf.predict(test[features])

# View the predicted probabilities of the first 10 observations
print_fuc('predicte prob')
print(clf.predict_proba(test[features])[0:])

# Create actual english names for the plants for each predicted plant class
preds = king.target_names[clf.predict(test[features])]
print_fuc('preds')
# View the PREDICTED species for the first five observations
print(preds)


print_fuc('species head')
# View the ACTUAL species for the first five observations
print(test['species'].head())

# Create confusion matrix
print_fuc('Confusion Matrix')
x = pd.crosstab(test['species'], preds, rownames=['Actual Species'], colnames=['Predicted Species'])
print(x)

print_fuc('Weigthing of importance')
# View a list of the features and their importance scores
print(list(zip(train[features], clf.feature_importances_)))