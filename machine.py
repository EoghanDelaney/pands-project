import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


file = 'iris.csv'
df = pd.read_csv(file)


###################
###################
### Machine Learning

# Adapted from https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# Adapted from https://www.kaggle.com/diegosch/classifier-evaluation-using-confusion-matrix
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



# At this point we split out the data - Train and Test data.
X = df.values[:,0:4]    # X being the measurements we know
Y = df.values[:,4]      # Y being the species/target

# X_train - Measurements to train the algorithm
# Y_train - Target to also train the algorithm
# X_validation - To test the measurements
# Y_validation - To test the target 
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.20, random_state=7)

def plot_graph(name, pred):
    cm = confusion_matrix(Y_validation,pred)
    cm_df = pd.DataFrame(cm,index = ['setosa','versicolor','virginica'], columns = ['setosa','versicolor','virginica'])
    plt.figure(figsize=(5.5,4))
    sns.heatmap(cm_df, annot=True)
    plt.title( name + ' \nAccuracy:{0:.3f}'.format(metrics.accuracy_score(Y_validation,pred)))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def knear_neig():
    print('************* K-Nearest Neighbors ******************')
    name = 'K-Nearest Neighbors'
    knc = KNeighborsClassifier(n_neighbors=3).fit(X_train, Y_train)
        
    pred = knc.predict(X_validation)
    plot_graph(name, pred)
    
    print('Accuracy score:' + str(metrics.accuracy_score(Y_validation,pred)))
    print('**** Confusion Matrix ****')
    print(confusion_matrix(Y_validation,pred))
    print('**** Classification Report ****')
    print(classification_report(Y_validation, pred))

def reg_mod():
    print('************* Logistic Regression ******************') 
    name = 'Logistic Regression'       
    reg = LogisticRegression(solver='liblinear', multi_class='ovr').fit(X_train, Y_train)
        
    pred = reg.predict(X_validation)
    plot_graph(name, pred)
    
    print('Accuracy score:' + str(metrics.accuracy_score(Y_validation,pred)))
    print('**** Confusion Matrix ****')
    print(confusion_matrix(Y_validation,pred))
    print('**** Classification Report ****')
    print(classification_report(Y_validation, pred))

def lin_dis():
    print('************* Linear Discriminant Analysis ******************')        
    name = 'Linear Discriminant Analysis'
    lnd = LinearDiscriminantAnalysis().fit(X_train, Y_train)
        
    pred = lnd.predict(X_validation)
    plot_graph(name, pred)
    
    print('Accuracy score:' + str(metrics.accuracy_score(Y_validation,pred)))
    print('**** Confusion Matrix ****')
    print(confusion_matrix(Y_validation,pred))
    print('**** Classification Report ****')
    print(classification_report(Y_validation, pred))

def gauss():
    print('************* Gaussian ******************')        
    name = 'Gaussian'
    gaus = GaussianNB().fit(X_train, Y_train)
        
    pred = gaus.predict(X_validation)
    plot_graph(name, pred)
    
    print('Accuracy score:' + str(metrics.accuracy_score(Y_validation,pred)))
    print('**** Confusion Matrix ****')
    print(confusion_matrix(Y_validation,pred))
    print('**** Classification Report ****')
    print(classification_report(Y_validation, pred))

knear_neig()
reg_mod()
lin_dis()
gauss()
















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
