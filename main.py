# Load libraries
import pandas as pd
import numpy as np
from pandas.tools.plotting import scatter_matrix
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

#names= #['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
dataset = pd.read_csv('train.csv')
dataset =dataset.drop([0])
dataset =dataset.drop('PassengerId',axis =1)
dataset =dataset.drop('Name',axis = 1)


new_Pclass = pd.Categorical(dataset["Pclass"], ordered= True)
new_Pclass = new_Pclass.rename_categories(["Class1","Class2","Class3"])
dataset['Pclass'] = new_Pclass
# Sex
new_sex = pd.Categorical(dataset['Sex'])

dataset['Sex'] = new_sex


# Cabin
char_cabin = dataset['Cabin'].astype(str)
new_cabin = np.array([cabin[0] for cabin in char_cabin])
new_cabin = pd.Categorical(new_cabin)
print(new_cabin.describe())
dataset['Cabin'] = new_cabin

# shape
print(dataset.shape)

# Filling in stuff for Age
missing = np.where(dataset['Age'].isnull() == True)

#dataset.hist(column='Age',figsize=(9,6),bins=20)

#plt.show()


new_age =np.where(dataset['Age'].isnull(),28,dataset['Age'])
dataset['Age'] = new_age

#Looking For Outliers




print(dataset.dtypes)
'''
# descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby('class').size())

# box and whisker plots
plt.show()

# histograms
dataset.hist()
plt.show()

# scatter plot matrix
scatter_matrix(dataset)
plt.show()


# Split-out validation dataset
array = dataset.values
X = array[:,1:]
Y = array[:,0]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation datasetknn = KNeighborsClassifier()
knn.fit(X, Y)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
'''


