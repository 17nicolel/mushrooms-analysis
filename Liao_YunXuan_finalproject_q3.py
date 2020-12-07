# YunXuan Liao
# ITP 449 Fall 2020
# Final Project
# Q3

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

#Build a classification tree. Random_state =2020. Training partition 0.7. stratify = y, max_depth = 6
mushrooms=pd.read_csv('mushrooms.csv')
print(mushrooms.columns)
print(mushrooms.head())
class_=mushrooms['class']
mushrooms.drop(['class'], axis = 1,inplace=True)
mushrooms['class']=class_
X=mushrooms.iloc[:,0:22]
y=mushrooms.iloc[:,22]
X=pd.get_dummies(X)




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=2020,stratify=y)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(max_depth=6,random_state=2020)
dt.fit(X_train,y_train)



#A. Print the confusion matrix. Also visualize the confusion matrix using plot_confusion_matrix from sklearn.metrics
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
y_pred = dt.predict(X_test)
confusionmatrix = metrics.confusion_matrix(y_test, y_pred)
print(confusionmatrix)
plot_confusion_matrix(dt, X_test, y_test)
plt.show()

#B. What was the accuracy on the training partition?
y_train_pred = dt.predict(X_train)
y_pred = dt.predict(X_test)
print('Accuracy =', metrics.accuracy_score(y_train,y_train_pred))

#C. What was the accuracy on the test partition?
print('Accuracy =', metrics.accuracy_score(y_test,y_pred))

#D. Show the classification tree.
from sklearn import tree
plt.figure(2)
fn=X.columns
cn=y.unique()
treevisual=tree.plot_tree(dt,feature_names=fn,class_names=cn,filled=True)
plt.show()

#E. List the top three most important features in your decision tree for determining toxicity.
#from the tree diagram the three most important features are: odor_n, stalk-root_c, spare_print_color_r


print('Feature Importance:', dt.feature_importances_)
importances = dt.feature_importances_
indices = np.argsort(importances)[::-1]
names = [X.columns[i] for i in indices]
plt.figure(3)
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), names, rotation=90)
plt.show()
#from the importance rating the three most important features are: if the odor is none or not, if the stalk-root is club or not,
#if the stalk-surface-below-ring is scaly or not.

#F. Classify the following mushroom.
sample = {'cap-shape':["x"], 'cap-surface':["s"], 'cap-color':["n"], 'bruises':["t"], 'odor':["y"],
                   'gill-attachment':["f"],'gill-spacing':["c"], 'gill-size':["n"],'gill-color':["k"],
                   'stalk-shape':["e"],'stalk-root':["e"], 'stalk-surface-above-ring':["s"], 'stalk-surface-below-ring':["s"],
                   'stalk-color-above-ring':["w"],'stalk-color-below-ring':["w"], 'veil-type':["p"], 'veil-color':["w"],
                   'ring-number':["o"], 'ring-type':["p"], 'spore-print-color':["r"], 'population':["s"], 'habitat':["u"]}

sampledf = pd.DataFrame(sample)
X1=mushrooms.iloc[:,0:22]

pd.set_option('display.max_columns', None)
new_X = pd.concat([X1, sampledf], ignore_index=True)
new_X=pd.get_dummies(new_X)
test=new_X.iloc[-1,:]
print(test)
sample_result=dt.predict([test])
print(sample_result)
print('The sample mushroom is poisonous.')
