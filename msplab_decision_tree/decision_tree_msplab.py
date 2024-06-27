from matplotlib import pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
import numpy as np
##other, smile, laugh, scowl
data=pd.read_csv('msplab_data.csv')
data=pd.DataFrame(data)
data_y_encoded = data['class']
print(data_y_encoded)
data_x_encoded = data.iloc[:,2:43]
print(data_x_encoded)
#print(list(data['Emotions'].unique()))
#print(list(data_x_encoded.columns))
X_train, X_test, y_train, y_test = train_test_split(data_x_encoded, data_y_encoded, test_size=0.2,random_state=0)

# dtree = DecisionTreeClassifier(max_depth=25, criterion='gini')
# dtree.fit(X_train, y_train)
dtree = RandomForestClassifier(n_estimators=100)  
dtree.fit(X_train, y_train)

fig = plt.figure(figsize=(50,50))
_ = tree.plot_tree(dtree.estimators_[0], 
                   feature_names=list(data_x_encoded.columns), 
                   #class_names=['other', 'smile', 'laughter', 'scowl'],
                   #class_names=list(data_label.unique()),
                   filled=True)
fig.savefig("decision_tree.png")
predictions = dtree.predict(X_test)
#print(y_test,predictions)
acc = accuracy_score(y_test, predictions)
print(acc)
     