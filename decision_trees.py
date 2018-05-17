# Implementing Decision Trees and Random Forest
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.ensemble import RandomForestClassifier
import pydotplus


input_file = 'C:/Users/Dilpreet Singh/Desktop/Machine Learning/DataScience-Python3/PastHires.csv'
df = pd.read_csv(input_file, header = 0)

# Mapping data to numerical values
d = {'Y':1, 'N':0}
df['Hired'] = df['Hired'].map(d)
df['Employed?'] = df['Employed?'].map(d)
df['Top-tier school'] = df['Top-tier school'].map(d)
df['Interned'] = df['Interned'].map(d)

d = {'BS':0, 'MS':1, 'PhD':2}
df['Level of Education'] = df['Level of Education'].map(d)

# Seperate the features from the target column
features = list(df.columns[:6])
y = df['Hired']
X = df[features]

# Constructing the Decision Tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

# Displaying the tree
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=features)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('tree.png')

# Ensemble learning: using a random forest
clf = RandomForestClassifier(n_estimators=10) # for 10 decision trees
clf = clf.fit(X, y)

# Predict employment of an employed 10-year veteran
print(clf.predict([[10,1,4,0,0,0]]))

# Predict employement of an unemployed 10-year veteran
print(clf.predict([[10,0,4,0,0,0]]))





