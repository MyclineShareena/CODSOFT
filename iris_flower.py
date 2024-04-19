#import all necessary library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.ensemble import ExtraTreesClassifier , RandomForestClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')

#import the dataset
df = pd.read_csv('IRIS.csv')
#show the data
df.head()
df.info()
df.columns
df.describe()

#Preprocessing
#checking is there any null value are or not
df.isnull().sum()

#encoding
le = LabelEncoder()
for i in df.columns:
    if is_numeric_dtype(df[i]):
        continue
    else:
        df[i] = le.fit_transform(df[i])
    
df.head()

#visualisation
sns.set()
sns.countplot(x = df.species)

df0 = df[df.species == 0]
df1 = df[df.species == 1]
df2 = df[df.species == 2]

plt.figure(figsize=(20,10))
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal_length'], df0['sepal_width'],color="green",marker='+' , label = df.species == 0)
plt.scatter(df1['sepal_length'], df1['sepal_width'],color="blue",marker='.',  label = df.species == 1)
plt.scatter(df2['sepal_length'], df2['sepal_width'],color="red",marker='*',  label = df.species == 2)
plt.legend()
plt.show()

sns.histplot(data = df , x = df.sepal_length ,color = 'red')

sns.histplot(data = df , x = df.sepal_width ,color = 'yellow')

sns.histplot(data = df , x = df.petal_length ,color = 'green')

sns.histplot(data = df , x = df.petal_width ,color = 'c')

#Split dataset
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report , confusion_matrix

X = df.drop(['species'], axis='columns')
y = df.species

xtrain , xtest , ytrain , ytest = train_test_split(X , y , test_size = .2)

#Model Selection
#ExtraTreesClassifier
etc = ExtraTreesClassifier(n_estimators = 200 , random_state=100)
etc.fit(xtrain , ytrain)
print(f'train score:{etc.score(xtrain , ytrain)}')
print(f'test score:{etc.score(xtest , ytest)}')

#RandomForestClassifier
ran = RandomForestClassifier(n_estimators = 300 , random_state=100)
ran.fit(xtrain , ytrain)
print(f'train score of RandomForestClassifier : {ran.score(xtrain , ytrain)}')
print(f'test score of RandomForestClassifier : {ran.score(xtest , ytest)}')

#SVM
sv = SVC()
sv.fit(xtrain , ytrain)
print(f'train score SVM:{sv.score(xtrain , ytrain)}')
print(f'test score SVM:{sv.score(xtest , ytest)}')

#for etc 
etc_con = confusion_matrix(ytest , etc.predict(xtest))
sns.heatmap(etc_con , annot = True )
 
ran_con = confusion_matrix(ytest , ran.predict(xtest))
sns.heatmap(ran_con , annot = True )