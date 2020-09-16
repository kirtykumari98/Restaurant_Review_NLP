#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
df=pd.read_csv("Restaurant_Reviews.tsv",delimiter="\t",quoting=3)

#cleaning the texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,1000):
    review=re.sub("[^a-zA-Z]"," ",df['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in stopwords.words("english")]
    review=" ".join(review)
    corpus.append(review)

#creating bag of words model    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1000)
X=cv.fit_transform(corpus)
y=df.iloc[:,1].values

#splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#fitting the decision tree to the training set
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)

#predicting the test set results
y_pred=classifier.predict(X_test)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
