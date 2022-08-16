import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from pandas.plotting import scatter_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pickle
true_data = pd.read_csv('True.csv')
fake_data = pd.read_csv('Fake.csv')

true_data['label']=['REAL']*len(true_data)
fake_data['label']=['FAKE']*len(fake_data)

data=true_data.append(fake_data).sample(frac=1).reset_index().drop(columns=['index'])
labels=data.label

x_train,x_test,y_train,y_test=train_test_split(data['text'], labels, test_size=0.2, random_state=7)

#Initialize a TfidfVectorizer
tf=TfidfVectorizer(stop_words='english' , max_df=0.7)

#Fit and transform train set, transform test set
tf_train=tf.fit_transform(x_train)
tf_test=tf.transform(x_test)

#initialize PassiveAggressiveClassifier
pc=PassiveAggressiveClassifier(max_iter=50)
pc.fit(tf_train,y_train)
#calculate accuracy
y_pred=pc.predict(tf_test)
score=accuracy_score(y_test,y_pred)
print(f'Acuuracy: {round(score*100,2)}%')
pickle.dump(pc,open('final_model','wb'))
pickle.dump(tf,open('tfidfit_model','wb'))
