import numpy as np
from nltk import word_tokenize
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
# importing necessary libraries 
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 

glove_file = datapath('C:\\Users\\Pathy\\Downloads\\glove.6B\\Hackathon\\glove.6B.50d.txt')
tmp_file = get_tmpfile("test_word2vec.txt")
_ = glove2word2vec(glove_file, tmp_file)
model = KeyedVectors.load_word2vec_format(tmp_file)
print('model built')

def document_vector(word2vec_model, doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in word2vec_model.vocab]
    return np.mean(word2vec_model[doc], axis=0)

def preprocess(text):
    text = text.lower()
    doc = word_tokenize(text)
    doc = [word for word in doc if word.isalpha()] #restricts string to alphabetic characters only
    return doc

with open('questions.txt') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 

with open('categories.txt') as f:
    cat = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
cat = [x.strip() for x in cat] 

vec = []


for text in content:
    doc = preprocess(text)
    vec.append(document_vector(model, doc))

  
X = vec 
y = cat 

#print(y)


# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 
  
# training a Naive Bayes classifier 
'''
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB().fit(X_train, y_train) 
gnb_predictions = gnb.predict(X_test) 
'''

'''
from sklearn import svm
clf = svm.SVC(gamma='scale')
clf.fit(X_train, y_train) 
clf_predictions = clf.predict(X_test) 
'''

'''
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                              random_state=0)
clf.fit(X_train, y_train) 
clf_predictions = clf.predict(X_test) 
'''

'''
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0, solver='lbfgs',
                          multi_class='multinomial').fit(X_train, y_train)							  
'''


from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(150,10), random_state=1)
				 
'''	
from xgboost import XGBClassifier
clf = XGBClassifier()
'''	

clf.fit(X_train, y_train) 
clf_predictions = clf.predict(X_test) 

'''

from imblearn.ensemble import BalancedRandomForestClassifier
clf = BalancedRandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
clf_predictions = clf.predict(X_test) 
'''

# accuracy on X_test 

accuracy = clf.score(X_test, y_test) 
print(accuracy)
 
# creating a confusion matrix 
cm = confusion_matrix(y_test, clf_predictions)
print(cm)

#function for inferring from model

def infer_from_model(class_model, text):
	text = preprocess(text)
	vec = document_vector(model, text)
	inferred = class_model.predict(vec.reshape(1, -1))
	return inferred
	
print(infer_from_model(clf, "What is your name?"))

new_ques = []
new_ans = []

clf.fit(X, y) 

import tkinter as tk
from tkinter import *


master = tk.Tk()

e1 = Entry(master, width = 50)
e2 = Entry(master, width = 15)
e3 = Entry(master, width = 15)

def compute(): 
	ques = e1.get()
	ques_min = ques.split(" ")[0] + " " + ques.split(" ")[1] 	
	ans = infer_from_model(clf, ques_min)
	e2.delete(0, tk.END)
	e2.insert(0, str(ans).replace("[\'","").replace("\']",""))

counter = 0
correct = 0.0

def check(): 
	ques = e1.get()
	global counter
	global correct
	global new_ans
	global new_ques
	counter+=1
	ques_min = ques.split(" ")[0] + " " + ques.split(" ")[1] 	
	ans = infer_from_model(clf, ques_min)
	e2.delete(0, tk.END)
	e2.insert(0, str(ans).replace("[\'","").replace("\']",""))
	act = e3.get()
	if (act in ans) :
		messagebox.showinfo( "Eval", "Correct!")
		correct+=1
	else :
		messagebox.showinfo( "Eval", "Incorrect!")
	new_ques.append(ques_min+"\n")
	new_ans.append(act+"\n")
	
def analytics():
	s = ""
	s = s + "Trials: "+str(counter)+"\n"
	s = s + "Total correct: "+str(correct)+"\n"
	if (counter == 0) :
		s = s + "Percentage correct: 0" + "\n"
	else :
		s = s + "Percentage correct: "+ str(correct/counter * 100) + "\n"
	
	messagebox.showinfo( "Analytics", s)

def retrain():
	f1 = open('questions_dynamic.txt', "a")
	f2 = open('categories_dynamic.txt', "a")
	for ques in new_ques:
		f1.write(ques)
	for ans in new_ans:
		f2.write(ans)
	f1.close()
	f2.close()
	with open('questions_dynamic.txt') as f:
		content = f.readlines()
	f.close()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [x.strip() for x in content] 
	
	with open('categories_dynamic.txt') as f:
		cat = f.readlines()
	f.close()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	cat = [x.strip() for x in cat] 
	vec = []
	for text in content:
		doc = preprocess(text)
		vec.append(document_vector(model, doc))
 
	X = vec 
	y = cat 
	clf.fit(X,y)


# Code to add widgets will go here...
Label(master, text="Question").grid(row=0)
Label(master, text="Type").grid(row=1)
Label(master, text="User Intent").grid(row=2)


e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
e3.grid(row=2, column=1)

Button(master, text='Compute', command=compute).grid(row=4, column=0, sticky=W, pady=4)
Button(master, text='Check', command=check).grid(row=4, column=1, sticky=W, pady=4)
Button(master, text='Analytics', command=analytics).grid(row=4, column=2, sticky=W, pady=4)
Button(master, text='Re-Train', command=retrain).grid(row=4, column=3, sticky=W, pady=4)

master.mainloop()

  

