#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 12:47:00 2018

@author: nick
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


cancer = load_breast_cancer()

#print(cancer.DESCR) # Print the data set description
print(cancer.keys())
#IMPORTANT TO ANALYSE DATA BEFORE ACTUALLY IMPLEMENTING ALGORITHMS

# You should write your whole answer within the function provided. The autograder will call
# this function and compare the return value against the correct solution value
def no_features():
    # This function returns the number of features of the breast cancer dataset, which is an integer. 
    # The assignment question description will tell you the general format the autograder is expecting
    return len(cancer['feature_names'])
# You can examine what your function returns by calling it in the cell. If you have questions
# about the assignment formats, check out the discussion forums for any FAQs


#print(np.concatenate((cancer['data'],cancer['target']),axis=1))


#display data
def disp_data(cancer):
    df=pd.DataFrame(cancer['data'], columns=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                                     'mean smoothness', 'mean compactness', 'mean concavity',
                                     'mean concave points', 'mean symmetry', 'mean fractal dimension',
                                     'radius error', 'texture error', 'perimeter error', 'area error',
                                     'smoothness error', 'compactness error', 'concavity error',
                                     'concave points error', 'symmetry error', 'fractal dimension error',
                                     'worst radius', 'worst texture', 'worst perimeter', 'worst area',
                                     'worst smoothness', 'worst compactness', 'worst concavity',
                                     'worst concave points', 'worst symmetry', 'worst fractal dimension'],index=pd.RangeIndex(start=0,stop=569,step=1))

    df1=pd.DataFrame(cancer['target'],columns=['target'],index=pd.RangeIndex(start=0,stop=569,step=1))
    print(df,df1)
    a=pd.concat((df,df1),axis=1)
    return(a)


dataset=disp_data(cancer)
print(dataset)

#create new variable for labels (y values)
labels=dataset['target']


def class_count():
    #change from boolean to string for interpretability
    def categorise():
        label=labels.replace(value="malignant",to_replace=1).replace(value="benign",to_replace=0)
        return(label)
    label=categorise()
    print(label.rename('target'))

    #count number of occurences of each class (in this case malignant and benign)
    target=label.value_counts()
    return(target)

print(class_count())


#Distinguish between input and output/target values
def x_y_vals():
    X=pd.DataFrame(dataset.iloc[:,:30])
    #remember iloc is integer-based
    y=labels
    return((X,y))

#Split into train and test set with 0 random state (can tweak)
def split():
    X, y = x_y_vals()
    X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0)
    #print(y_train)
    return(X_train, X_test, y_train, y_test)


def fit_data():
    X_train, X_test, y_train, y_test = split()
    #create classifier instance
    knn=KNeighborsClassifier(n_neighbors=50)

    #keep tweaking number of neighbours
    #accuracy increases as n_neighbours increases


    #print(knn.score(X_test,y_test))

    #Fit classifier to data
    return(knn.fit(X_train,y_train))



#cancerdf = answer_one(cancer)
#means = cancerdf.mean()[:-1].values.reshape(1, -1)
#print(means)

#Predict based on average of all features (not really serving the purpose of the classifier)
def mean_pred():
    cancerdf = disp_data(cancer)
    means = cancerdf.mean()[:-1].values.reshape(1,-1)
    print(means)
    knn=fit_data()
    print(knn)
    mean_predict=knn.predict(means)
    
    #changing Boolean to string to make things clear
    if mean_predict == [1]:
        mean_predict="malignant"
    else:
        mean_predict="benign"
    return mean_predict
print("Prediction for average of all features:", mean_pred())

#Predict based on test data
def test_pred():
    X_train, X_test, y_train, y_test = split()
    knn = fit_data()
    test_predict0=knn.predict(X_test)
    test_predict=pd.Series(test_predict0)
    test_predict=test_predict.replace(value="malignant",to_replace=1).replace(value="benign",to_replace=0)
    return(test_predict)
print("Prediction for test data:\n", test_pred() )

def classifier_score():
    X_train, X_test, y_train, y_test = split()
    knn = fit_data()
    knn_score=knn.score(X_test,y_test)
    return (knn_score)

print("Prediction accuracy:", classifier_score())

def accuracy_plot():

    
    X_train, X_test, y_train, y_test = split()

    # Find the training and testing accuracies by target value (i.e. malignant, benign)
    mal_train_X = X_train[y_train==0]
    mal_train_y = y_train[y_train==0]
    ben_train_X = X_train[y_train==1]
    ben_train_y = y_train[y_train==1]

    mal_test_X = X_test[y_test==0]
    mal_test_y = y_test[y_test==0]
    ben_test_X = X_test[y_test==1]
    ben_test_y = y_test[y_test==1]

    knn = fit_data()

    scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y), 
              knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]


    plt.figure()

    # Plot the scores as a bar chart
    bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

    # directly label the score onto the bars
    for bar in bars:
        height = bar.get_height()
        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), 
                     ha='center', color='w', fontsize=11)

    # remove all the ticks (both axes), and tick labels on the Y axis
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

    # remove the frame of the chart
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)

    plt.show()
    return

accuracy_plot()
max_value=[0]
k_optim=[]
k_vals=[]
knn_scores=[]
X_train, X_test, y_train, y_test = split()
for k in range(1, 100):
    knn=KNeighborsClassifier(n_neighbors=k)
    fit=knn.fit(X_train, y_train)
    knn_score = knn.score(X_test, y_test)
    k_vals.append(k)
    knn_scores.append(knn_score)

    #attempt to store max value with its corresponding k value

    if knn_score>max(max_value):
        max_value.append(knn_score)
        k_optim.append(k)
    print("Score for %s neighbours:" %k, knn_score)

print("Optimized k value is {}".format(k_optim[-1]), "with accuracy of : {}".format(max(max_value)))

#make a dataframe to plot values (one way of plotting)
k_df= pd.DataFrame({'k': k_vals, 'scores': knn_scores})
# we use dictionaries in this case to build the dataframe

k_df.plot('k', 'scores', kind='line')

plt.show()



"""
"""

"""
def answer_one(cancer):
    df=pd.DataFrame(np.concatenate((cancer['data'],cancer['target']),axis=1), columns=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                                     'mean smoothness', 'mean compactness', 'mean concavity',
                                     'mean concave points', 'mean symmetry', 'mean fractal dimension',
                                     'radius error', 'texture error', 'perimeter error', 'area error',
                                     'smoothness error', 'compactness error', 'concavity error',
                                     'concave points error', 'symmetry error', 'fractal dimension error',
                                     'worst radius', 'worst texture', 'worst perimeter', 'worst area',
                                     'worst smoothness', 'worst compactness', 'worst concavity',
                                     'worst concave points', 'worst symmetry', 'worst fractal dimension','target'],index=pd.RangeIndex(start=0,stop=569,step=1))
    return(df)
    
""" 

"""answer_zero() 
print(cancer['feature_names'])

print(cancer['data'].shape)
print(cancer['target'].shape)
print(cancer['data'])
print(cancer['target'])
print(type(cancer['data']))"""
