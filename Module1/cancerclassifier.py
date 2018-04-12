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

cancer = load_breast_cancer()

#print(cancer.DESCR) # Print the data set description
print(cancer.keys())
#IMPORTANT TO ANALYSE DATA BEFORE ACTUALLY IMPLEMENTING ALGORITHMS

# You should write your whole answer within the function provided. The autograder will call
# this function and compare the return value against the correct solution value
def answer_zero():
    # This function returns the number of features of the breast cancer dataset, which is an integer. 
    # The assignment question description will tell you the general format the autograder is expecting
    return len(cancer['feature_names'])
# You can examine what your function returns by calling it in the cell. If you have questions
# about the assignment formats, check out the discussion forums for any FAQs


#print(np.concatenate((cancer['data'],cancer['target']),axis=1))

def answer_one(cancer):
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


dataset=answer_one(cancer)
print(dataset)

labels=dataset['target']

def answer_two():
    def categorise():
        label=labels.replace(value="malignant",to_replace=1).replace(value="benign",to_replace=0)
        return(label)
    label=categorise()
    print(label.rename('target'))
    target=label.value_counts()
    return(target)

print(answer_two())



def answer_three():
    X=pd.DataFrame(dataset.iloc[:,:30])
    #remember iloc is integer-based
    y=labels
    return((X,y))

def answer_four():
    X, y = answer_three()
    X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0)
    #print(y_train)
    return(X_train, X_test, y_train, y_test)

def answer_five():
    X_train, X_test, y_train, y_test = answer_four()
    knn=KNeighborsClassifier(n_neighbors=1)
    #print(knn.score(X_test,y_test))
    # Your code here
    return(knn.fit(X_train,y_train))



#cancerdf = answer_one(cancer)
#means = cancerdf.mean()[:-1].values.reshape(1, -1)
#print(means)

def answer_six():
    cancerdf = answer_one(cancer)
    means = cancerdf.mean()[:-1].values.reshape(1,-1)
    print(means.shape)
    knn=answer_five()
    print(knn)
    mean_predict=knn.predict(means)
    
    #changing Boolean to string to make things clear
    if mean_predict == [1]:
        mean_predict="malignant"
    else:
        mean_predict="benign"
    return mean_predict
print("Prediction for average of all features:", answer_six())

def answer_seven():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    test_predict0=knn.predict(X_test)
    test_predict=pd.Series(test_predict0)
    test_predict=test_predict.replace(value="malignant",to_replace=1).replace(value="benign",to_replace=0)
    return(test_predict)
print("Prediction for test data:\n", answer_seven() )

def answer_eight():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    knn_score=knn.score(X_test,y_test)
    return (knn_score)

print("Prediction accuracy:", answer_eight())

def accuracy_plot():
    import matplotlib.pyplot as plt
    
    X_train, X_test, y_train, y_test = answer_four()

    # Find the training and testing accuracies by target value (i.e. malignant, benign)
    mal_train_X = X_train[y_train==0]
    mal_train_y = y_train[y_train==0]
    ben_train_X = X_train[y_train==1]
    ben_train_y = y_train[y_train==1]

    mal_test_X = X_test[y_test==0]
    mal_test_y = y_test[y_test==0]
    ben_test_X = X_test[y_test==1]
    ben_test_y = y_test[y_test==1]

    knn = answer_five()

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
    
    return

accuracy_plot()



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
