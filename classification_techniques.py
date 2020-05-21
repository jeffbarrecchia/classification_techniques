#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:38:58 2020

@author: jeffbarrecchia
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split as ttSplit
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

diabetes = pd.read_csv('~/Documents/Kaggle_Projects/diabetes.csv')
# print(diabetes.info())
# print(diabetes.groupby('Outcome').size())
# sb.countplot('Outcome', data = diabetes)

diabetes_x = diabetes.drop(columns = ['Outcome'])
diabetes_y = diabetes['Outcome']

x_train, x_test, y_train, y_test = ttSplit(diabetes_x, diabetes_y, train_size = 0.8, stratify = diabetes['Outcome'], random_state = 4)

# =============================================================================
# Using K Nearest Neighbor Classification
# 
# Results in a 75% accuracy for the Training Set and a 74.7% accuracy for the 
# Test Set
# =============================================================================

neighbor_settings = range(1, 100)

training_acc = []
test_acc = []

for n_neighbors in neighbor_settings:
    knc = KNeighborsClassifier(n_neighbors = n_neighbors)
    knc.fit(x_train, y_train)
    training_acc.append(knc.score(x_train, y_train))
    test_acc.append(knc.score(x_test, y_test))
    
plt.plot(neighbor_settings, training_acc, label = 'Training Accuracy')
plt.plot(neighbor_settings, test_acc, label = 'Test Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('n_neighbors')
plt.legend()

# =============================================================================
# Using Decision Tree Classification
# 
# Achieves a 79.6% accuracy on the training dataset and a 69.5% accuracy 
# on the test dataset
# =============================================================================

# tree_class = DecisionTreeClassifier(max_depth = 3, random_state = 0)
# tree_class.fit(x_train, y_train)

# print('\nTraining accuracy: {:.3f}'.format(tree_class.score(x_train, y_train)))
# print('\nTest accuracy: {:.3f}'.format(tree_class.score(x_test, y_test)))

# =============================================================================
# Finds the important factors being measured that are related to diabetes
# 
# Based off of this, there are three important factors related to diabetes:
#     1. Glucose
#     2. BMI
#     3. Age
# =============================================================================

def importantFeaturesOfDiabetes(model):
    plt.figure(figsize = (20, 10))
    numFeatures = 8
    plt.barh(range(numFeatures), model.feature_importances_, align = 'center')
    plt.yticks(np.arange(numFeatures), diabetes[0:8])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')

# importantFeaturesOfDiabetes(tree_class)

# =============================================================================
# Uses RandomForestClassifier to predict who is diabetic and who is not
# 
# Based off of this classifier, we get the same three important factors as the
# previous classification method as well: Glucose, BMI, and Age
# =============================================================================

# rfc = RandomForestClassifier(max_depth = 2, n_estimators = 100, random_state = 10)
# rfc.fit(x_train, y_train)

# print('\nTraining accuracy: {:.3f}'.format(rfc.score(x_train, y_train)))
# print('\nTest accuracy: {:.3f}'.format(rfc.score(x_test, y_test)))

# importantFeaturesOfDiabetes(rfc)

# =============================================================================
# Uses RandomForestClassifier to predict who is diabetic and who is not
# 
# Gets the same important features: Glucose, BMI, and Age
# =============================================================================

# grad = GradientBoostingClassifier(max_depth = 5, learning_rate = 0.005)
# grad.fit(x_train, y_train)

# print('Training accuracy: {:.3f}'.format(grad.score(x_train, y_train)))
# print('Test accuracy: {:.3f}'.format(grad.score(x_test, y_test)))

# importantFeaturesOfDiabetes(grad)

# =============================================================================
# Support Vector Classification
# =============================================================================

# svc = SVC()
# svc.fit(x_train, y_train)

# print('Training Accuracy: {:.3f}'.format(svc.score(x_train, y_train)))
# print('Test Accuracy: {:.3f}'.format(svc.score(x_test, y_test)))

