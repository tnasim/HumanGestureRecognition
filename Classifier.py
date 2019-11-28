#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import  log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import warnings
warnings.filterwarnings("ignore")


class Classifier(object):
    
    def __init__(self):
        
        self.classifiers = pkl.load(open('classifier.pkl', 'rb'))
        self.classifier_name = list(self.classifiers.keys())
        self.label_to_class_map = pkl.load(open('label_to_class_map.pkl', 'rb'))
        
    def classify(self, frame):
        
        feature = frame.mean().values.reshape(1, -1)
        predicted = []
        for classifier_ in self.classifier_name:
            clf = self.classifiers[classifier_]
            predict = clf.predict(feature)
            class_ = self.label_to_class_map[predict[0]]
            predicted.append(class_)
        return {i+1: label for i,label in enumerate(predicted)}
        