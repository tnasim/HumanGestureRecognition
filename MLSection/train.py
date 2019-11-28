# -*- coding: utf-8 -*-
from tqdm import tqdm
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
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
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
import operator
import json
import pickle as pkl
from random import shuffle

def read_data():
    
    data_dir = 'data'
    path_to_folder = [os.path.join(data_dir, folder) for folder in os.listdir(data_dir)]
    
    path_to_files = []
    for folder in path_to_folder:
        temp = [os.path.join(folder, files) for files in os.listdir(folder)]
        path_to_files.append(temp)
    
    data_by_activities = {}
    print(path_to_files)
    for folder in tqdm(path_to_files):
        activity = folder[0].split('\\')[1]
        data_by_activities[activity] = []
        for file in folder:
            data_by_activities[activity].append(pd.read_csv(file))
            
    return data_by_activities

def get_features(data_by_activities, label_map):

    features = {}
    for activity in tqdm(list(data_by_activities.keys())):
        frame_list = data_by_activities[activity]
        features[activity] = []
        for frame in frame_list:
            mean_ = frame.iloc[:,1:].mean().tolist()
            features[activity].append(mean_)
            
    X = []
    y = []
    for activity in tqdm(list(label_map.keys())):
        for item in features[activity]:
            X.append(item)            
            y.append(label_map[activity])
    X = np.asarray(X)
    y = np.asarray(y)
    
    return X, y

if __name__ == "__main__":
    
    data_by_activities = read_data()
    label_map = {label:i for i,label in enumerate(list(data_by_activities.keys()))}
    label_to_class_map = {i:label for i,label in enumerate(list(data_by_activities.keys()))}
    
    train_activites = {}
    test_activities = {}
    test_size = 0.20
    np.random.seed(42)
    for action in list(data_by_activities.keys()):
        list_ = data_by_activities[action]
        shuffle(list_)
        index = int(len(list_) * (1-test_size))
        train_activites[action] = list_[:index]
        test_activities[action] = list_[index:]
        
    X_train, y_train = get_features(train_activites, label_map)
    X_test, y_test = get_features(test_activities, label_map)
    
    classifiers = [
    LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial'),
    KNeighborsClassifier(20),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]
    
    results = {}
    
    trained_classifier = {}
    for clf in classifiers:
        acc = []
        ll = []
        name = clf.__class__.__name__
        clf.fit(X_train, y_train)
        trained_classifier[name] = clf
            
# =============================================================================
#     pkl.dump(trained_classifier, open('classifier.pkl', 'wb'))
#     trained_classifier = None
#     trained_classifier = pkl.load(open('classifier.pkl', 'rb'))
# =============================================================================
    
    for classifier in list(trained_classifier.keys()):
        
        clf = trained_classifier[classifier]
        train_predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, train_predictions)
        
        print("="*30)
        print(classifier)
        
        print('****Results****')
        print("Accuracy: {:.4%}".format(accuracy))
        results[classifier] = accuracy
        print("="*30)        
    
    
    classifier_sorted_by_performance = sorted(results.items(), key=operator.itemgetter(1), reverse = True)        
    top_4 = []
    for classifier in classifier_sorted_by_performance[:4]:
        top_4.append(list(classifier)[0])
    
    top_classifier = {}
    for classifier in top_4:
        top_classifier[classifier] = trained_classifier[classifier]
    
    pkl.dump(label_map, open('label_map.pkl', 'wb'))
    pkl.dump(label_to_class_map, open('label_to_class_map.pkl', 'wb'))
    pkl.dump(top_classifier, open('classifier.pkl', 'wb'))
