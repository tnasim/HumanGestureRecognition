import pandas as pd
import numpy as np
import glob
import sys
import json, pickle
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift
from numpy import linalg as LA
import copy
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from operator import itemgetter
import pickle as pkl
import warnings
warnings.filterwarnings("ignore")


# -------------------------------------------------------------------------------

def read_data_all():

    strlist = ['buy', 'communicate', 'fun', 'hope', 'mother', 'really']
    all_data = {}
    for str in strlist:
        path = r"C:\Users\smousav9\Dropbox (ASU)\Semester 5- Fall 2019- Starting August 22th\CSE 535_Mobile Computing\CSE 535- Assignment2\CSV_data_Tuesday\\" + str
        # path = r"D:\Dropbox (ASU)\Semester 5- Fall 2019- Starting August 22th\CSE 535_Mobile Computing\CSE 535- Assignment2\CSV_data_Tuesday\\" + str
        # path = r'/home/local/ASUAD/tnasim/Documents/Courses/CSE 535/Assignments/Assignment2/CSV/' + str
        all_files = glob.glob(path + "/*.csv")
        data_dict = {}
        for i, filename in enumerate(all_files):
            df = pd.read_csv(filename, index_col=None, header=0)
            data_dict[i] = df.drop('Frames#', axis=1)    #df.iloc[:,0]
        all_data[str] = data_dict
    return all_data

def feature_creation(all_data):
    feature_label = []
    for i, ky in enumerate(all_data.keys()):
        for smpl_num in all_data[ky].keys():
            if len(all_data[ky][smpl_num])>=120:
                mean_ = np.round(np.mean(all_data[ky][smpl_num].to_numpy()[5:120,:], axis=0), 3).tolist()
            else:
                mean_ = np.round(np.mean(all_data[ky][smpl_num].to_numpy()[5:-1,:], axis=0), 3).tolist()
            label_ = [i]
            feature_label.append((mean_+ label_))
    feature_label = np.asarray(feature_label)
    return feature_label


if __name__ == '__main__':
    all_data = read_data_all()
    feature_label = feature_creation(all_data)
    X = feature_label[:, :-1]
    y = feature_label[:, -1]
    skf = StratifiedKFold(n_splits=8)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42, shuffle=True)
    classifiers = [
        LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', C=1e20, max_iter=5000),
        KNeighborsClassifier(20),
        SVC(kernel="rbf", C=0.025, probability=True),
        NuSVC(probability=True),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        GradientBoostingClassifier(loss='deviance', learning_rate=0.02, n_estimators=500,
            min_samples_split=5, min_samples_leaf=3, max_depth=3, warm_start=True),
        GaussianProcessClassifier(kernel=5.0 * RBF(2.0), optimizer='fmin_l_bfgs_b', n_restarts_optimizer=4, max_iter_predict=400,
            warm_start=False, copy_X_train=True, random_state=None, multi_class='one_vs_rest', n_jobs=None),
        # activation = 'identity', 'logistic', 'tanh', 'relu'
        MLPClassifier(hidden_layer_sizes=(150, 70, 1), activation='tanh', learning_rate='adaptive', learning_rate_init=0.0001,
                max_iter=10000, tol=0.00001, warm_start=True, n_iter_no_change=20),
        RadiusNeighborsClassifier(radius=50.0, weights='distance', leaf_size=30, p=6, outlier_label=0)]
    accuracy_scores_dict = {}
    precision_scores_dict = {}
    recall_scores_dict = {}
    f1_scores_dict = {}
    trained_classifiers = {}
    for clf in classifiers:
        classifier_name = clf.__class__.__name__
        clf.fit(X_train, y_train)
        trained_classifiers[classifier_name] = clf
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='micro')
        recall = recall_score(y_test, y_pred, average='micro')
        f1 = f1_score(y_test, y_pred, average='micro')
        accuracy_scores_dict[classifier_name] = accuracy
        precision_scores_dict[classifier_name] = precision
        recall_scores_dict[classifier_name] = recall
        f1_scores_dict[classifier_name] = f1
        print('------------------------------------------------')
        print(classifier_name)
        print('Classification performance')
        print("Accuracy: {:.4%}".format(accuracy))
        print("Precision: {:.4%}".format(precision))
        print("Recall: {:.4%}".format(recall))
        print("F1: {:.4%}".format(f1))
        print("------------------------------------------------")

    classifiers_sorted_by_accuracy = sorted(accuracy_scores_dict.items(), key=itemgetter(1), reverse=True)
    top_4 = []
    for classifier in classifiers_sorted_by_accuracy[:4]:
        top_4.append(list(classifier)[0])

    top_4classifiers = {}
    for classifier_name in top_4:
        top_4classifiers[classifier_name] = trained_classifiers[classifier_name]
        print(classifier_name)

    pkl.dump(top_4classifiers, open('trained_classifier.pkl', 'wb'))

    # # plot of accuracy
    # accuracy_dict_sorted = dict(sorted(accuracy_scores_dict.items(), key=itemgetter(1), reverse=True))
    # fig = plt.figure()
    # fig.suptitle('Classifiers Performance')
    # ax1 = fig.add_subplot(221)
    # objects = tuple(accuracy_dict_sorted.keys())
    # y_pos = np.arange(len(objects))
    # performance = accuracy_dict_sorted.values()
    # plt.barh(y_pos, performance, align='center', alpha=0.5)
    # # *zip(*accuracy_dict_sorted.items())
    # plt.yticks(y_pos, objects)
    # plt.xlabel('Accuracy')
    #
    # # plot of precision
    # precision_dict_sorted = dict(sorted(precision_scores_dict.items(), key=lambda x: x[1], reverse=True))
    #     # dict(sorted(precision_scores_dict.items(), key=itemgetter(1), reverse=True))
    # ax2 = fig.add_subplot(222)
    # objects = tuple(precision_dict_sorted.keys())
    # y_pos = np.arange(len(objects))
    # performance = precision_dict_sorted.values()
    # plt.barh(y_pos, performance, align='center', alpha=0.5)
    # plt.yticks(y_pos, objects)
    # plt.xlabel('Rrecision')
    #
    # # plot of recall
    # recall_dict_sorted = dict(sorted(recall_scores_dict.items(), key=lambda x: x[1], reverse=True))
    # ax3 = fig.add_subplot(223)
    # objects = tuple(recall_dict_sorted.keys())
    # y_pos = np.arange(len(objects))
    # performance = recall_dict_sorted.values()
    # plt.barh(y_pos, performance, align='center', alpha=0.5)
    # plt.yticks(y_pos, objects)
    # plt.xlabel('Recall')
    #
    # # plot of F1
    # f1_dict_sorted = dict(sorted(f1_scores_dict.items(), key=lambda x:x[1], reverse=True))
    # ax4 = fig.add_subplot(224)
    # objects = tuple(f1_dict_sorted.keys())
    # y_pos = np.arange(len(objects))
    # performance = f1_dict_sorted.values()
    # plt.barh(y_pos, performance, align='center', alpha=0.5)
    # plt.yticks(y_pos, objects)
    # plt.xlabel('F1-score')
    # plt.show()

