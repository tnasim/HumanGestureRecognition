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

# ---------------------------------------------------------------------------------

def read_test_data(data): # only one sample
    columns = ['score_overall', 'nose_score', 'nose_x', 'nose_y', 'leftEye_score', 'leftEye_x', 'leftEye_y',
               'rightEye_score', 'rightEye_x', 'rightEye_y', 'leftEar_score', 'leftEar_x', 'leftEar_y',
               'rightEar_score', 'rightEar_x', 'rightEar_y', 'leftShoulder_score', 'leftShoulder_x', 'leftShoulder_y',
               'rightShoulder_score', 'rightShoulder_x', 'rightShoulder_y', 'leftElbow_score', 'leftElbow_x',
               'leftElbow_y', 'rightElbow_score', 'rightElbow_x', 'rightElbow_y', 'leftWrist_score', 'leftWrist_x',
               'leftWrist_y', 'rightWrist_score', 'rightWrist_x', 'rightWrist_y', 'leftHip_score', 'leftHip_x',
               'leftHip_y', 'rightHip_score', 'rightHip_x', 'rightHip_y', 'leftKnee_score', 'leftKnee_x', 'leftKnee_y',
               'rightKnee_score', 'rightKnee_x', 'rightKnee_y', 'leftAnkle_score', 'leftAnkle_x', 'leftAnkle_y',
               'rightAnkle_score', 'rightAnkle_x', 'rightAnkle_y']
    csv_data = np.zeros((len(data), len(columns)))
    for i in range(csv_data.shape[0]):
        one = []
        one.append(data[i]['score'])
        for obj in data[i]['keypoints']:
            one.append(obj['score'])
            one.append(obj['position']['x'])
            one.append(obj['position']['y'])
        csv_data[i] = np.array(one)
    test_case= pd.DataFrame(csv_data, columns=columns)
    return test_case # the format is pandas dataframe

def feature_test(test_case):
    y = fft(test_case, axis=0)
    fftpeak = np.round(np.max(abs(y),axis=0), 3)
    sd_temp = np.round(np.std(test_case.to_numpy(), axis=0), 3)
    # print(stdfeature.isnull().sum().sum())
    ave_temp = np.round(np.mean(test_case.to_numpy(), axis=0), 3)
    data_temp = np.concatenate((fftpeak, sd_temp, ave_temp), axis=0)
    data = data_temp.reshape(1, len(data_temp))
    return data  # the format of data is a nparray

def predict_gesture(json_object):
    filename = 'trained_models.sav'
    trained_models_loaded = pickle.load(open(filename, 'rb'))
    filename = 'scaler.sav'
    scaler_loaded = pickle.load(open(filename, 'rb'))
    test_case = read_test_data(json_object)
    x_test = feature_test(test_case)
    x_test = scaler_loaded.transform(x_test)
    for name in trained_models_loaded.keys():
        print(name)
        y_pred = trained_models_loaded[name].predict(x_test)
        print(y_pred)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        fname = sys.argv[1]
        data = json.loads(open(fname, 'r').read())
        predict_gesture(data)
    else:
        print('Usage: %s file_name'% sys.argv[0])

