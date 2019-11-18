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

# -------------------------------------------------------------------------------

def read_data_all(strlist):
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

def read_test_data(testdata): # only one sample
    columns = ['score_overall', 'nose_score', 'nose_x', 'nose_y', 'leftEye_score', 'leftEye_x', 'leftEye_y',
               'rightEye_score', 'rightEye_x', 'rightEye_y', 'leftEar_score', 'leftEar_x', 'leftEar_y',
               'rightEar_score', 'rightEar_x', 'rightEar_y', 'leftShoulder_score', 'leftShoulder_x', 'leftShoulder_y',
               'rightShoulder_score', 'rightShoulder_x', 'rightShoulder_y', 'leftElbow_score', 'leftElbow_x',
               'leftElbow_y', 'rightElbow_score', 'rightElbow_x', 'rightElbow_y', 'leftWrist_score', 'leftWrist_x',
               'leftWrist_y', 'rightWrist_score', 'rightWrist_x', 'rightWrist_y', 'leftHip_score', 'leftHip_x',
               'leftHip_y', 'rightHip_score', 'rightHip_x', 'rightHip_y', 'leftKnee_score', 'leftKnee_x', 'leftKnee_y',
               'rightKnee_score', 'rightKnee_x', 'rightKnee_y', 'leftAnkle_score', 'leftAnkle_x', 'leftAnkle_y',
               'rightAnkle_score', 'rightAnkle_x', 'rightAnkle_y']
    data = json.loads(open(testdata, 'r').read())
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
    return test_case

def normalize(data):
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    rng = maxs - mins
    norm_data = 1 - ((maxs - data) / rng)
    return norm_data

def shuffling(x):
 indx = np.arange(x.shape[0])          # create a array with indexes for X data
 np.random.shuffle(indx)
 y = x[indx]
 return y

def lowpassfilt(data):
    # First, design the Buterworth filter
    N = 3  # Filter order
    Wn = 0.9  # Cutoff frequency
    B, A = signal.butter(N, Wn, 'low', output='ba')
    # w, mag, phase = signal.bode((B, A))
    # print(phase)

    # Second, apply the filter
    datafilt = signal.filtfilt(B, A, data, axis=0)
    # print(buy_dict[1].loc[0:2,:])
    datafilt_frm = pd.DataFrame(data=datafilt, columns=data.columns.tolist())
    # print(datafilt_frm.loc[0:2,:])
    return datafilt_frm

def apply_to_dict(data_dict):
    data_applied = {}
    for ky in data_dict.keys():
        # data_dict_norm_temp = normalize(data_dict[ky])
        data_dict_norm_filt_temp = lowpassfilt(data_dict[ky])
        data_applied[ky] = data_dict_norm_filt_temp
    return data_applied

# To create a feature matrix for each category buy, communicate, ...

def feature_category(data_used_featurecreate):
    str='_fft'
    fftcolumnheader = [s+str for s in data_used_featurecreate[1].columns.tolist()]
    fftfeature = pd.DataFrame(columns=fftcolumnheader)

    str='_std'
    stdcolumnheader = [s+str for s in data_used_featurecreate[1].columns.tolist()]
    stdfeature = pd.DataFrame(columns=stdcolumnheader)

    str='_ave'
    avecolumnheader = [s+str for s in data_used_featurecreate[1].columns.tolist()]
    avefeature = pd.DataFrame(columns=avecolumnheader)
    for ky in data_used_featurecreate.keys():
        # fft transform
        y = fft(data_used_featurecreate[ky], axis=0)
        fftpeak = np.round(np.max(abs(y),axis=0),3)
        fftfeature.loc[ky, :] = fftpeak
        # std of data
        sd_temp = np.round(np.std(data_used_featurecreate[ky].to_numpy(), axis=0), 3)
        stdfeature.loc[ky, :] = sd_temp
        # print(stdfeature.isnull().sum().sum())
        ave_temp = np.round(np.mean(data_used_featurecreate[ky].to_numpy(), axis=0), 3)
        avefeature.loc[ky, :] = ave_temp
        # print(avefeature.isnull().sum().sum())

    result = pd.concat([fftfeature, stdfeature, avefeature], axis=1)

    return result

def feature_test(test_case):
    print('test_case',test_case.shape)
    print('test_case', type(test_case))
    print('test_case', test_case.columns.tolist())
    str='_fft'
    fftcolumnheader = [s+str for s in test_case.columns.tolist()]
    str='_std'
    stdcolumnheader = [s+str for s in test_case.columns.tolist()]
    str='_ave'
    avecolumnheader = [s+str for s in test_case.columns.tolist()]
    headers = fftcolumnheader + stdcolumnheader + avecolumnheader

    y = fft(test_case, axis=0)
    fftpeak = np.round(np.max(abs(y),axis=0), 3)
    sd_temp = np.round(np.std(test_case.to_numpy(), axis=0), 3)
    # print(stdfeature.isnull().sum().sum())
    ave_temp = np.round(np.mean(test_case.to_numpy(), axis=0), 3)
    data_temp = np.concatenate((fftpeak, sd_temp, ave_temp), axis=0)
    data = data_temp.reshape(1, len(data_temp))
    # print(type(data))
    # print(data.shape)
    # feature = pd.DataFrame(data, columns=headers)
    # print(feature)
    return data

def classifier_init():

#     No good result for the following classifiers: GaussianNB, MultinomialNB, BernoulliNB, AdaBoost

    LogReg = LogisticRegression(C=1e20, solver='warn', multi_class='ovr', fit_intercept=True, max_iter=5000, intercept_scaling=1)

    KneighberC= KNeighborsClassifier(n_neighbors=6, p=4, weights='distance')

    svm = SVC(C=3.0, kernel='poly', degree=6, gamma='scale', coef0=20.0, shrinking=True, probability=False, tol=0.0001,
            cache_size=300, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)

    Nusvm = NuSVC(nu=0.2, kernel='poly', degree=6, gamma='scale', coef0=19.0, shrinking=True, probability=False, tol=0.0001,
            cache_size=300, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)

    DTC = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=10,
            min_samples_leaf=3, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)

    QDA = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.999, store_covariance=False, tol=0.0001)

    LDA = LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001)

    GradBoostC = GradientBoostingClassifier(loss='deviance', learning_rate=0.05, n_estimators=500, subsample=1.0, criterion='friedman_mse',
            min_samples_split=5, min_samples_leaf=3, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0,
            min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=True,
            presort='auto', validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)

    GaussianProcClass= GaussianProcessClassifier(kernel=5.0 * RBF(2.0), optimizer='fmin_l_bfgs_b', n_restarts_optimizer=4, max_iter_predict=400,
            warm_start=False, copy_X_train=True, random_state=None, multi_class='one_vs_rest', n_jobs=None)

    MLPC = MLPClassifier(hidden_layer_sizes=(150, 70, 6), activation='logistic', solver='adam', alpha=0.0001, batch_size='auto',
        learning_rate='adaptive', learning_rate_init=0.0001, power_t=0.5, max_iter=10000, shuffle=True, random_state=None,
        tol=0.00001, verbose=False, warm_start=True, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)
    RndForC = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=10, min_samples_split=4, min_samples_leaf=2,
    min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
    min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=True, class_weight=None)
    RadiusNeighClass = RadiusNeighborsClassifier(radius=50.0, weights='distance', algorithm='auto', leaf_size=30, p=6, metric='minkowski',
        outlier_label=0, metric_params=None, n_jobs=None)

    # classifiers = [LogReg, KneighberC, svm, Nusvm, DTC, QDA, LDA, GradBoostC, GaussianProcClass, MLPC, RndForC, RadiusNeighClass]
    classifiers = [LogReg, KneighberC, svm, Nusvm]
    return classifiers

def train_ML_models():
    strlist = ['buy', 'communicate', 'fun', 'hope', 'mother', 'really']
    # labels = [0:'buy', 1:'communicate', 2:'fun', 3:'hope', 4:'mother', 5:'really']
    all_data = read_data_all(strlist)
    # creating the labels
    labels = []
    for i, ky in enumerate(all_data.keys()):
        feature_m_temp = feature_category(all_data[ky])
        labels.extend([i] * feature_m_temp.shape[0])
        # print(feature_m_temp.shape)
        # print(len(labels))
        if i == 0:
            feature_matrix = pd.DataFrame(columns=feature_m_temp.columns.tolist())
            feature_matrix = feature_matrix.append(feature_m_temp, ignore_index=True, sort=False)
        else:
            feature_matrix = feature_matrix.append(feature_m_temp, ignore_index=True, sort=False)

    # print(feature_matrix.isnull().sum())
    # print(feature_matrix.shape)
    labels_series = pd.Series(labels, name='class_label')

    # print(labels_series.unique())
    feature_matrix['class_label'] = labels_series
    feature_m_label = feature_matrix.copy()
    # feature_m_label = pd.concat([feature_matrix, labels_series], axis=1, ignore_index=True)

    classifiers = classifier_init()
    names = []

    # df.apply(np.random.permutation, axis=1)
    # df = df.sample(frac=1, axis=1).reset_index(drop=True)
    # feature_m_label.sample(frac=1, axis=0).reset_index(drop=True)
    feature_m_label_shuf = pd.DataFrame(feature_m_label.apply(np.random.permutation, axis=0), columns=feature_m_label.columns.tolist())
    # feature_m_label_shuf = feature_m_label.apply(np.random.permutation, axis=0)
    y_train = feature_m_label_shuf['class_label'].copy()
    x_train = feature_m_label_shuf.iloc[:, 0:-1].copy()

    scaler = StandardScaler()
    scal = scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    trained_models = {}
    for clf in classifiers:
        name = clf.__class__.__name__
        print(name)
        names.append(name)
        trained_models[name] = clf.fit(x_train, y_train)

    filename = 'trained_models.sav'
    pickle.dump(trained_models, open(filename, 'wb'))

    filename = 'scaler.sav'
    pickle.dump(scal, open(filename, 'wb'))
    return trained_models, scal

if __name__ == '__main__':
    train_flag = 1
    if train_flag == 1:
        trained_models, scal = train_ML_models()
    else:
        filename = 'trained_models.sav'
        trained_models_loaded = pickle.load(open(filename, 'rb'))
        filename = 'scaler.sav'
        scaler_loaded = pickle.load(open(filename, 'rb'))
        test_case = read_test_data('BUY_1_BAKRE.json')    # this is a pandas dataframe
        x_test = feature_test(test_case)
        for name in trained_models_loaded.keys():
            print(name)
            y_pred = trained_models_loaded[name].predict(x_test)
            print(y_pred)


    # if len(sys.argv) > 1:
    #     fname = sys.argv[1]
    #     trained_models, scal = train_ML_models()
    #     test_case = read_test_data(fname)    # this is a pandas dataframe
    #     x_test = feature_test (test_case)
    #     for model in train_ML_models:
    #         name = model.__class__.__name__
    #         print(name)
    #         y_pred = model.predict(x_test)
    #         print(y_pred)
    # else:
    #     print('Usage: %s file_name'% sys.argv[0])
