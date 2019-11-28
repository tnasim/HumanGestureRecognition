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

def edistance(x1, x2, y1, y2): # inputs are pandas series
    # this function will accept 4 vectors x1,y1, x2,y2 of the same length and calculate the euclidean distance[sqrt((x1-x2)^2 + (y1-y2)^2)] of them and return a vector of the same length
    x_ = x1-x2
    x = pow(x_, 2)
    y_ = y1-y2
    y = pow(y_, 2)
    return pd.Series(np.round(np.sqrt(x+y), 3))

def sample_normalization(sample):

    scale_x = edistance(sample['leftHip_x'], sample['rightHip_x'], sample['leftHip_y'], sample['rightHip_y'])
    x_hip_middle = (sample['leftHip_x']+ sample['rightHip_x'])/2.000
    y_hip_middle = (sample['leftHip_y'] + sample['rightHip_y'])/2.000
    scale_y = edistance (sample['nose_x'],x_hip_middle ,sample['nose_y'], y_hip_middle)
    norm_sample = sample.copy()

    collist_x = ['nose_x',  'leftShoulder_x', 'rightShoulder_x', 'leftElbow_x','rightElbow_x', 'leftWrist_x','rightWrist_x']
    collist_y = ['nose_y', 'leftShoulder_y','rightShoulder_y','leftElbow_y', 'rightElbow_y', 'leftWrist_y', 'rightWrist_y']
    for colname_x in collist_x:
        str = 'norm_'
        norm_sample[str+colname_x] = (norm_sample[colname_x]-norm_sample['nose_x'])/scale_x
    for colname_y in collist_y:
        norm_sample[str+colname_y] = (norm_sample[colname_y]-norm_sample['nose_y'])/scale_y
    return norm_sample

def get_feature_foroneclass(data): # data_used_featurecreate is a dictionay of users of each sign(class)

  # To create a feature matrix for each category buy, communicate, ...

    # colhead= ['score_overall', 'nose_score', 'nose_x', 'nose_y', 'leftEye_score', 'leftEye_x', 'leftEye_y',
    #            'rightEye_score', 'rightEye_x', 'rightEye_y', 'leftEar_score', 'leftEar_x', 'leftEar_y',
    #            'rightEar_score', 'rightEar_x', 'rightEar_y', 'leftShoulder_score', 'leftShoulder_x', 'leftShoulder_y',
    #            'rightShoulder_score', 'rightShoulder_x', 'rightShoulder_y', 'leftElbow_score', 'leftElbow_x',
    #            'leftElbow_y', 'rightElbow_score', 'rightElbow_x', 'rightElbow_y', 'leftWrist_score', 'leftWrist_x',
    #            'leftWrist_y', 'rightWrist_score', 'rightWrist_x', 'rightWrist_y', 'leftHip_score', 'leftHip_x',
    #            'leftHip_y', 'rightHip_score', 'rightHip_x', 'rightHip_y', 'leftKnee_score', 'leftKnee_x', 'leftKnee_y',
    #            'rightKnee_score', 'rightKnee_x', 'rightKnee_y', 'leftAnkle_score', 'leftAnkle_x', 'leftAnkle_y',
    #            'rightAnkle_score', 'rightAnkle_x', 'rightAnkle_y']

    for ky in data.keys():
        data[ky]['Hipx_baseline'] = (data[ky]['leftHip_x'] + data[ky]['rightHip_x'])/ 2.0
        data[ky]['Hipy_baseline'] = (data[ky]['leftHip_y'] + data[ky]['rightHip_y'])/ 2.0

        # wrist hip distance
        x = pow((data[ky]['leftWrist_x'] - data[ky]['Hipx_baseline']), 2)
        y = pow((data[ky]['leftWrist_y'] - data[ky]['Hipy_baseline']), 2)
        data[ky]['leftWrist_hip_distance'] = pd.Series(np.round(np.sqrt(x+y),3))

        x = pow((data[ky]['rightWrist_x'] - data[ky]['Hipx_baseline']), 2)
        y = pow((data[ky]['rightWrist_y'] - data[ky]['Hipy_baseline']), 2)
        data[ky]['rightWrist_hip_distance'] = pd.Series(np.round(np.sqrt(x+y),3))

        # wrist nose distance
        x = pow((data[ky]['leftWrist_x'] - data[ky]['nose_x']), 2)
        y = pow((data[ky]['leftWrist_y'] - data[ky]['nose_y']), 2)
        data[ky]['leftWrist_nose_distance'] = pd.Series(np.round(np.sqrt(x+y),3))

        x = pow((data[ky]['rightWrist_x'] - data[ky]['nose_x']), 2)
        y = pow((data[ky]['rightWrist_y'] - data[ky]['nose_y']), 2)
        data[ky]['rightWrist_nose_distance'] = pd.Series(np.round(np.sqrt(x+y),3))

        # Elbow hip distance
        x = pow((data[ky]['leftElbow_x'] - data[ky]['Hipx_baseline']), 2)
        y = pow((data[ky]['leftElbow_y'] - data[ky]['Hipy_baseline']), 2)
        data[ky]['leftElbow_hip_distance']= pd.Series(np.round(np.sqrt(x+y),3))

        x = pow((data[ky]['rightElbow_x'] - data[ky]['Hipx_baseline']), 2)
        y = pow((data[ky]['rightElbow_y'] - data[ky]['Hipy_baseline']), 2)
        data[ky]['rightElbow_hip_distance']= pd.Series(np.round(np.sqrt(x+y),3))

        # Elbow nose distance
        x = pow((data[ky]['leftElbow_x'] - data[ky]['nose_x']), 2)
        y = pow((data[ky]['leftElbow_y'] - data[ky]['nose_y']), 2)
        data[ky]['leftElbow_nose_distance']= pd.Series(np.round(np.sqrt(x+y),3))

        x = pow((data[ky]['rightElbow_x'] - data[ky]['nose_x']), 2)
        y = pow((data[ky]['rightElbow_y'] - data[ky]['nose_y']), 2)
        data[ky]['rightElbow_nose_distance']= pd.Series(np.round(np.sqrt(x+y),3))

    collist = ['leftWrist_hip_distance','rightWrist_hip_distance','leftWrist_nose_distance', 'rightWrist_nose_distance', 'leftElbow_hip_distance','rightElbow_hip_distance', 'leftElbow_nose_distance', 'rightElbow_nose_distance']
    features = pd.DataFrame(columns=collist)
    for ky in data.keys():
        startframe = 0
        endframe = data[ky].shape[0]
        rowlist = range(startframe, endframe)
        data_selected_for_featurecreation = data[ky].loc[rowlist, collist].copy()
        feature_step = pd.Series(np.round(np.mean(data_selected_for_featurecreation, axis=0), 3))
        # features.loc[ky, :] = feature_step
        features = features.append(feature_step, ignore_index=True)
    # print(features.shape)
    # print(features.columns.tolist())
    return features, data

def classifier_init():

#     No good result for the following classifiers: GaussianNB, MultinomialNB, BernoulliNB, AdaBoost

    LogReg = LogisticRegression(C=1e20, solver='warn', multi_class='ovr', fit_intercept=True, max_iter=5000, intercept_scaling=1)

    Nusvm = NuSVC(nu=0.2, kernel='poly', degree=6, gamma='scale', coef0=19.0, shrinking=True, probability=False, tol=0.0001,
            cache_size=300, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)

    MLPC = MLPClassifier(hidden_layer_sizes=(150, 70, 6), activation='logistic', solver='adam', alpha=0.0001, batch_size='auto',
        learning_rate='adaptive', learning_rate_init=0.0001, power_t=0.5, max_iter=10000, shuffle=True, random_state=None,
        tol=0.00001, verbose=False, warm_start=True, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)

    RndForC = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=10, min_samples_split=4, min_samples_leaf=2,
    min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
    min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=True, class_weight=None)
    RadiusNeighClass = RadiusNeighborsClassifier(radius=50.0, weights='distance', algorithm='auto', leaf_size=30, p=6, metric='minkowski',
        outlier_label=0, metric_params=None, n_jobs=None)

    KneighberC= KNeighborsClassifier(n_neighbors=6, p=4, weights='distance')

    svm = SVC(C=3.0, kernel='poly', degree=6, gamma='scale', coef0=20.0, shrinking=True, probability=False, tol=0.0001,
            cache_size=300, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)

    DTC = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=10,
            min_samples_leaf=3, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)

    QDA = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.999, store_covariance=False, tol=0.0001)
    #
    LDA = LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001)

    GradBoostC = GradientBoostingClassifier(loss='deviance', learning_rate=0.05, n_estimators=500, subsample=1.0, criterion='friedman_mse',
            min_samples_split=5, min_samples_leaf=3, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0,
            min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=True,
            presort='auto', validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)

    GaussianProcClass= GaussianProcessClassifier(kernel=5.0 * RBF(2.0), optimizer='fmin_l_bfgs_b', n_restarts_optimizer=4, max_iter_predict=400,
            warm_start=False, copy_X_train=True, random_state=None, multi_class='one_vs_rest', n_jobs=None)

    classifiers = [LogReg, KneighberC, svm, Nusvm, DTC, QDA, LDA, GradBoostC, GaussianProcClass, MLPC, RndForC, RadiusNeighClass]
    # classifiers = [LogReg, Nusvm, LDA, MLPC]
    return classifiers

def train_ML_models():
    # labels = [0:'buy', 1:'communicate', 2:'fun', 3:'hope', 4:'mother', 5:'really']
    all_data = read_data_all()

    # # ---------------------------- data plots for data visualization ---------------------------------------
    # buy, data_buy = feature_foroneclass(all_data['buy'])
    # communicate, data_communicate = feature_foroneclass(all_data['communicate'])
    # fun, data_fun = feature_foroneclass(all_data['fun'])
    # hope, data_hope = feature_foroneclass(all_data['hope'])
    # mother, data_mother = feature_foroneclass(all_data['mother'])
    # really, data_really = feature_foroneclass(all_data['really'])

    # Make plots
    # fig = plt.figure()
    # ax1 = fig.add_subplot(211)
    # plt.plot(communicate['leftWrist_hip_distance'], 'r-')
    # j = 20
    # print('data_buy',data_buy[j].shape)
    # print('data_communicate',data_communicate[j].shape)
    # print('data_fun',data_fun[j].shape)
    # print('data_hope',data_hope[j].shape)
    # print('data_mother',data_mother[j].shape)
    # print('data_really',data_really[j].shape)
    # plt.plot(data_buy[j]['leftWrist_hip_distance'], color='blue')
    # plt.plot(data_communicate[j]['leftWrist_hip_distance'], color='pink')
    # plt.plot(data_fun[j]['leftWrist_hip_distance'], color='green')
    # plt.plot(data_hope[j]['leftWrist_hip_distance'], color='magenta')
    # plt.plot(data_mother[j]['leftWrist_hip_distance'], color='cyan')
    # plt.plot(data_really[j]['leftWrist_hip_distance'], color='purple')
    # plt.title("leftWrist_hip_distance")
    # ax2 = fig.add_subplot(212)
    # plt.plot(data_buy[j]['leftWrist_nose_distance'], color='blue')
    # plt.plot(data_communicate[j]['leftWrist_nose_distance'], color='pink')
    # plt.plot(data_fun[j]['leftWrist_nose_distance'], color='green')
    # plt.plot(data_hope[j]['leftWrist_nose_distance'], color='magenta')
    # plt.plot(data_mother[j]['leftWrist_nose_distance'], color='cyan')
    # plt.plot(data_really[j]['leftWrist_nose_distance'], color='purple')
    # # plt.ylabel("Temperature (oC)")
    # plt.legend(['buy', 'communicate', 'fun', 'hope', 'mother', 'really'])
    # plt.title("leftWrist_nose_distance")
    # ax1.axes.get_xaxis().set_visible(False)
    # plt.show()

    # # Make plots
    # fig = plt.figure()
    # ax1 = fig.add_subplot(211)
    # # plt.plot(communicate['leftWrist_hip_distance'], 'r-')
    # name = 'rightElbow_hip_distance'
    # plt.plot(buy[name], color='blue')
    # plt.plot(communicate[name], color='pink')
    # plt.plot(fun[name], color='green')
    # plt.plot(hope[name], color='magenta')
    # plt.plot(mother[name], color='cyan')
    # plt.plot(really[name], color='purple')
    # plt.title(name)
    # ax2 = fig.add_subplot(212)
    # name2= 'rightWrist_nose_distance'
    # plt.plot(buy[name2], color='blue')
    # plt.plot(communicate[name2], color='pink')
    # plt.plot(fun[name2], color='green')
    # plt.plot(hope[name2], color='magenta')
    # plt.plot(mother[name2], color='cyan')
    # plt.plot(really[name2], color='purple')
    # # plt.ylabel("Temperature (oC)")
    # plt.legend(['buy', 'communicate', 'fun', 'hope', 'mother', 'really'])
    # plt.title(name2)
    # ax1.axes.get_xaxis().set_visible(False)
    # plt.show()
    # -----------------------------------------------------------------------------------------------------------
    # creating the labels
    labels = []
    for i, ky in enumerate(all_data.keys()):
        feature_m_temp, data = feature_foroneclass(all_data[ky])
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
    feature_m_label = feature_matrix.copy()
    feature_m_label['class_label'] = labels_series
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
