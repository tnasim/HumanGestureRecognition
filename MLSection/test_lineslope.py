from __future__ import division
import pandas as pd
import numpy as np
import glob
import sys
import json
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift
import copy
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
import warnings
warnings.filterwarnings("ignore")
import pickle as pkl
from tqdm import tqdm


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
    return pd.Series(np.round(np.sqrt(x+y),3))

def get_feature_onesample(sample): # creates a one row feature for each sample

    scale_x = edistance(sample['leftHip_x'], sample['rightHip_x'], sample['leftHip_y'], sample['rightHip_y'])
    x_hip_middle = (sample['leftHip_x']+ sample['rightHip_x'])/2.000
    y_hip_middle = (sample['leftHip_y'] + sample['rightHip_y'])/2.000
    scale_y = edistance (sample['nose_x'],x_hip_middle ,sample['nose_y'], y_hip_middle)
    norm_sample = sample.copy()

    collist_x = ['nose_x', 'leftShoulder_x', 'rightShoulder_x', 'leftElbow_x','rightElbow_x', 'leftWrist_x', 'rightWrist_x']
    collist_y = ['nose_y','leftShoulder_y','rightShoulder_y','leftElbow_y', 'rightElbow_y', 'leftWrist_y', 'rightWrist_y']
    for colname_x in collist_x:
        str = 'norm_'
        norm_sample[str+colname_x] = (norm_sample[colname_x]-norm_sample['nose_x'])/scale_x
    for colname_y in collist_y:
        norm_sample[str+colname_y] = (norm_sample[colname_y]-norm_sample['nose_y'])/scale_y
    feature_one_sample = pd.DataFrame()
    feature_one_sample['leftwrist_nose_slope'] = np.divide((norm_sample['norm_rightWrist_y'] - norm_sample['norm_nose_y']) , (norm_sample['norm_leftWrist_x'] - norm_sample['norm_nose_x']))
    feature_one_sample['rightwrist_nose_slope'] = np.divide((norm_sample['norm_rightWrist_y'] - norm_sample['norm_nose_y']),(norm_sample['norm_rightWrist_x'] - norm_sample['norm_nose_x']))
    feature_one_sample['leftElbow_nose_slope'] = np.divide((norm_sample['norm_leftElbow_y'] - norm_sample['norm_nose_y']),(norm_sample['norm_leftElbow_x'] - norm_sample['norm_nose_x']))
    feature_one_sample['rightElbow_nose_slope'] = np.divide((norm_sample['norm_rightElbow_y'] - norm_sample['norm_nose_y']),(norm_sample['norm_rightElbow_x'] - norm_sample['norm_nose_x']))
    feature_one_sample['rightwrist_rightshoulder_slope'] = np.divide((norm_sample['norm_rightWrist_y'] - norm_sample['norm_rightShoulder_x']),(norm_sample['norm_rightWrist_x'] - norm_sample['norm_rightShoulder_y']))
    if sample.shape[0]> 125:
        feature_one_sample['rightwrist_rightshoulder_distance'] = edistance(norm_sample['norm_rightWrist_x'][5:125],norm_sample['norm_rightShoulder_x'][5:125],norm_sample['norm_rightWrist_y'][5:125],norm_sample['norm_rightShoulder_y'][5:125])
        feature_one_sample['leftwrist_leftShoulder_distance'] = edistance(norm_sample['norm_leftWrist_x'][5:125], norm_sample['norm_leftShoulder_x'][5:125], norm_sample['norm_leftWrist_y'][5:125], norm_sample['norm_leftShoulder_y'][5:125])
    else:
        feature_one_sample['rightwrist_rightshoulder_distance'] = edistance(norm_sample['norm_rightWrist_x'],norm_sample['norm_rightShoulder_x'],norm_sample['norm_rightWrist_y'],norm_sample['norm_rightShoulder_y'] )
        feature_one_sample['leftwrist_leftShoulder_distance'] = edistance(norm_sample['norm_leftWrist_x'], norm_sample['norm_leftShoulder_x'], norm_sample['norm_leftWrist_y'], norm_sample['norm_leftShoulder_y'])
    feature_one_sample = np.round(feature_one_sample.mean(axis=0), 3)

    return feature_one_sample

def get_feature_allsamples(all_data):
    feature_label_all_sample = []
    mytemp = []
    for i, cat in enumerate(all_data.keys()):
        for sample in all_data[cat].keys():
            feature_tmp = get_feature_onesample(all_data[cat][sample])
            temp = [feature_tmp['leftwrist_leftShoulder_distance']]
            label_tmp = [i]
            if cat in ['mother', 'really','fun']:
                label2_temp = [0]
            elif cat in ['buy', 'hope','communicate']:
                label2_temp = [1]
            feature_label_tmp = feature_tmp.tolist() + label_tmp + label2_temp
            feature_label_all_sample.append(feature_label_tmp)
            mytemp.append((temp + label2_temp))

    feature_label_all_sample = np.asarray(feature_label_all_sample)
    return feature_label_all_sample, mytemp

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

    LDA = LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001)

    GradBoostC = GradientBoostingClassifier(loss='deviance', learning_rate=0.05, n_estimators=500, subsample=1.0, criterion='friedman_mse',
            min_samples_split=5, min_samples_leaf=3, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0,
            min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=True,
            presort='auto', validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)

    GaussianProcClass= GaussianProcessClassifier(kernel=5.0 * RBF(2.0), optimizer='fmin_l_bfgs_b', n_restarts_optimizer=4, max_iter_predict=400,
            warm_start=False, copy_X_train=True, random_state=None, multi_class='one_vs_rest', n_jobs=None)

    classifiers = [LogReg, KneighberC, svm, Nusvm, DTC, QDA, LDA, GradBoostC, GaussianProcClass, MLPC, RndForC, RadiusNeighClass]
    # classifiers = [LogReg, Nusvm, LDA, MLPC, DTC]
    return classifiers

#
#     # # ---------------------------- data plots for data visualization ---------------------------------------
#     # buy, data_buy = feature_foroneclass(all_data['buy'])
#     # communicate, data_communicate = feature_foroneclass(all_data['communicate'])
#     # fun, data_fun = feature_foroneclass(all_data['fun'])
#     # hope, data_hope = feature_foroneclass(all_data['hope'])
#     # mother, data_mother = feature_foroneclass(all_data['mother'])
#     # really, data_really = feature_foroneclass(all_data['really'])
#
#     # Make plots
#     # fig = plt.figure()
#     # ax1 = fig.add_subplot(211)
#     # plt.plot(communicate['leftWrist_hip_distance'], 'r-')
#     # j = 20
#     # print('data_buy',data_buy[j].shape)
#     # print('data_communicate',data_communicate[j].shape)
#     # print('data_fun',data_fun[j].shape)
#     # print('data_hope',data_hope[j].shape)
#     # print('data_mother',data_mother[j].shape)
#     # print('data_really',data_really[j].shape)
#     # plt.plot(data_buy[j]['leftWrist_hip_distance'], color='blue')
#     # plt.plot(data_communicate[j]['leftWrist_hip_distance'], color='pink')
#     # plt.plot(data_fun[j]['leftWrist_hip_distance'], color='green')
#     # plt.plot(data_hope[j]['leftWrist_hip_distance'], color='magenta')
#     # plt.plot(data_mother[j]['leftWrist_hip_distance'], color='cyan')
#     # plt.plot(data_really[j]['leftWrist_hip_distance'], color='purple')
#     # plt.title("leftWrist_hip_distance")
#     # ax2 = fig.add_subplot(212)
#     # plt.plot(data_buy[j]['leftWrist_nose_distance'], color='blue')
#     # plt.plot(data_communicate[j]['leftWrist_nose_distance'], color='pink')
#     # plt.plot(data_fun[j]['leftWrist_nose_distance'], color='green')
#     # plt.plot(data_hope[j]['leftWrist_nose_distance'], color='magenta')
#     # plt.plot(data_mother[j]['leftWrist_nose_distance'], color='cyan')
#     # plt.plot(data_really[j]['leftWrist_nose_distance'], color='purple')
#     # # plt.ylabel("Temperature (oC)")
#     # plt.legend(['buy', 'communicate', 'fun', 'hope', 'mother', 'really'])
#     # plt.title("leftWrist_nose_distance")
#     # ax1.axes.get_xaxis().set_visible(False)
#     # plt.show()
#
#     # # Make plots
#     # fig = plt.figure()
#     # ax1 = fig.add_subplot(211)
#     # # plt.plot(communicate['leftWrist_hip_distance'], 'r-')
#     # name = 'rightElbow_hip_distance'
#     # plt.plot(buy[name], color='blue')
#     # plt.plot(communicate[name], color='pink')
#     # plt.plot(fun[name], color='green')
#     # plt.plot(hope[name], color='magenta')
#     # plt.plot(mother[name], color='cyan')
#     # plt.plot(really[name], color='purple')
#     # plt.title(name)
#     # ax2 = fig.add_subplot(212)
#     # name2= 'rightWrist_nose_distance'
#     # plt.plot(buy[name2], color='blue')
#     # plt.plot(communicate[name2], color='pink')
#     # plt.plot(fun[name2], color='green')
#     # plt.plot(hope[name2], color='magenta')
#     # plt.plot(mother[name2], color='cyan')
#     # plt.plot(really[name2], color='purple')
#     # # plt.ylabel("Temperature (oC)")
#     # plt.legend(['buy', 'communicate', 'fun', 'hope', 'mother', 'really'])
#     # plt.title(name2)
#     # ax1.axes.get_xaxis().set_visible(False)
#     # plt.show()
#     # -----------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    print('a')
    all_data = read_data_all()
    print('b')
    feature_label, mytemp = get_feature_allsamples(all_data)
    print('c')
    print(mytemp[1])
    plt.plot(mytemp[mytemp[1,:]==0], color='green')
    plt.plot(mytemp[mytemp[1,:]==1], color='blue')
    plt.show()
    # X = feature_label[:, :-1]
    # y = feature_label[:, -1]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42, shuffle=True)
    # accuracy_scores_dict = {}
    # precision_scores_dict = {}
    # recall_scores_dict = {}
    # f1_scores_dict = {}
    # trained_classifiers = {}
    # classifiers = classifier_init()
    # for clf in tqdm(classifiers):
    #     classifier_name = clf.__class__.__name__
    #     print(classifier_name)
    #     clf.fit(X_train, y_train)
    #     trained_classifiers[classifier_name] = clf
    #     y_pred = clf.predict(X_test)
    #     accuracy = accuracy_score(y_test, y_pred)
    #     precision = precision_score(y_test, y_pred, average='micro')
    #     recall = recall_score(y_test, y_pred, average='micro')
    #     f1 = f1_score(y_test, y_pred, average='micro')
    #     accuracy_scores_dict[classifier_name] = accuracy
    #     precision_scores_dict[classifier_name] = precision
    #     recall_scores_dict[classifier_name] = recall
    #     f1_scores_dict[classifier_name] = f1
    #     print("="*30)
    #     print(classifier_name)
    #     print('****Results****')
    #     print("Accuracy: {:.4%}".format(accuracy))
    #     print("Precision: {:.4%}".format(precision))
    #     print("Recall: {:.4%}".format(recall))
    #     print("F1: {:.4%}".format(f1))
    #     print("="*30)
    #
    # classifiers_sorted_by_accuracy = sorted(accuracy_scores_dict.items(), key=itemgetter(1), reverse=True)
    # top_4 = []
    # for classifier in classifiers_sorted_by_accuracy[:4]:
    #     top_4.append(list(classifier)[0])
    #
    # top_4classifiers = {}
    # for classifier_name in top_4:
    #     top_4classifiers[classifier_name] = trained_classifiers[classifier_name]
    #     print(classifier_name)
    #
    # pkl.dump(top_4classifiers, open('trained_classifier.pkl', 'wb'))

