from flask import Flask
from flask import Response
from flask import request
from flask import jsonify
import json
from Classifier import Classifier
import pandas as pd
import numpy as np
import glob
import sys
import pickle
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

models = {}

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


def predict_gesture(json_object):
    test_case = read_test_data(json_object)
    klass = Classifier()
    klass.__init__
    result = klass.classify(test_case)
    print(result)
    return result
    # label_dict = {0: 'buy', 1: 'communicate', 2: 'fun', 3: 'hope', 4: 'mother', 5: 'really'}
	# predictions = {}
	# for i, name in enumerate(trained_models_loaded.keys()):
	# 	y_pred = trained_models_loaded[name].predict(x_test)
	# 	predictions[str(i+1)] = label_dict[y_pred[0]]
	# 	print "y_pred[0] = ", y_pred[0], " ",  name, " ", predictions[str(i+1)]
	# return predictions


app = Flask(__name__)

@app.route('/')
def root():
	html =		"<html>" \
			+ "<head><title>CSE535-Gesture Predictor</title></head>" \
			+ 	"<body>" \
			+ 		"<h1>Gesture Predictor - CSE 535 Assignment 2 - Group 3</h1>" \
			+ 		"<h2>API Endpoints:</h2>" \
			+ 		"<ul>"\
			+ 			"<li>" + "<b>Gesture Prediction:</b> " + request.host_url + "api/predict_gesture" + " [POST] 'application/json'" + "</li>" \
			+ 		"</ul>" \
			+ 	"</body>" \
			+ "</html>"
	return html

@app.route('/api/predict_gesture', methods=[ 'POST' ])
def api_predict_gesture():
	pose_frames = request.get_json()
	if(pose_frames == None):
		return "ERROR reading json data from request"
	result = predict_gesture(pose_frames)
	print("result: ", result)
	return Response(json.dumps(result), mimetype='application/json')


@app.route('/api/test_json', methods=['POST'])
def test_json():
	pose_frames = request.get_json()
	if(pose_frames == None):
		return "ERROR reading json data from request"
	return Response(json.dumps(pose_frames), mimetype='application/json')


if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0')
