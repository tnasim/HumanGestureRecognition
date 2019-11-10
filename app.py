from flask import Flask
from flask import Response
from flask import request
from flask import jsonify
import json

app = Flask(__name__)

@app.route('/')
def root():
	return 'Gesture Server - CSE 535 Assignment 2 - Group 3'


@app.route('/api/predict_gesture')
def predict_gesture():
	return 'yet to be developed'


@app.route('/api/test_json', methods=['POST'])
def test_json():
	pose_frames = request.get_json()
	if(pose_frames == None):
		return "ERROR reading json data from request"
	return Response(json.dumps(pose_frames), mimetype='application/json')


if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0')
