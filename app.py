from flask import Flask
from flask import Response
from flask import request
from flask import jsonify
import json

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
	return Response(json.dumps(pose_frames), mimetype='application/json')


@app.route('/api/test_json', methods=['POST'])
def test_json():
	pose_frames = request.get_json()
	if(pose_frames == None):
		return "ERROR reading json data from request"
	return Response(json.dumps(pose_frames), mimetype='application/json')


if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0')
