# cse535-2019F-A2
<h1>CSE 535 - Assignment 2 - Team 3</h1>
<b>Human Gesture Prediction using PoseNet data.</b>

This is a server application that serves with the name of a gesture when provided the json data generated by posenet using the videos of the gestures.

<h2>Details<h2>:
The server mainly provides with api that receives 'application/json' data and responds with json data as well.
The input format is same as the json files generate by the 'scale_to_videos.js' file in this repository:
https://github.com/prashanthnetizen/posenet_nodejs_setup

How to start the server:
1. Install docker for your OS:
	https://www.docker.com/
2. For Linux users, add your user to the docker group:
	https://docs.docker.com/install/linux/linux-postinstall/
3. Clone this repository in your machine:
	git clone https://github.com/tnasim/cse535-2019F-A2.git
4. Go to the downoad directory:
	cd cse535-2019F-A2
5. Build the docker image (notice the dot, '.', at the end):
	docker build -t myserver .
6. Run the docker image just created:
	docker run -ti --rm 5000:5000 myserver python3 app.py
7. Check if the server is running on browser:'
	http://<your-server-url>:5000/
	example: http://127.0.0.1:5000/
