# cse535-2019F-A2
<h1>CSE 535 - Assignment 2 - Team 3</h1>
<b>Human Gesture Prediction using PoseNet data.</b>

This is a server application that serves with the name of a gesture when provided the json data generated by posenet using the videos of the gestures.

<h2>Details</h2>
The server mainly provides with api that receives 'application/json' data and responds with json data as well.
The input format is same as the json files generate by the 'scale_to_videos.js' file in this repository:
https://github.com/prashanthnetizen/posenet_nodejs_setup

<h3>How to start the server</h3>
<ol>
	<li>Install docker for your OS: <br/>
		https://www.docker.com/
	</li>
	<li>For Linux users, add your user to the docker group: <br/>
		https://docs.docker.com/install/linux/linux-postinstall/
	</li>
	<li>Clone this repository in your machine: <br/>
		<pre>git clone https://github.com/tnasim/cse535-2019F-A2.git</pre>
	</li>
	<li>Go to the downoad directory: <br/>
		<pre>cd cse535-2019F-A2</pre>
	</li>
	<li>Build the docker image (notice the dot, '.', at the end): <br/>
		<pre>docker build -t myserver .</pre>
	</li>
	<li>Run the docker image just created: <br/>
		<pre>docker run -ti --rm 5000:5000 myserver python3 app.py</pre>
	</li>
	<li>Check if the server is running on browser: <br/>
		http://{your-server-url}:5000/ <br/>
		example: http://127.0.0.1:5000/
	</li>
</ol>
