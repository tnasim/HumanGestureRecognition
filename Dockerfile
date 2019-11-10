FROM python:3
#FROM ubuntu:18.04

MAINTAINER Tariq M Nasim "tnasim@asu.edu"

USER root

RUN apt-get update -y \
	&& apt-get install -y python3-pip python3-dev nano curl \
	&& cd /usr/local/bin \
	&& pip3 install --upgrade pip
	# && ln -s /usr/bin/python3 python \

# Copy the requirements.txt
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip3 install -r requirements.txt

# Copy all other files from the current directory into the working directory.
COPY . /app

#ENTRYPOINT [ "/bin/bash" ]

#ENTRYPOINT [ "python3" ]

#CMD [ "app.py" ]
