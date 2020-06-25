FROM python:3.6

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y \
	emacs \
	git \
	wget
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash -
RUN apt-get install -y nodejs
