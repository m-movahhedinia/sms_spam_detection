FROM python:3.8

RUN mkdir /conversationHealth

ADD requirements.txt /conversationHealth/
ADD launch_server.sh /conversationHealth/
ADD servable_model /conversationHealth/servable_model/
ADD service /conversationHealth/service/

RUN pip install --upgrade pip

RUN pip install -r /conversationHealth/requirements.txt

RUN chmod +x /conversationHealth/launch_server.sh

WORKDIR /conversationHealth

ENV MODEL_LOCATION=/conversationHealth/servable_model

ENTRYPOINT /conversationHealth/launch_server.sh