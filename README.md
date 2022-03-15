# SMS Spam Detection
A simple distilbert based approach to the famous sms spam detection problem which employs both k-fold validation and 
transfer learning (to some extent). To run the project, you follow the steps bellow.
All commands in this readme have to run from the project root.

1. Create a virtual environment and install the requirements files.
2. Download the data from the link below and place it in a directory called data in the root of the project.
https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
3. Run the command bellow to.
```commandline
python run_training.py --model_location distilbert-base-uncased
```
This will launch the training phase of the project in its simplest form.

The project also contains a server and an API. However, you need a trained model to start it. If you have a trained
model you can start the server using either of the commands bellow.
Flask:
```commandline
python run_server.py
```

Gunicorn + Flask:
```commandline
gunicorn --chdir service app:app -w 2 --threads 2 -b 0.0.0.0:8000
```

However, you will need to define an environmental variable called `MODEL_LOCATION` to store the model location first.
The server is actually meant to be launched through a docker container. Accordingly, that would be the best option. You
can use the command bellow to create a docker container.
```commandline
sudo docker build -t sms_spam_classifier .
```
And then.
```commandline
sudo docker run -p 8000:8000 test
```

After the container is created, you can send post requests to the server. Use the curl bellow as a sample. 
```commandline
curl --location --request POST 'http://127.0.0.1:8000/detect_spam' --header 'Content-Type: application/json' --data-raw '{"utterances": ["try this", "and now this"]}'
```

# Current Shortcomings
- The data is not normalized and there are lots of sentences in it which have more than 512 tokens. These messages will be
truncated and may cause loss of decisive data. A normalization process can be beneficial to remove the expendable
tokens.
- The data contains a considerable amount of informal expressions. Some of these can be normalized, like contractions, 
to provide better understanding for the model.
- Using k-fold validation with a pretrained model increases the temporal complexity of the training. It is possible to 
achieve comparable state-of-the-art results with normal training. However, if the data was grouped, using k-fold would
have been much more effective.
- The implemented transfer learning is very rudimentary for now. Improvements are in order.
