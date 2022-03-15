# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 19:19:53 2022

@author: mansour

Holds a simple app server and its endpoints.
"""

from logging import INFO, basicConfig, getLogger
from os import environ

from flask import Flask, jsonify, request

from service.servable_application.classifier import BadPayloadException, SpamClassifier

basicConfig(level=INFO)
log = getLogger()

app = Flask(__name__)


@app.route("/")
def check():
    """
    A simple endpoint to be used to see if the server is up or not.
    Returns:
        (str): A simple message.
    """
    return "And thus it shall live!"


@app.route("/detect_spam", methods=["POST"])
def detect_spam():
    """
    Provides an endpoint to the classifier. It accepts a post request and expects it to contain either of the four
    utterances, queries, sentences, or texts keys. Will return the error if the value of the key is not recognized and
    a 500 state if it faces an undefined error.

    Returns:
        (str): The json representation of a dict.
    """
    payload = request.get_json(force=True)
    utterances = payload.get("utterances") or payload.get("queries") or payload.get("sentences") or payload.get("texts")
    if utterances:
        try:
            result = SpamClassifier.infer(utterances, environ.get("MODEL_LOCATION"))
        except BadPayloadException as e:
            log.error(e)
            result = {"error": f"{e}"}
        except Exception as e:
            log.error(e)
            return "Either you killed us or we goofed up. Either way, internal server error (500)!", 500
    else:
        log.error("Payload missing.")
        result = {"error": "Payload missing."}
    return jsonify(result)
