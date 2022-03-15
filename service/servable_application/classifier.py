# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 19:19:53 2022

@author: mansour

Simple servable module to serve the model as a classifier with.
"""

from json import dumps, load
from logging import getLogger
from pathlib import Path
from typing import Dict, List, NoReturn, Union

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

log = getLogger()


class SpamClassifier:
    """
    Holds the methods to initiate and serve a classifier. Can be initiated only once. Will update the module state when
    initiated.
    """
    # Initiating class variables.
    environment_initiated = False
    classifier = None
    label_map = None

    @classmethod
    def _prepare_environment(cls, model_path: str) -> NoReturn:
        """
        Prepares the environment by loading the model, tokenizer, and label_map. Will load these elements only once.
        Will change the class definition state to avoid the need to reload them. Expects the path to the model files to
        be provided as a string object.

        Args:
            model_path (str): The path to the model files.
        """
        if cls.environment_initiated is False:
            log.info("Initializing model and tokenizer.")
            model_path = Path(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path.as_posix())
            tokenizer = AutoTokenizer.from_pretrained(model_path.as_posix())
            cls.classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
            cls.label_map = load(open(model_path.joinpath("labels_map.json")))

    @classmethod
    def infer(cls, texts: Union[str, List[str]], model_path: str) -> List[Dict[str, str]]:
        """
        Performs the classification and returns the label of the class, the confidence score, and the original
        utterance for each utterance in the form of a list of dictionaries. Will try to initialize the model, etc.

        Args:
            texts (Union[str, List[str]]): The list of the utterances to process or a single utterance.
            model_path (str): The path to the model.

        Returns:
            List[Dict[str, str]]: A list of the dictionary containing the inferred class' label, the confidence score,
            and the original utterance.

        """
        log.info(f"Running inference on:\n\t{texts}")
        cls._prepare_environment(model_path)
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(texts, list):
            results = cls.classifier(texts)
        else:
            raise BadPayloadException("Bad payload structure. Was expecting a list/array.")

        final_result = [{"label": cls.label_map[result["label"].lower().strip("label_")],
                         "confidence": result["score"],
                         "utterances": texts[index]} for index, result in enumerate(results)]
        log.info(f"Results:\n{dumps(final_result, indent=4)}")
        return final_result


class BadPayloadException(BaseException):
    """
    An exception to call when the payload is not of an expected data type.
    """

    def __init__(self, message: str):
        """
        Raises a custom :class:`BaseException` to indicate a problematic payload.

        Args:
            message (str): The message of the exception
        """
        super().__init__(message)
