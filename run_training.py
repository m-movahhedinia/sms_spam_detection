# -*- coding: utf-8 -*-
"""
Created on 

@author: mansour
"""

from argparse import ArgumentParser
from logging import INFO, basicConfig, getLogger

from src.modeling.trainer import ModelManager

basicConfig(level=INFO)
log = getLogger()


class RunTraining:
    def __init__(self):
        parser = ArgumentParser()
        parser.add_argument("--model_location", required=False, type=str, default="models/distilbert-base-uncased",
                            help="The location of the model.")
        parser.add_argument("--data_location", required=False, type=str, default="data/SMSSpamCollection",
                            help="The location of the data.")
        parser.add_argument("--output_location", required=False, type=str, default="servable_model",
                            help="The location to save the trained model.")

        self.args = parser.parse_args()

    def run(self):
        modeler = ModelManager(self.args.model_location, self.args.data_location, self.args.output_location)
        modeler.run_training()


if __name__ == "__main__":
    operation = RunTraining()
    operation.run()
