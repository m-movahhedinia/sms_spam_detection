# -*- coding: utf-8 -*-
"""
Created on 

@author: mansour
"""

from logging import getLogger

from torch.nn import Dropout, Linear, Module, ReLU
from transformers import DistilBertForSequenceClassification

log = getLogger()


class ModelMaker(Module):
    def __init__(self, number_of_classes: int, model_name_or_path: str, dropout_probability: float = 0.3):
        log.info("Building model architecture.")
        super(ModelMaker, self).__init__()
        self.base_model = DistilBertForSequenceClassification.from_pretrained(model_name_or_path)
        self.dropout = Dropout(p=dropout_probability)
        self.linear = Linear(self.base_model.config.hidden_size, number_of_classes)
        self.relu = ReLU()

    def forward(self, input_ids, attention_mask, labels):
        pooled_output = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = pooled_output.loss
        dropout_output = self.dropout(pooled_output.logits)
        linear_output = self.linear(dropout_output)
        relu_output = self.relu(linear_output)
        return loss, relu_output
