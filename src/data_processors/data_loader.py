# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 19:19:53 2022

@author: mansour
"""

from torch import tensor
from torch.utils.data import Dataset


class DataProvider(Dataset):
    def __init__(self, texts: list, ids: list, labels: list, attention_masks: list):
        self.attention_masks = attention_masks
        self.ids = ids
        self.texts = texts
        self.labels = labels  # list of 0-1 ints

    def __getitem__(self, item):
        return {"ids": self.ids[item], "texts": self.texts[item], "labels": tensor(self.labels[item]),
                "attention_masks": self.attention_masks[item]}

    def __len__(self):
        return len(self.labels)
