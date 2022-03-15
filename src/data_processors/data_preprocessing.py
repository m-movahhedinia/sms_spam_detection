# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 19:19:53 2022

@author: mansour
"""

from json import dump
from logging import getLogger
from operator import methodcaller
from pathlib import Path
from time import time
from typing import Union

from sklearn.model_selection import StratifiedKFold

log = getLogger()


class DataPreprocessor:
    def __init__(self, input_file: Union[str, Path]):
        log.info("Preprocessing the data.")
        self.input_file = Path(input_file)
        self.texts = []
        self.labels = []
        self.label_map = {}
        self.max_token_is_valid = None
        self.folds = {}

    def load_data(self) -> "DataPreprocessor":
        for record in open(self.input_file, encoding="utf-8").readlines():
            label, text = record.strip().split(maxsplit=1)
            self.labels.append(label.lower())
            self.texts.append(text.strip().lower())
        log.info("Finished loading the data.")
        return self

    def check_max_token_size(self, max_threshold: int = 512) -> "DataPreprocessor":
        sizes = list(map(len, map(methodcaller("split"), self.texts)))
        max_token = max(sizes)
        if max_token <= max_threshold:
            log.info(f"The maximum number of tokens in the dataset is within the threshold. {max_token} <= "
                     f"{max_threshold}")
            self.max_token_is_valid = True
        else:
            out_of_bounds = [i > 512 for i in sizes]
            log.warning(f"The maximum number of tokens in the dataset is more than the threshold. {sum(out_of_bounds)} "
                        f"records will be truncated if used for training. {max_token} > {max_threshold}")
            self.max_token_is_valid = False
        return self

    def map_labels(self):
        self.label_map = dict(enumerate(set(self.labels)))
        label_map_reverse = dict(map(reversed, self.label_map.items()))
        self.labels = [label_map_reverse[label] for label in self.labels]
        log.info("Finished processing the labels.")
        return self

    def save_label_map(self, output_location: Union[Path, str]) -> "DataPreprocessor":
        location = Path(output_location)
        location.parent.mkdir(parents=True, exist_ok=True)
        with open(location, "w", encoding="utf-8") as file:
            dump(self.label_map, file, indent=4)
        log.info(f"Saved label map to {location.as_posix()}.")
        return self

    def prep_k_fold(self, k: int = 5, random_state: int = int(time())) -> "DataPreprocessor":
        k_folder = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        for fold, (train, test) in enumerate(k_folder.split(self.texts, self.labels)):
            self.folds[fold] = {"train": train, "test": test}
        log.info(f"Prepared {k} folds data.")
        return self
