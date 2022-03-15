# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 19:19:53 2022

@author: mansour
"""

from json import dump
from logging import getLogger
from pathlib import Path
from typing import Dict, List, NoReturn, Union

log = getLogger()


def batch_maker(batch_size, data):
    log.info("Creating batches from data.")
    for beginning in range(0, len(data), batch_size):
        yield data[beginning: beginning + batch_size]


def save_json(file_location: str, data: Union[List, Dict, str]) -> NoReturn:
    log.info("Saving json.")
    location = Path(file_location)
    location.parent.mkdir(parents=True, exist_ok=True)
    with open(location, "w", encoding="utf-8") as file:
        dump(data, file, indent=4)


def subset_data(full_data: list, subset: List[int]):
    log.info("Creating subset from data.")
    return [full_data[chosen] for chosen in subset]
