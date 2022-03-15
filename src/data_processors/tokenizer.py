# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 19:19:53 2022

@author: mansour
"""

from concurrent.futures import ProcessPoolExecutor
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Tuple, Union

from tqdm import tqdm
from transformers import DistilBertTokenizerFast

log = getLogger()


class Tokenizer:
    def __init__(self, tokenizer_name_or_path: Union[str, Path]):
        log.info("Loading tokenizer.")
        tokenizer_name_or_path = tokenizer_name_or_path.as_posix() if isinstance(
                tokenizer_name_or_path, Path) else tokenizer_name_or_path
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_name_or_path)

    def _run_tokenizer(self, data: List[str]) -> Dict[str, List]:
        return self.tokenizer(data, padding="max_length", truncation=True, add_special_tokens=True,
                              return_attention_mask=True, return_token_type_ids=False, return_tensors='pt')

    def concurrent_tokenize(self, data: List[List[str]]) -> Tuple[List, List]:
        log.warning("Performing concurrent tokenization. Use this method only if your data is massive.")
        with ProcessPoolExecutor() as pool:
            results = list(tqdm(pool.map(self._run_tokenizer, data), total=len(data), desc="Batches"))

        input_ids = []
        attention_masks = []
        for result in results:
            input_ids.extend(result["input_ids"])
            attention_masks.extend(result["attention_mask"])

        return input_ids, attention_masks

    def tokenize(self, data: List[List[str]]) -> Tuple[List, List]:
        log.info("Performing tokenization.")
        input_ids = []
        attention_masks = []
        for batch in data:
            result = self._run_tokenizer(batch)
            input_ids.extend(result["input_ids"])
            attention_masks.extend(result["attention_mask"])

        return input_ids, attention_masks
