from logging import getLogger
from pathlib import Path

import torch
from numpy import mean
from torch import cuda, device
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import AdamW, DistilBertForSequenceClassification, get_linear_schedule_with_warmup

from src.commons.utils import batch_maker, subset_data
from src.data_processors.data_loader import DataProvider
from src.data_processors.data_preprocessing import DataPreprocessor
from src.data_processors.tokenizer import Tokenizer

log = getLogger()


class ModelManager:
    def __init__(self, model_name_or_path: str, data_location: str, output_location: str, warmup_steps: int = 0,
                 learning_rate: float = 5.0e-5, epochs: int = 10, batch_size: int = 16):
        log.info("Initializing model training setup.")
        self.output_location = Path(output_location)
        self.data = DataPreprocessor(Path(data_location)).load_data().map_labels().save_label_map(
                self.output_location.joinpath("labels_map.json")).prep_k_fold()
        self.device = device("cuda" if cuda.is_available() else "cpu")
        log.info(f"Setting device to {self.device.type}.")

        self.model = DistilBertForSequenceClassification.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.tokenizer = Tokenizer(model_name_or_path)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.schedular = None
        self.batch_size = batch_size

    def run_training(self):
        log.info("Starting training process.")
        self.tokenizer.tokenizer.save_pretrained(self.output_location)

        best_fold = {}
        with logging_redirect_tqdm():
            for fold, data_sub_set in self.data.folds.items():
                training_data_raw_texts = subset_data(self.data.texts, data_sub_set["train"])
                training_data_input_ids, training_data_attention_masks = self.tokenizer.tokenize(
                        batch_maker(512, training_data_raw_texts))
                training_data_labels = subset_data(self.data.labels, data_sub_set["train"])
                test_data_raw_texts = subset_data(self.data.texts, data_sub_set["test"])
                test_data_input_ids, test_data_attention_masks = self.tokenizer.tokenize(batch_maker(
                        512, test_data_raw_texts))
                test_data_labels = subset_data(self.data.labels, data_sub_set["test"])

                train_data_loader = DataLoader(
                        DataProvider(training_data_raw_texts, training_data_input_ids, training_data_labels,
                                     training_data_attention_masks), self.batch_size, shuffle=True)
                test_data_loader = DataLoader(DataProvider(test_data_raw_texts, test_data_input_ids, test_data_labels,
                                                           test_data_attention_masks), self.batch_size, shuffle=True)
                total_steps = len(train_data_loader) * self.epochs
                self.schedular = get_linear_schedule_with_warmup(
                        optimizer=self.optimizer, num_training_steps=total_steps, num_warmup_steps=self.warmup_steps)

                best_accuracy = 0
                for epoch in range(self.epochs):
                    progressbar_description = f"Fold {fold}, epoch {epoch}"
                    train_accuracy, train_loss = self.train(train_data_loader, len(training_data_raw_texts),
                                                            progressbar_description)
                    evaluation_accuracy, evaluation_loss = self.evaluate(test_data_loader, len(training_data_raw_texts))
                    log.info(f"Fold {fold}, epoch {epoch} result:\n"
                             f"\tTraining:\tLoss: {train_loss}\t Accuracy: {train_accuracy.item()}\n"
                             f"\tEvaluation:\tLoss: {evaluation_loss}\t Accuracy: {evaluation_accuracy.item()}")

                    if evaluation_accuracy >= best_accuracy:
                        log.info(f"New best model accuracy achieved for fold {fold} in epoch {epoch}.")
                        best_accuracy = evaluation_accuracy
                    else:
                        log.info(f"Training resulted in an accuracy lower than the previous at epoch {epoch} in fold "
                                 f"{fold}. Model stability is now ambiguous. The model may go through a rebound with "
                                 f"more training. However, further training of this fold is being stopped for now.")
                        break

                if best_fold:
                    if best_fold["accuracy"] < best_accuracy:
                        log.info("A new best fold was found. Committing regicide.")
                        best_fold = {"fold": fold, "accuracy": best_accuracy}
                        self.model.save_pretrained(self.output_location, save_config=True)
                else:
                    log.info("Crowning this fold as the current best fold.")
                    best_fold = {"fold": fold, "accuracy": best_accuracy}
                    self.model.save_pretrained(self.output_location, save_config=True)

    def train(self, data_loader, sample_size: int, progressbar_description: str):
        self.model.train()
        losses = []
        true_positives_count = 0
        for train_data in tqdm(data_loader, desc=progressbar_description):
            self.optimizer.zero_grad()
            input_ids = train_data["ids"].to(self.device)
            attention_mask = train_data["attention_masks"].to(self.device)
            labels = train_data["labels"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            _, predictions = torch.max(outputs.logits, dim=1)
            loss = outputs.loss

            true_positives_count += torch.sum(predictions == labels)
            losses.append(loss.item())

            loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.schedular.step()

        return true_positives_count.double() / sample_size, mean(losses)

    def evaluate(self, data_loader, sample_size):
        self.model = self.model.eval()

        losses = []
        true_positives_count = 0
        with torch.no_grad():
            for test_data in tqdm(data_loader, desc="Evaluating"):
                input_ids = test_data["ids"].to(self.device)
                attention_mask = test_data["attention_masks"].to(self.device)
                labels = test_data["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                _, predictions = torch.max(outputs.logits, dim=1)

                loss = outputs.loss

                true_positives_count += torch.sum(predictions == labels)
                losses.append(loss.item())

            return true_positives_count.double() / sample_size, mean(losses)
