import os
import torch
import random
import pandas as pd
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split

class NERModelTrainer:
    def __init__(self, model_name="bert-base-cased", num_examples_per_entity=50):
        self.model_name = model_name
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.num_examples_per_entity = num_examples_per_entity
        self.label_to_id = {}
        self.id_to_label = {}
        self.model = None
        self.training_args = None
        self.trainer = None

    def generate_dataset(self, entities, sentence_templates):
        dataset = []
        for entity in entities:
            for _ in range(self.num_examples_per_entity):
                template = random.choice(sentence_templates)
                sentence = template.format(entity=entity)
                dataset.append({"text": sentence, "entity": entity})
        return dataset

    def text_to_bio(self, text, entity):
        tokens = text.split()
        labels = ["O"] * len(tokens)
        entity_tokens = entity.split()
        entity_length = len(entity_tokens)
        for i in range(len(tokens) - entity_length + 1):
            if tokens[i:i + entity_length] == entity_tokens:
                labels[i] = "ANIMAL"
        return tokens, labels

    def prepare_dataset(self, entities, sentence_templates):
        dataset = self.generate_dataset(entities, sentence_templates)
        bio_dataset = []
        for example in dataset:
            tokens, labels = self.text_to_bio(example["text"], example["entity"])
            bio_dataset.append({"tokens": tokens, "labels": labels})
        return bio_dataset

    def tokenize_and_align_labels(self, example):
        tokenized = self.tokenizer(example['tokens'], is_split_into_words=True, truncation=True, padding='max_length')
        word_ids = tokenized.word_ids()
        labels = []
        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            elif word_idx != prev_word_idx:
                labels.append(self.label_to_id[example['labels'][word_idx]])
            else:
                labels.append(self.label_to_id[example['labels'][word_idx]])
            prev_word_idx = word_idx
        tokenized['labels'] = labels
        return tokenized

    def train(self, bio_dataset):
        label_list = sorted(list(set(label for example in bio_dataset for label in example['labels'])))
        self.label_to_id = {label: i for i, label in enumerate(label_list)}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}

        train_data, eval_data = train_test_split(bio_dataset, test_size=0.2, random_state=42)
        train_dataset = Dataset.from_list(train_data).map(self.tokenize_and_align_labels, batched=False)
        eval_dataset = Dataset.from_list(eval_data).map(self.tokenize_and_align_labels, batched=False)

        self.model = BertForTokenClassification.from_pretrained(self.model_name, num_labels=len(label_list))
        self.training_args = TrainingArguments(
            output_dir="./ner_model",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            save_strategy="epoch",
            logging_dir="./logs",
        )
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        self.trainer.train()
        self.model.save_pretrained("./ner_model")
        self.tokenizer.save_pretrained("./ner_model")


if __name__ == "__main__":
    entities = ['turtle', 'jelly fish', 'dolphin', 'sharks', 'sea urchins',
                 'whale', 'octopus', 'puffers', 'sea rays', 'nudibranchs']
    sentence_templates = [
        "The {entity} was swimming in the ocean.",
        "{entity} is a fascinating marine creature.",
        "I saw a {entity} during my last snorkeling trip.",
        "Did you know that {entity} can grow up to 10 feet long?",
        "Many people are afraid of {entity}, but they are actually harmless.",
        "{entity} plays an important role in the marine ecosystem.",
        "If you visit the aquarium, don't miss the {entity} exhibit.",
        "Scientists are studying the behavior of {entity} in the wild.",
        "The population of {entity} has been declining due to climate change.",
        "{entity} is one of the most beautiful creatures in the sea."
    ]

    trainer = NERModelTrainer()
    dataset = trainer.prepare_dataset(entities, sentence_templates)
    trainer.train(dataset)
