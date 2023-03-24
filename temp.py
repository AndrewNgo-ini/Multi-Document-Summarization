from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from torch.utils.data import DataLoader, Dataset
import torch
import os 
import pandas as pd 
import pickle
from utils import *
os.environ["WANDB_DISABLED"] = "true"


def prepare_input(text, tokenizer):
    inputs = tokenizer.encode_plus(
        text, 
        return_tensors = None, 
        add_special_tokens = True, 
        truncation = True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype = torch.long)
    return inputs


class custom_dataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df['text'].values
        self.labels = df['label'].values
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        inputs = prepare_input(self.texts[item], tokenizer)
        label = torch.tensor(self.labels[item])
        inputs["labels"] = label
        return inputs
        

import random

def pick_subset_by_percentage(items, percentage):
    """
    Picks a random subset of items from a list based on a percentage.
    :param items: A list of items to choose from.
    :param percentage: The percentage of items to select.
    :return: A list of randomly selected items from items.
    """
    if percentage < 0 or percentage > 100:
        raise ValueError("Percentage should be between 0 and 100.")

    num_items = len(items)
    num_select = round(num_items * percentage / 100)

    if num_select == 0:
        return []

    return random.sample(items, num_select)

import numpy as np

import evaluate

accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

if __name__ == "__main__":
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    tokenizer= AutoTokenizer.from_pretrained("vinai/phobert-base")
    model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base",
                                                               num_labels=2, id2label=id2label, label2id=label2id)
    # Freeze the pretrained BERT layers
    for param in model.base_model.parameters():
        param.requires_grad = False

    training_args = TrainingArguments(
        output_dir="my_awesome_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    ds = read_data_tok()
    lb = read_label_tok()

    random.seed(42)
    val_cluster = pick_subset_by_percentage(ds.keys(), 20)
    print(val_cluster)
    with open('val_cluster.pkl', 'wb') as file:
        pickle.dump(val_cluster, file)
    texts = []
    labels = []
    clusters = []
    for k,v in ds.items():
        for sentence in v:
            if sentence in lb[k]:
                labels.append(1)
            else:
                labels.append(0)
            texts.append(sentence)
            clusters.append(k)
    df = pd.DataFrame({"text": texts, "label": labels, "cluster": clusters})
    print(df.shape)
    print(df["label"].value_counts())

    ds_train = custom_dataset(df[~df["cluster"].isin(val_cluster)], tokenizer)
    ds_val = custom_dataset(df[df["cluster"].isin(val_cluster)], tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
