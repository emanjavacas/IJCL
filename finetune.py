
"""
finetune-cv.py: This script performs training on the provided train file, and
inference on the provided test file.
"""

__author__    = "Enrique Manjavacas"
__copyright__ = "Copyright 2022, Enrique Manjavacas"
__license__   = "GPL"
__version__   = "1.0.1"

import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding, EarlyStoppingCallback
from transformers import Trainer, TrainingArguments

from preprocess import encode_data, read_data


def get_dataset(tokenizer, sents, spans, labels=None):
    dataset = {'text': sents, 'spans': spans}
    if labels is not None:
        dataset['label'] = labels
    dataset = Dataset.from_dict(dataset)

    return dataset.map(
        lambda examples: tokenizer(examples['text'], truncation=True, max_length=512),
        batched=True
    ).remove_columns('text')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        "This file performs finetuning on a given train file and a given pre-trained "
        "transformer model. After training the model is used to perform inference on "
        "the provided test file. The results are stored on a csv file with the same "
        "name as the input file but different extension.")
    parser.add_argument('--modelname', required=True,
                        help="The name of a transformer model (huggingface).")
    parser.add_argument('--train-file', required=True, help="File to do training on.")
    parser.add_argument('--test-file', required=True, help="File to do inference on.")
    parser.add_argument('--label', required=True, help="Name of the label column.")
    parser.add_argument('--lhs', default='left', help='Name of the left context column.')
    parser.add_argument('--target', default='hit', help='Name of the left context column.')
    parser.add_argument('--rhs', default='right', help='Name of the left context column.')
    parser.add_argument('--epochs', type=int, default=6, help="Number of epochs to train.")
    parser.add_argument('--output-dir', required=True, help="Directory to store the finetuned model.")
    parser.add_argument('--dev-split', type=float, default=0.1, 
                        help="Size for the dev split in the 0-1 range.")
    args = parser.parse_args()

    # Normalise whitespaces
    def normalise(example):
        return ' '.join(example.split())

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('PyTorch is using CUDA enabled GPU')
    else:
        device = torch.device('cpu')
        print('PyTorch is using CPU')

    tokenizer = AutoTokenizer.from_pretrained(args.modelname)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[TGT]']})

    data = pd.read_csv(args.train_file)
    for heading in [args.lhs, args.target, args.rhs]:
        data[heading] = data[heading].transform(normalise)
    sents, starts, ends = read_data(data[args.lhs], data[args.target], data[args.rhs])
    sents, spans = encode_data(tokenizer, sents, starts, ends)
    sents, spans = np.array(sents), np.array(spans)
    # prepare labels
    label2id = {label: id for id, label in enumerate(sorted(data[args.label].unique()))}
    id2label = {id: label for label, id in label2id.items()}
    y = np.array([label2id[label] for label in data[args.label]])

    train, dev = train_test_split(
        np.arange(len(sents)), 
        stratify=y, 
        test_size=args.dev_split)
    train_dataset = get_dataset(tokenizer, sents[train], spans[train], y[train])
    dev_dataset = get_dataset(tokenizer, sents[dev], spans[dev], y[dev])

    model = AutoModelForSequenceClassification.from_pretrained(
            args.modelname, num_labels=len(set(y))
    ).to(device)
    # this is needed, since we have expanded the tokenizer to incorporate
    # the target special token [TGT]
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=4.5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=args.epochs,
        weight_decay=0.1,
        do_eval=True,
        save_strategy='epoch',
        evaluation_strategy="epoch",
        load_best_model_at_end=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        # early stopping
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])

    trainer.train()

    # inference on test data
    test_data = pd.read_csv(args.train_file)
    for heading in [args.lhs, args.target, args.rhs]:
        test_data[heading] = test_data[heading].transform(normalise)
    test_sents, test_starts, test_ends = read_data(
        test_data[args.lhs], test_data[args.target], test_data[args.rhs])
    test_sents, test_spans = encode_data(tokenizer, test_sents, test_starts, test_ends)
    test_sents, test_spans = np.array(test_sents), np.array(test_spans)

    test_dataset = get_dataset(tokenizer, test_sents, test_spans)

    # these are actually the logits
    preds, _, _ = trainer.predict(test_dataset)
    scores = np.max(softmax(preds, axis=1), axis=1)
    preds = np.argmax(preds, axis=1)
    preds = [id2label[i] for i in preds]

    prefix, suffix = os.path.splitext(args.test_file)
    pd.DataFrame.from_dict(
        {'score': scores, 'pred': preds}
    ).to_csv(''.join([prefix, '.finetune.results.csv']))
