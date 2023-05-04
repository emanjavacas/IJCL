
"""
finetune-cv.py: This script runs 5-fold cross-validation on your input file,
finetuning the model to classify the labels in your dataset.
It outputs a parquet file with the results of all of the folds. 
If, beside a training file, you already have a specific test file where you, 
would like to test the resulting model, you should use the finetune.py script.
For more information on how to run this script, run the command 
‘python finetune-cv.py --help’ in the command line.
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding, EarlyStoppingCallback
from transformers import Trainer, TrainingArguments

from preprocess import encode_data, read_data, sample_up_to_n


def get_dataset(tokenizer, sents, spans, labels):
    dataset = {'text': sents, 'spans': spans}
    dataset['label'] = labels
    dataset = Dataset.from_dict(dataset)

    return dataset.map(
        lambda examples: tokenizer(examples['text'], truncation=True, max_length=512),
        batched=True
    ).remove_columns('text')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        "This file performs cross-validation on a given input file and a given pre-trained "
        "transformer model. The results are stored on a parquet file with the same name as "
        "the input file but different extension.")
    parser.add_argument('--modelname', required=True,   
                        help="The name of a transformer model (huggingface).")
    parser.add_argument('--input-file', required=True,
                        help="CSV file with at least three field specifying the left "
                        "and right context as well as the target word and the label.")
    parser.add_argument('--label', required=True, help="Name of the label column.")
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--lhs', default='left', help='Name of the left context column.')
    parser.add_argument('--target', default='hit', help='Name of the left context column.')
    parser.add_argument('--rhs', default='right', help='Name of the left context column.')
    parser.add_argument('--epochs', type=int, default=6, help="Number of epochs to train.")
    parser.add_argument('--output-dir', required=True, help="Directory to store the finetuned model.")
    parser.add_argument('--results-path', help='Custom dir path for the results.')
    parser.add_argument('--max-per-class', default=np.inf, type=float, 
                        help="Max items per class to train on.")
    parser.add_argument('--n-folds', type=int, default=10, help="Number of folds over the data.")
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

    data = pd.read_csv(args.input_file)
    for heading in [args.lhs, args.target, args.rhs]:
        data[heading] = data[heading].fillna('').transform(normalise)
    sents, starts, ends = read_data(data[args.lhs], data[args.target], data[args.rhs])
    sents, spans = encode_data(tokenizer, sents, starts, ends)
    sents, spans = np.array(sents), np.array(spans)
    # prepare labels
    label2id = {label: id for id, label in enumerate(sorted(data[args.label].unique()))}
    id2label = {id: label for label, id in label2id.items()}
    y = np.array([label2id[label] for label in data[args.label]])

    cv = StratifiedKFold(args.n_folds, shuffle=True, random_state=135)
    folds = []

    for fold, (train, test) in enumerate(cv.split(np.zeros(len(y)), y)):
        train, dev = train_test_split(train, shuffle=True, stratify=y[train], test_size=0.1)

        if args.max_per_class < np.inf:
            train = pd.DataFrame(
                {'labels': y[train], 'index': train}
            ).groupby('labels').apply(
                lambda g: sample_up_to_n(g, int(args.max_per_class))
            ).reset_index(drop=True)['index'].values
            print("Training on", len(train), "instances")

        train_dataset = get_dataset(tokenizer, sents[train], spans[train], y[train])
        dev_dataset = get_dataset(tokenizer, sents[dev], spans[dev], y[dev])
        test_dataset = get_dataset(tokenizer, sents[test], spans[test], y[test])

        model = AutoModelForSequenceClassification.from_pretrained(
                args.modelname, num_labels=len(set(y))
            ).to(device)
        # this is needed, since we have expanded the tokenizer to incorporate
        # the target special token [TGT]
        model.resize_token_embeddings(len(tokenizer))

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            learning_rate=4.5e-5,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
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
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])

        trainer.train()
        # these are actually the logits
        preds, _, _ = trainer.predict(test_dataset)
        scores = np.max(softmax(preds, axis=1), axis=1)
        preds = np.argmax(preds, axis=1)
        preds = [id2label[i] for i in preds]

        folds.append(pd.DataFrame({
            'fold': fold, 
            'test': test,
            'trues': [id2label[y[i]] for i in test],
            'scores': scores, 
            'preds': preds}))
        
    prefix, _ = os.path.splitext(args.input_file)
    if args.results_path:
        if not os.path.isdir(args.results_path):
            os.makedirs(args.results_path)
        prefix = os.path.basename(prefix)
        prefix = os.path.join(args.results_path, prefix)
    pd.concat(folds).to_parquet(''.join([prefix, '.finetune-cv.results.parquet']))