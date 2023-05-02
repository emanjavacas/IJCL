MacBERTh fine-tuned "ing"-forms
---

This repository accompanies the paper ... 
It provides the necessary code to replicate the experiments presented in the paper, and also a guide to replicate the same type of semi-automated analysis of corpus data, leveraging so-called "found"-data.

The repository relies on the `transformers` library.

# Overview of the provided files

## Python scripts (.py)
`finetune.py` can be used to perform fine-tuning on a given training file, and utilize the model to tag given test files. 

`finetune-cv.py` performs cross-validation over the input training file.

`preprocess.py` provides functions for the other scripts and also allows to extract embeddings from an input file.

More information can be found in the files themselves.

## Jupyter notebooks (.ipynb)

`Inference.ipynb` shows how to use the fine-tuned model for inference.