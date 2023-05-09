Annotating corpus data
---

This repository accompanies the paper ["Using Machine Learning to automate data annotation in Corpus Linguistics: a case study with MacBERTh"](link.to.journal). It provides the code and data necessary to replicate the case study presented in the paper, and also serves as a guide to conduct the same type of semi-automated analysis of corpus data.

Our example case study focusses on _ing_-forms retrieved from various corpora of historical English. In order to automatically classify these _ing_-forms into a set of categories, we work with MacBERTh, which is a type of historical language model based on the 'transformer' architecture. For other models trained on other languages and language varieties, you can browse the [Hugging Face model repository](https://huggingface.co/models).

Besides the annotated corpus data needed to replicate our example case study, this repository contains python scripts and jupyter notebooks. The python scripts should be run from [the command line](https://melaniewalsh.github.io/Intro-Cultural-Analytics/01-Command-Line/01-The-Command-Line.html). Information on how to install Jupyter (and Python) can be found [here](https://melaniewalsh.github.io/Intro-Cultural-Analytics/02-Python/01-Install-Python.html), for example.

To run the python scripts from the command line, a number of arguments should be specified. You can learn more about which arguments are required by running `python finetune.py --help`, `python finetune-cv.py --help` or `python preprocess.py --help` from the command line.

The repository relies on the `transformers` library.


# Overview of files

## Python scripts (.py)

`preprocess.py` allows you to extract embeddings from an input file (specified by you) by means of a model of your choosing. It also provides functions needed to run the other scripts (`finetune.py` and `finetune-cv.py`).

`finetune.py` can be used to perform fine-tuning on a file with training data. Such training data consists of input examples and accompanying labels (following a classification scheme of your choosing). After fine-tuning, the script lets you use the model to tag an (unlabelled) test file.

`finetune-cv.py` performs cross-validation over the input training file. This script is useful if you want to assess the classification accuracy of your fine-tuned model.

More information can be found in the files themselves (and by using the `--help` argument).

## Jupyter notebooks (.ipynb)

`Inference.ipynb` shows how to use the fine-tuned model for inference (that is, how a model can be used to label new examples after a model has been fine-tuned). 

`Visualization.ipynb` shows how to visualize the results of the fine-tuning scripts.