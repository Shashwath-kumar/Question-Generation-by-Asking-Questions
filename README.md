# Generate Question by Asking Question: A Primal-Dual Approach with Uncommon Word Generation

This project implements a unified primal-dual framework for automatic question generation and question answering, designed to produce high-quality, relevant questions from passages and answers.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Usage](#usage)
- [References](#references)

## Overview

The framework combines question generation (primal) with question answering (dual) and consists of three main components:

- Question generation: Jointly encodes answer and passage, produces question
- Question answering: Re-asks generated question to ensure target answer is obtained
- Knowledge distillation: Improves generalization ability for generating uncommon words

## Finetuned model

- QG model finetuned on SQuAD dataset. [link](https://github.com/Shashwath-kumar/Question-Generation-by-Asking-Questions/releases/download/QG_SQuAD/QG_SQuAD.pt)

## Requirements

To install the required dependencies, run:

```py
pip install -r requirements.txt
```

## Usage

1. To train the model, run:
```py
python train.py
```
2. To evaluate the model with BLEU score, run:
```py
python eval.py
```
3. To perform inference with the trained model, run:
```py
python inference.py
```
4. The demo.ipynb notebook provides an interactive environment to test the model.

## References

This is an unofficial implementation of the following EMNLP paper:

```
@inproceedings{wang-etal-2022-learning-generate,
    title = "Learning to Generate Question by Asking Question: A Primal-Dual Approach with Uncommon Word Generation",
    author = "Wang, Qifan  and
      Yang, Li  and
      Quan, Xiaojun  and
      Feng, Fuli  and
      Liu, Dongfang  and
      Xu, Zenglin  and
      Wang, Sinong  and
      Ma, Hao",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.4",
    pages = "46--61",
}
```
