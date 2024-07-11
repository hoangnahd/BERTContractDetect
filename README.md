## Introduction
#### This project leverages machine learning, specifically a BERT model, to detect vulnerabilities in smart contracts. The methodology is based on the paper [ASSBert: Active and semi-supervised bert for smart contract vulnerability detection](https://www.sciencedirect.com/science/article/abs/pii/S221421262300008X) , which provides a detailed approach to using BERT for this task. The aim is to improve the security of smart contracts by identifying potential vulnerabilities early in the development process.
## Features
#### Vulnerability Detection:
Identify various types of vulnerabilities in smart contracts using a BERT model.
#### BERT Model Customization:
Generate a vocabulary tailored to the Solidity programming language for use with BERT.
#### Source Code Normalization:
Clean unwanted blank spaces, line breaks, and comments from the dataset.
Normalize contract names, modifier names, function names, and variable names with scope awareness. Standardize to a consistent format (e.g., VAR1, VAR2, FUN1, FUN2, ...).
Utilize the solidity-parser module to create an Abstract Syntax Tree (AST) for easier scope access during normalization.

## Usage
Run the script in different modes: train, predict, or eval. Below are the examples of how to use each mode.

### 1. Training Mode
python3 run.py --mode train --dataset dataset/train_test_data.json
### 2. Prediction Mode
python3 run.py --mode predict --model_path model/bert.pth --input_file input_file.sol
### 3. Evaluation Mode
python3 run.py --mode eval --eval_dataset eval_data.json --model_path model/bert.pth
### 4. Arguments
**--mode:** Execution mode: train, predict, or eval.\
**--dataset:** Path to the dataset JSON file (for training).\
**--eval_dataset:** Path to the evaluation dataset JSON file (for eval mode).\
**--bert_model_name:** Name of the BERT model (default: bert-base-uncased).\
**--num_classes:** Number of classes for classification (default: 2).\
--max_length: Maximum length of the input sequences (default: 128).\
--batch_size: Batch size for training/testing (default: 16).\
--num_epochs: Number of epochs for training (default: 4).\
--learning_rate: Learning rate for the optimizer (default: 2e-5).\
--model_path: Path to the trained model (for prediction and evaluation) (default: model/bert.pth).\
--input_file: Path to the input file for prediction (for predict mode).
