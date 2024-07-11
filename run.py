from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from BertModel.BertClassifier import BERTClassifier
from BertModel.BertClassifier import TextClassificationDataset
from tqdm import tqdm
import torch
from torch import nn
import pandas as pd
import argparse
import json

with open("vocab.txt", "r", encoding="utf-8") as f:
    unique_list = f.read().splitlines()

def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    for batch in tqdm(data_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

def predict_sentiment(text, model, tokenizer, device, max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
      outputs = model(input_ids=input_ids, attention_mask=attention_mask)
      _, preds = torch.max(outputs, dim=1)
    return "positive" if preds.item() == 1 else "negative"

def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)

def main(args):
    try:
        if args.mode == 'train':
            # Load and preprocess the dataset
            with open(args.dataset, "r") as f:
                jsonData = json.load(f)

            # Print out the parameters to confirm they are set correctly
            print(f"Training BERT Model: {args.bert_model_name}")
            print(f"Number of Classes: {args.num_classes}")
            print(f"Max Length: {args.max_length}")
            print(f"Batch Size: {args.batch_size}")
            print(f"Number of Epochs: {args.num_epochs}")
            print(f"Learning Rate: {args.learning_rate}")

            train_data, val_data = train_test_split(jsonData, test_size=0.2, random_state=42)
            tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)

            train_dataset = TextClassificationDataset(train_data, tokenizer, args.max_length)
            val_dataset = TextClassificationDataset(val_data, tokenizer, args.max_length)

            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = BERTClassifier(args.bert_model_name, args.num_classes, unique_list).to(device)

            optimizer = AdamW(model.parameters(), lr=args.learning_rate)
            total_steps = len(train_dataloader) * args.num_epochs
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

            import time
            start = time.time()
            for epoch in range(args.num_epochs):
                print(f"Epoch {epoch + 1}/{args.num_epochs}")

                # Train the model on the training dataset
                train(model, train_dataloader, optimizer, scheduler, device)

                # Evaluate the model on the validation dataset
                accuracy, report = evaluate(model, val_dataloader, device)

                # Print validation accuracy and evaluation report
                print(f"Validation Accuracy: {accuracy:.4f}")
                print(report)

                # Save the model for each epoch
            model_save_path = f"model/bert.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at: {model_save_path}")
            print("Training Time: " + str(time.time() - start))

        elif args.mode == 'predict':
            if not args.input_file:
                raise ValueError("Prediction mode requires --input_file argument specifying the file to predict.")

            # Load the model
            tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)
            model = BERTClassifier(args.bert_model_name, args.num_classes).to(device)
            model.load_state_dict(torch.load(args.model_path))  # Assuming you have a model path argument
            model.eval()

            # Prepare input for prediction
            with open(args.input_file, "r") as f:
                input_text = f.read()

            # Tokenize input text
            inputs = tokenizer(input_text, return_tensors="pt", max_length=args.max_length, truncation=True)
            inputs.to(device)

            # Perform prediction
            with torch.no_grad():
                outputs = model(**inputs)

            # Assuming a binary classification task
            predicted_class = torch.argmax(outputs.logits).item()
            print(f"Predicted Class: {predicted_class}")

        elif args.mode == 'eval':
            # Load the model for evaluation
            tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)
            model = BERTClassifier(args.bert_model_name, args.num_classes).to(device)
            model.load_state_dict(torch.load(args.model_path))  # Assuming you have a model path argument
            model.eval()

            # Load evaluation dataset
            with open(args.eval_dataset, "r") as f:
                eval_data = json.load(f)

            eval_dataset = TextClassificationDataset(eval_data, tokenizer, args.max_length)
            eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)

            # Evaluate the model on the evaluation dataset
            accuracy, report = evaluate(model, eval_dataloader, device)

            # Print evaluation accuracy and report
            print(f"Evaluation Accuracy: {accuracy:.4f}")
            print(report)

        else:
            raise ValueError(f"Invalid mode: {args.mode}. Please choose from 'train', 'predict', or 'eval'.")
    
    except ValueError as ve:
        print(f"Error: {str(ve)}")
        exit(1)  # Exit with a non-zero status code to indicate failure

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BERT Model Training, Prediction, and Evaluation.")

    parser.add_argument('--mode', choices=['train', 'predict', 'eval'], default='train', help='Execution mode: train, predict, or eval.')
    parser.add_argument('--dataset', type=str, default='dataset/train_test_data.json', help='Path to the dataset JSON file (for training).')
    parser.add_argument('--eval_dataset', type=str, default='dataset/eval_data.json', help='Path to the evaluation dataset JSON file (for eval mode).')
    parser.add_argument('--bert_model_name', type=str, default='bert-base-uncased', help='Name of the BERT model.')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes for classification.')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum length of the input sequences.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training/testing.')
    parser.add_argument('--num_epochs', type=int, default=4, help='Number of epochs for training.')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for the optimizer.')
    parser.add_argument('--model_path', type=str, default='model/bert.pth', help='Path to the trained model (for prediction and evaluation).')
    parser.add_argument('--input_file', type=str, default=None, help='Path to the input file for prediction (for predict mode).')

    args = parser.parse_args()
    main(args)
