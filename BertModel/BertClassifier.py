from transformers import BertTokenizer, BertModel
from torch import nn
from torch.utils.data import Dataset
import torch

class TextClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.texts = [contract["source"] for contract in data]
        self.labels = [1 if contract["label"] else 0 for contract in data]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label)
        }
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes, new_vocab):
        super(BERTClassifier, self).__init__()

        # Load the BERT model and tokenizer
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        # Extend the vocabulary of the tokenizer with new_vocab
        self.tokenizer.add_tokens(new_vocab)

        # Resize the token embeddings matrix of the model
        self.bert.resize_token_embeddings(len(self.tokenizer))

        # Rest of the model setup
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits