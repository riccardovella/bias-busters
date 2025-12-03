#######################################################
# Implementation of a classification model using contextual embeddings.
# 
# Here there should be the logic for training and evaluating a classifier
# that uses contextual embeddings as input features.
#######################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer

import pandas as pd
import numpy as np
from tqdm import tqdm

# bert_model_name = 'prajjwal1/bert-tiny'

def _load_data_twitter(file_path):
    # Load the training data
    train_data = pd.read_csv(file_path,
                            header=None, 
                            names=['id', 'entity', 'sentiment', 'text'])

    # Remove irrelevant sentiment rows
    train_data = train_data[train_data['sentiment'] != 'Irrelevant']

    labels = train_data['sentiment'].tolist()

    # turn data_y into numerical labels
    label_mapping = {'Positive': 2, 'Neutral':1, 'Negative': 0}

    data_x = np.array(train_data['text'].tolist())
    data_y = np.array([label_mapping[label] for label in labels])

    return data_x, data_y

class TwitterDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
    
class SentenceClassifier(nn.Module):
    def __init__(self, bert_model_name, n_classes, debiaser=None):
        super(SentenceClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.debiaser = debiaser

    def forward(self, input_ids, attention_mask):
        # get the sentence embeddings from BERT
        pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )['pooler_output'] # this is the embedding for [CLS] token

        # apply debiasing transform if provided
        if self.debiaser is not None:
            pooled_output = self.debiaser(pooled_output)

        output = self.drop(pooled_output)
        return self.out(output)

def _train_model(
        model, loader, criterion, optimizer, num_epochs, device, verbose=True):
    def train_epoch():
        model.train()
        total_loss = 0

        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        return total_loss / len(loader)

    # Train the model
    for epoch in range(num_epochs):
        if verbose:
            print(f'Epoch {epoch + 1}/{num_epochs}')
        train_loss = train_epoch()
        if verbose:
            print(f'Train loss: {train_loss}')

def _eval_model(model, data_loader, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            y_true = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            y_pred = torch.argmax(outputs, dim=1)
            correct_predictions += torch.sum(y_pred == y_true)
            total_predictions += y_true.size(0)

    return correct_predictions.double() / total_predictions

def evaluate_on_sentence_classification(bert_model_name, debiaser=None,
                                        device=None, num_epochs=3, 
                                        verbose=True):
    '''
    Evaluate the model on a sentence classification task.
    
    Args:
        bert_model_name (str): Name of the BERT model to use.
        debiaser (torch.nn.Module, optional): 
            A debiasing transformation to apply to the embeddings.
            Defaults to None.
        device (torch.device, optional): 
            Device to run the model on. Defaults to None.
        num_epochs (int, optional): 
            Number of epochs to train the model. Defaults to 3.
    '''
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    x_train, y_train = _load_data_twitter('data/twitter_sentiment/twitter_training.csv')
    x_val, y_val = _load_data_twitter('data/twitter_sentiment/twitter_validation.csv')

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    # Create datasets
    train_dataset = TwitterDataset(x_train, y_train, tokenizer)
    val_dataset = TwitterDataset(x_val, y_val, tokenizer)

    # Create data loaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    model = SentenceClassifier(
        bert_model_name, n_classes=3, debiaser=debiaser)
    model = model.to(device)

    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)

    # Define loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Train the model
    num_epochs = 3
    _train_model(
        model, train_loader, criterion, optimizer, num_epochs, verbose=verbose)

    # Evaluate the model
    val_accuracy = _eval_model(model, val_loader, device)
    return val_accuracy