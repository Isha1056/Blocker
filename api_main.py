from cgi import test
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS, cross_origin
import json
from bs4 import BeautifulSoup
from sympy import re

import string


source_folder = 'content/'
destination_folder = 'content/'

import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from sklearn import metrics

# Preliminaries

from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator

# Models

import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

# Training

import torch.optim as optim

# Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

import csv

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.stem import WordNetLemmatizer
import pickle
import pandas as pd
import time
import numpy as np
import re
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('wordnet')
from nltk.stem import PorterStemmer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def make_csv(l):

    t = ['class','text']
    x = []
    for i in range(min(100,len(l))):
        x.append([0,l[i]])
        
    # dictionary of lists 
    with open(destination_folder+'/test.csv', 'w', newline='') as f:
      
    # using csv.writer method from CSV package
        write = csv.writer(f)
        
        write.writerow(t)
        write.writerows(x)
 

# Model parameter
def predict_set():
    MAX_SEQ_LEN = 128
    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    # Fields

    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                    fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
    fields = [('class', label_field), ('text', text_field)]

    # TabularDataset

    train, valid, test = TabularDataset.splits(path=source_folder, train='train.csv', validation='dev.csv',
                                            test='test.csv', format='CSV', fields=fields, skip_header=True)
# Iterators

    train_iter = BucketIterator(train, batch_size=16, sort_key=lambda x: len(x.text),
                                device=device, train=True, sort=True, sort_within_batch=True)
    valid_iter = BucketIterator(valid, batch_size=16, sort_key=lambda x: len(x.text),
                                device=device, train=True, sort=True, sort_within_batch=True)
    test_iter = Iterator(test, batch_size=16, device=device, train=False, shuffle=False, sort=False)
    return test_iter


class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name)

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]

        return loss, text_fea


# Save and Load Functions

def save_checkpoint(save_path, model, valid_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_metrics(load_path):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


# Evaluation Function

def evaluate(model, test_loader):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
      for (labels, texts), _ in test_loader:
        #print(labels, texts)
        labels = labels.type(torch.LongTensor)           
        labels = labels.to(device)
        texts = texts.type(torch.LongTensor)  
        texts = texts.to(device)
        output = model(texts, labels)

        _, output = output
        y_pred.extend(torch.argmax(output, 1).tolist())
        y_true.extend(labels.tolist())
        #print(y_pred,y_true)
      

      cm = confusion_matrix(y_true, y_pred, labels=[1,0])
      ax= plt.subplot()
      sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

      ax.set_title('Confusion Matrix')

      ax.set_xlabel('Predicted Labels')
      ax.set_ylabel('True Labels')

      ax.xaxis.set_ticklabels(['SPOILER', 'NON-SPOILER'])
      ax.yaxis.set_ticklabels(['SPOILER', 'NON-SPOILER'])

      #fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
      return y_pred


def predict(l):
    make_csv(l)
    test_iter = predict_set()
    best_model = BERT().to(device)

    load_checkpoint(destination_folder + '/model.pt', best_model)

    output = evaluate(best_model, test_iter)
    
    return output











app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def get_preds():
    pass

def remove_tags(html):
  
    # parse html content
    soup = BeautifulSoup(html, "html.parser")
  
    for data in soup(['style', 'script']):
        # Remove tags
        data.decompose()
    
    # return data by retrieving the tag content
    return list(soup.stripped_strings)
from random import randint
@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method=='POST':
        json1=request.get_json()
        # json1 = {"title": "html", "status": "recieved"}
        #pay = json.dumps(json1)
        page = json1['title']
        # print(json1['title'])
        page = remove_tags(json1['title'])

        page1=[''.join(x for x in i if x in string.printable) for i in page]
        output = predict(page1)
        res = []
        for i in range(len(output)):
            if output[i] == 1:
                if randint(0,1) == 1:
                    res.append(page[i])
        response = jsonify(
            response=res,
            mimetype='application/json'
        )
        return response

if __name__ == "__main__":
    app.run(debug=True)