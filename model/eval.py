#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
from pymongo import MongoClient
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import torch
from torch.nn.functional import relu
import pickle
import numpy as np
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.nn.aggr import AttentionalAggregation
from collections import deque
import pandas as pd
import os
from functools import partial
from sklearn.utils import compute_class_weight
from sklearn.preprocessing import StandardScaler, RobustScaler
from torch_geometric.data import Data, Batch


# In[3]:

import argparse

from data import TweetEmbeddings, UserDataset, merge_dataset_instances_eval, preprocess
from model import SpreaderPerdictor

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--user_graph_features', required=True, help='Path to the feaures for the user graph')
parser.add_argument('-u', '--user_features', required=True, help='Path to the feaures for the user')
parser.add_argument('-ug', '--user_graph', required=True, help='Path to the user graph')
parser.add_argument('-tg', '--tweet_graph', required=True, help='Path to the tweet graph')
parser.add_argument('-p', '--partition', required=True, help='Path to the user partition file (train/test)')
parser.add_argument('-te', '--tweet_embedings', required=True, help='Path to the tweet embedding matrix')
parser.add_argument('-tm', '--tweet_embedings_map', required=True, help='Path to the tweet id -> position in the tweet embedding matrix map')

graph_features_name = parser.parse_args().user_graph_features
user_features_name =  parser.parse_args().user_features
user_graph_name = parser.parse_args().user_graph
tweet_graph_name = parser.parse_args().tweet_graph
partition_name = parser.parse_args().partition
tweet_embedings_name = parser.parse_args().tweet_embedings
tweet_embedings_map_name = parser.parse_args().tweet_embedings_map



def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


test_graph = load_pickle(user_graph_name)
tweet_graph = load_pickle(tweet_graph_name)
user_graph_features = pd.read_pickle(graph_features_name)
user_graph_features = user_graph_features.drop('score_spreader', axis=1)
user_features = pd.read_pickle(user_features_name)
user_features = user_features.drop('score_spreader', axis=1)
user_partitions = load_pickle(partition_name)


@torch.no_grad()
def predict(test_graph, tweet_graph, user_partitions, user_graph_features_preprocessed, user_features, 
    threshold, partition):
    user_features_preprocessed  = preprocess(user_features, list(user_partitions[partition][0]))
    user_graph_features_preprocessed = preprocess(user_graph_features, list(user_partitions[partition][0]))
    #user_features_preprocessed = user_features
    
    ds = UserDataset(test_graph, tweet_graph, user_partitions[partition][1], 
                     user_features_preprocessed,
                     user_graph_features_preprocessed, 
                     TweetEmbeddings(tweet_embedings_map_name, 
                                     tweet_embedings_name), inverted_graph=False, threshold=threshold, max_depth_user=3, max_tweets=5)
    dl = torch.utils.data.DataLoader(ds, batch_size=10, collate_fn=merge_dataset_instances_eval, num_workers=0, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpreaderPerdictor(len(user_features.columns), len(user_graph_features.columns),
                              [32, 32, 32], [100], [70, 70], sigmoid=False)
    model.load_state_dict(torch.load(f'checkpoints/model-epoch_20.pt'))
    model = model.to(device)
    model.eval()
    user = list()
    y_true = list()
    y_pred = list()
    for x, y in tqdm(dl):
        x_proc = {}
        x_proc['user_graph'] = x['user_graph'].to(device)
        x_proc['user_features'] = x['user_features'].to(device)
        x_proc['tweet_graph'] = [b.to(device) for b in x['tweet_graph']]
        # Make the predictions
        pred = model(**x_proc)
        user.extend(x['user'])
        y_true.append(y.cpu().numpy()[:, 0])
        y_pred.append(pred.cpu().numpy()[:, 0])
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    df = pd.DataFrame(data={'User': user, 'y_true': np.where(y_true>0, 1, 0), 'y_pred': 1 / (1 + np.exp(-y_pred)), 
                            'y_true_raw': y_true, 'y_pred_raw': y_pred}, 
                      columns=['User', 'y_true', 'y_pred', 'y_true_raw', 'y_pred_raw'])
    df.set_index('User')
    return df


# In[9]:


partition = 0 
df = predict(test_graph, tweet_graph, user_partitions, user_graph_features, user_features, 0.5, partition)

# In[ ]:

from sklearn.metrics import roc_auc_score, recall_score, precision_score, balanced_accuracy_score, matthews_corrcoef

y_pred = df['y_pred']
y_true = df['y_true']

print(f'Evaluating for {graph_features_name} - {user_features_name}')
print(f'ROC {roc_auc_score(y_true, y_pred)}')
print(f'Recall {recall_score(y_true, y_pred > 0.5)}')
print(f'Precision {precision_score(y_true, y_pred > 0.5)}')
print(f'Balanced Accuracy {balanced_accuracy_score(y_true, y_pred > 0.5)}')
print(f'Recall (Weighted) {recall_score(y_true, y_pred > 0.5, average="weighted")}')
print(f'Precision  (Weighted) {precision_score(y_true, y_pred > 0.5, average="weighted")}')
print(f'Matthews Coef {matthews_corrcoef(y_true, y_pred > 0.5)}')
