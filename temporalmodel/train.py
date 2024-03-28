#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tqdm.auto import tqdm
import torch
import pickle
import numpy as np
import pandas as pd
import os
from functools import partial
from sklearn.utils import compute_class_weight
import random
from data import TweetEmbeddings, UserDataset, load_data, merge_dataset_instances_train, preprocess
from time_slot import filter_time_slots
from model import SpreaderPerdictor

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# In[2]:

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--user_graph_features', required=True, help='Path to the feaures for the user graph')
parser.add_argument('-u', '--user_features', required=True, help='Path to the feaures for the user')
parser.add_argument('-ug', '--user_graph', required=True, help='Path to the user graph')
parser.add_argument('-tg', '--tweet_graph', required=True, help='Path to the tweet graph')
parser.add_argument('-p', '--partition', required=True, help='Path to the user partition file (train/test)')
parser.add_argument('-tp', '--temporal_partition', required=True, help='Path to the temporal partition of the users/tweets')
parser.add_argument('-te', '--tweet_embedings', required=True, help='Path to the tweet embedding matrix')
parser.add_argument('-tm', '--tweet_embedings_map', required=True, help='Path to the tweet id -> position in the tweet embedding matrix map')

graph_features_name = parser.parse_args().user_graph_features
user_features_name =  parser.parse_args().user_features
user_graph_name = parser.parse_args().user_graph
tweet_graph_name = parser.parse_args().tweet_graph
partition_name = parser.parse_args().partition
temporal_partition_name = parser.parse_args().temporal_partition
tweet_embedings_name = parser.parse_args().tweet_embedings
tweet_embedings_map_name = parser.parse_args().tweet_embedings_map


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


train_graph = load_pickle(user_graph_name)
tweet_graph = load_pickle(tweet_graph_name)
user_partitions = load_pickle(partition_name)

time_slots = load_pickle(temporal_partition_name)


def train(model, device, train, optimizer, loss_fuction, epochs, weights, checkpoint=None):
    for e in range(1, epochs + 1):
        print(f'Epoch {e} of {epochs}')
        tbar = tqdm(train)
        losses = 0
        cant = 0
        for x, y in tbar:
            optimizer.zero_grad()
            x_proc = {}
            x_proc['user_graph'] = x['user_graph'].to(device)
            x_proc['user_features'] = x['user_features'].to(device)
            x_proc['tweet_graph'] = [b.to(device) for b in x['tweet_graph']]
            y_proc = y.to(device)
            # Make the predictions
            pred = model(**x_proc)#, user_tweets=x['tweet_len'])
            # Calculate loss
            weight = weights[torch.where(y_proc[:, 0].long() > 0, 1, 0)]
            loss = loss_fuction(pred, y_proc)
            loss = torch.mean(weight * loss)
            # Calculate the gradients
            loss.backward()
            # Optimize
            optimizer.step()
            # Update print information
            losses += loss.item()
            cant += 1
            tbar.set_postfix(loss=(losses / cant))
        loss = {losses / cant}
        print(f'Final average loss {loss}')
        if checkpoint is not None:
            checkpoint(model=model, epoch=e)
    pass


# In[49]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HingeLoss:
    def __init__(self, margin=1, device=device):
        self.margin = torch.scalar_tensor(margin, dtype=torch.float32, device=device)
        self.zero = torch.scalar_tensor(0, dtype=torch.float32, device=device)

    def __call__(self, pred, true):
        res = self.margin - pred * true
        res = torch.where(res >= self.zero, res, self.zero)
        return res

def checkpoint(model, epoch, model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = f'{model_dir}{os.sep}model-epoch_{epoch}.pt'
    torch.save(model.state_dict(), model_path)


def train_for_partition(user_graph, tweet_graph, partition, slot, max_epochs=5):
    prev_dir = f'model_temporal_{slot-1}'
    checkpoint_p = partial(checkpoint, model_dir=f'model_temporal_{slot}')
    
    user_graph_features = pd.read_pickle(f'{graph_features_name}_user_n_split_temporal_10__{slot}_score_slot.pickle')
    user_graph_features = user_graph_features.drop('score_spreader', axis=1)
    user_features = pd.read_pickle(f'{user_features_name}_user_n_split_temporal_10__{slot}_score_slot.pickle')
    labels = user_features['score_spreader'].apply(lambda x: 1 if x > 0.5 else -1)
    user_features = user_features.drop('score_spreader', axis=1)
    
    user_graph, tweet_graph, users = filter_time_slots(slot, time_slots,user_graph, tweet_graph, user_partitions[partition][0])
    user_features_preprocessed = preprocess(user_features, users)
    user_graph_features_preprocessed = preprocess(user_graph_features, users)
    #user_graph_features_preprocessed = user_graph_features
    ds = UserDataset(user_graph, tweet_graph, users,
                     user_features_preprocessed,
                     user_graph_features_preprocessed, 
                     TweetEmbeddings(tweet_embedings_map_name, 
                                     tweet_embedings_name), 
                     labels=labels,
                     inverted_graph=False, max_depth_user=3, max_tweets=5, threshold=0.5)
    dl = torch.utils.data.DataLoader(ds, batch_size=10, collate_fn=merge_dataset_instances_train, num_workers=0, shuffle=True)
    
    classes = np.asarray([1 if ds.user_graph.nodes[node]['score_graph'] > ds.threshold else 0 for node in ds.users])
 
    weights = compute_class_weight('balanced', classes=[0, 1], y=classes)

    weights = torch.from_numpy(np.asarray(weights)).float().to(device)
    model = SpreaderPerdictor(len(user_features.columns), len(user_graph_features_preprocessed.columns), 
                              [32, 32, 32], [100], [70, 70], sigmoid=False)
    model.to(device)
    load_data(model, prev_dir)
    optimizer = torch.optim.Adam(model.parameters())
    loss = HingeLoss()

    train(model, device, dl, optimizer, loss, max_epochs, weights, checkpoint_p)


for slot in range(10):
    train_for_partition(train_graph, tweet_graph, 0, slot, 2)

