#!/usr/bin/env python
# coding: utf-8

from tqdm.auto import tqdm
import torch
import pickle
import numpy as np
import pandas as pd
from data import TweetEmbeddings, UserDataset, load_data, merge_dataset_instances_eval, preprocess
from metric import METRICS, compute_metrics
from time_slot import filter_time_slots
from model import SpreaderPerdictor


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


test_graph = load_pickle(user_graph_name)
tweet_graph = load_pickle(tweet_graph_name)
user_partitions = load_pickle(partition_name)

time_slots = load_pickle(temporal_partition_name)


@torch.no_grad()
def predict(test_graph, tweet_graph, 
            user_partitions,
            user_graph_features_temp, user_features_temp, labels,
            threshold, partition, time_slots, slot):
    current_dir = f'model_temporal_{slot}'
    _, _, train_partitions = filter_time_slots(slot, time_slots,test_graph, tweet_graph, user_partitions[partition][0])
    test_graph, tweet_graph, user_partitions = filter_time_slots(slot, time_slots,test_graph, tweet_graph, user_partitions[partition][0], not_in=True, valid_users=set(test_graph.nodes))

    user_features_preprocessed  = preprocess(user_features_temp, list(train_partitions))
    user_graph_features_preprocessed = preprocess(user_graph_features_temp, list(train_partitions))
    #user_features_preprocessed = user_features
    ds = UserDataset(test_graph, tweet_graph, user_partitions, 
                     user_features_preprocessed,
                     user_graph_features_preprocessed, 
                     TweetEmbeddings(tweet_embedings_map_name, 
                                     tweet_embedings_name), labels=labels, inverted_graph=False, threshold=threshold, max_depth_user=3, max_tweets=5)
    dl = torch.utils.data.DataLoader(ds, batch_size=500, collate_fn=merge_dataset_instances_eval, num_workers=0, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpreaderPerdictor(len(user_features_temp.columns), len(user_graph_features_preprocessed.columns),
                              [32, 32, 32], [100], [70, 70], sigmoid=False)
    load_data(model, current_dir)
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

results = []
for slot in range(10):
    user_graph_features_temp = pd.read_pickle(f'{graph_features_name}_user_n_split_temporal_10__{slot}_score_slot.pickle')
    user_graph_features_temp = user_graph_features_temp.drop('score_spreader', axis=1)
    user_features_temp = pd.read_pickle(f'{user_features_name}_user_n_split_temporal_10__{slot}_score_slot.pickle')
    labels = user_features_temp['score_spreader'].apply(lambda x: 1 if x > 0.5 else -1)
    user_features_temp = user_features_temp.drop('score_spreader', axis=1)
    df = predict(test_graph, tweet_graph, user_partitions, 
                 user_graph_features_temp, user_features_temp, labels,
                 0.5, partition, time_slots, slot)

    y_pred = df['y_pred']
    y_true = df['y_true']
    metrics = compute_metrics(y_true, y_pred)
    metrics['slot'] = slot
    metrics['positives'] = sum(y_true)
    metrics['negatives'] = len(y_true) - sum(y_true)
    results.append(metrics)

res = pd.DataFrame(data=results, columns=['Slot', 'positives', 'negatives'] + METRICS)
res.to_csv('temporal_results.csv')

