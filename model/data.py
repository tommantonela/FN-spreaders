
import networkx as nx
from torch.utils.data import Dataset
from collections import deque
import torch
import pickle
import numpy as np
from torch_geometric.data import Data, Batch
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
import pandas as pd


def tweets_relation(edge_data, relation):
    if relation in edge_data:
        relation_date = f'{relation}_date'
        return [t for t in zip(edge_data[relation], edge_data[relation_date])]
    return []


def invert_graph(graph):
    n_graph = nx.DiGraph()
    for node in graph.nodes:
        n_graph.add_node(node, **graph.nodes[node])
    for edge in graph.edges:
        n_graph.add_edge(*edge[::-1], **graph.edges[edge])
    return n_graph


def create_adyacency_matrix(graph):
    nodes = list(graph.nodes)
    nodes.sort()
    nodes_id = {v: i for i, v in enumerate(nodes)}
    edges = {}
    for n in nodes:
        node_id = nodes_id[n]
        neigbors = [nodes_id[t] for _, t in graph.edges(n)]
        edges[node_id] = neigbors
    return edges, nodes_id


def depth_subgrap(graph, start, max_depth, include_start=True):
    to_process = deque()
    to_process.append((start, 0))
    added = set()
    nodes = list()
    while len(to_process) != 0:
        val = to_process.popleft()
        if val[0] not in added:
            node, depth = val
            added.add(node)
            if include_start or depth > 0:
                nodes.append(node)
            if depth < max_depth:
                for _, next_node in graph.edges(node):
                    to_process.append((next_node, depth + 1))
    return nodes


def depth_inverse_subgrap(graph, start, max_depth):
    to_process = deque()
    to_process.append((start, 0))
    added = set()
    nodes = list()
    while len(to_process) != 0:
        val = to_process.popleft()
        if val[0] not in added:
            node, depth = val
            added.add(node)
            nodes.append(node)
            if depth < max_depth:
                for next_node in graph.pred[node]:
                    to_process.append((next_node, depth + 1))
    return nodes

def get_subgrap_edges(nodes, graph):
    #Maps the internal position on the large graph to the position on the subgraph
    node_map = {u: i for i, u in enumerate(nodes)}
    #Gets for each user in the neigborhood its neigbors
    #and re-map it to the internal position on the subgraph
    edge_u = []
    edge_v = []
    valid_nodes = set(nodes)
    for u in nodes:
        neigbors = [v for v in graph[u] if v in valid_nodes]
        edge_u.extend([node_map[u]] * len(neigbors))
        edge_v.extend([node_map[v] for v in neigbors])
    edges = torch.tensor([edge_u, edge_v]).long()
    return edges


class TweetEmbeddings:

    def __init__(self, id_path, emb_path):
        with open(id_path, 'rb') as f:
            self.id_pos = pickle.load(f)
        self.embs = np.load(emb_path)['embs']
        pass

    def __getitem__(self, tweet_id):
        return self.embs[self.id_pos[tweet_id], :]

    def __contains__(self, tweet_id):
        return tweet_id in self.id_pos
        

class UserDataset(Dataset):
    
    def __init__(self, user_graph, tweet_graph, users, user_features, 
                 user_graph_features, tweet_embeddings,
                 inverted_graph=True,
                 max_depth_user=1, max_depth_tweet=1, max_tweets=None, threshold=0.5):
        self.user_graph = user_graph
        self.tweet_graph = tweet_graph
        self.user_features = user_features
        self.user_graph_features = user_graph_features
        self.users = list(users)
        self.users.sort()
        self.user_tweets = {}
        self._init_tweets()
        if inverted_graph:
            self.user_graph = invert_graph(user_graph)
            self.tweet_graph = invert_graph(tweet_graph)
        self.tweet_embeddings = tweet_embeddings
        self.a_user, self.user_idx = create_adyacency_matrix(self.user_graph)
        self.a_tweet, self.tweet_idx = create_adyacency_matrix(self.tweet_graph)
        self.max_depth_user = max_depth_user
        self.max_depth_tweet = max_depth_tweet
        self.max_tweets = max_tweets
        self.threshold = threshold
        pass
    
    def _init_tweets(self):
        print('Int tweets')
        for u in tqdm(self.users):
            #Tweets que no generan relación
            if 'tweets' in self.user_graph.nodes[u]:
                tweets = list(self.user_graph.nodes[u]['tweets'])
            else:
                tweets = []
            #Tweets que generan relación
            for e in self.user_graph.edges(u):
                edge_data = self.user_graph.edges[e]
                replies = tweets_relation(edge_data, 'replies')
                mentions = tweets_relation(edge_data, 'mentions')
                parent = tweets_relation(edge_data, 'parent')
                full = replies + mentions + parent
            rels = set()
            for t in full:
                if t[0] not in rels:
                    tweets.append(t)
                    rels.add(t[0])
            tweets.sort(key=lambda x:x[1], reverse=False)
            tweets = [x[0] for x in tweets]
            self.user_tweets[u] = tweets
        pass 
    
    def __len__(self):
        return len(self.users)
        
    def __getitem__(self, idx):
        #Users
        user = self.users[idx]
        #Get the subgraph for the user up to max_depth_users elementens
        users = depth_subgrap(self.user_graph, user, self.max_depth_user)
        #Maps the User id to the internal position on the large graph
        users_ids = np.asarray([self.user_idx[u] for u in users])
        #Get edges
        user_edges = get_subgrap_edges(users_ids, self.a_user)
        #Tweets
        tweet_trees = []
        tweets = self.user_tweets[user] if self.max_tweets is None else self.user_tweets[user][:self.max_tweets]
        for t in tweets:
            tree = depth_inverse_subgrap(self.tweet_graph, t, self.max_depth_tweet) + \
                   depth_subgrap(self.tweet_graph, t, self.max_depth_tweet, include_start=True)
            edges = get_subgrap_edges([self.tweet_idx[u] for u in tree], self.a_tweet)
            tweets_embs = np.asarray([
                self.tweet_embeddings[tweet_id] for tweet_id in tree
            ])
            tweets_embs = torch.from_numpy(tweets_embs).float()
            tweet_trees.append(Data(x=tweets_embs, edge_index=edges))

        tweet_trees = Batch.from_data_list(tweet_trees)    
        #Feature 
        user_graph_features = torch.from_numpy(self.user_graph_features.loc[users].values).float()
        user_features = torch.from_numpy(self.user_features.loc[user].values).float()
        #label
        label = 1 if self.user_graph.nodes[user]['score_graph'] > self.threshold else -1
        return {'user_graph': Data(x=user_graph_features, edge_index=user_edges), 'users':users,
                'user_features': user_features, 
                'tweet_graph': tweet_trees,
                'user': user,
                'label': label}
    

def merge_dataset_instances_train(batch):
    #user part
    user_graph = Batch.from_data_list([d['user_graph'] for d in batch])
    user_features = torch.vstack([d['user_features'] for d in batch])
    #bert part
    tweet_graph =[d['tweet_graph'] for d in batch]
    #data copy
    labels = torch.tensor([[d['label']] for d in batch])
    return {'user_graph': user_graph, 
            'user_features': user_features,
            'tweet_graph': tweet_graph
            }, labels


def merge_dataset_instances_eval(batch):
    #user part
    user_graph = Batch.from_data_list([d['user_graph'] for d in batch])
    user_features = torch.vstack([d['user_features'] for d in batch])
    #bert part
    tweet_graph =[d['tweet_graph'] for d in batch]
    #data copy
    labels = torch.tensor([[d['label']] for d in batch])
    return {'user_graph': user_graph, 
            'user_features': user_features,
            'tweet_graph': tweet_graph,
            'user': [x['user'] for x in batch]
            }, labels


def preprocess(df, users):
    df = df.fillna(0)
    train_df = df.loc[users]
    #Columnas con un valor
    one_value_columns = []
    for c in train_df.columns:
        f_value = train_df[c].iloc[0]
        if (train_df[c] == f_value).all():
            one_value_columns.append(c)
    train_df = train_df.drop(one_value_columns, axis=1)
    valid_columns = []
    for c in train_df.columns:
        column = train_df[c]
        if -1 <= min(column) and max(column) <= 1:
            valid_columns.append(c)
    train_df_to_process = train_df.drop(valid_columns, axis=1)
    train_df = train_df[valid_columns]
    #Scalando valores
    scaler = StandardScaler()
    scaler.fit_transform(train_df_to_process)
    #Procesando
    df = df.drop(one_value_columns, axis=1)
    df_freeze = df[valid_columns]
    df_freeze = df_freeze.astype(np.float32)
    df_process = df.drop(valid_columns, axis=1)
    index = df_process.index
    df_process = scaler.transform(df_process)
    df_process = pd.DataFrame(data=np.clip(df_process, -2, 2), 
                              columns=train_df_to_process.columns,
                              index=index)
    df = pd.concat([df_freeze, df_process], axis=1)
    return df