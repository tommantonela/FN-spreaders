import torch
from torch.nn.functional import relu
from torch_geometric.nn import GCNConv
from torch_geometric.nn.aggr import AttentionalAggregation


class SpreaderPerdictor(torch.nn.Module):
    """
    :user_features: number of features for the user
    :user_graph_features: number of features for the user
    :gcn_users: list of the units for user GCNs
    :gcn_tweets: list of the units for user GCNs
    :dense: list of size for the last dense/linear layers
    """
    def __init__(self, user_features, user_graph_features, gcn_users, gcn_tweets, 
                dense, text_embd_size=768, sigmoid=True):
        super().__init__()
        #User social part
        self.gcn_users = []
        prev_features = user_graph_features
        for units in gcn_users:
            self.gcn_users.append(GCNConv(prev_features, units))
            prev_features = units
        user_gcn_out = prev_features
        self.gcn_users = torch.nn.ModuleList(self.gcn_users)
        self.user_aggregation = AttentionalAggregation(
            torch.nn.Sequential(torch.nn.Linear(user_gcn_out, user_gcn_out), 
                                torch.nn.Linear(user_gcn_out, user_gcn_out)))
        #Tweet Part
        self.gcn_tweets = []
        prev_features = text_embd_size #bert size
        for units in gcn_tweets:
            self.gcn_tweets.append(GCNConv(prev_features, units))
            prev_features = units
        self.gcn_tweets = torch.nn.ModuleList(self.gcn_tweets)
        tweet_gcn_out = prev_features
        self.tweets_aggregation = AttentionalAggregation(
            torch.nn.Sequential(torch.nn.Linear(tweet_gcn_out, tweet_gcn_out), 
                                torch.nn.Linear(tweet_gcn_out, tweet_gcn_out)))
        self.tweets_gru = torch.nn.GRU(tweet_gcn_out, tweet_gcn_out, batch_first=True)
        #Dense Part
        prev_features = user_features + user_gcn_out + tweet_gcn_out
        self.dense = []
        for v in dense:
            self.dense.append(torch.nn.Linear(prev_features, v))
            prev_features = v
        self.dense = torch.nn.ModuleList(self.dense)  
        self.pred = torch.nn.Linear(prev_features, 1)
        self.sigmoid = sigmoid
        pass
    
    def forward(self, user_graph, tweet_graph, user_features):
        #Users social interactions 
        c_user_graph_features = user_graph.x
        for gcn_u in self.gcn_users:
            c_user_graph_features = gcn_u(c_user_graph_features, user_graph.edge_index)
            c_user_graph_features = relu(c_user_graph_features)
        c_user_graph_features = self.user_aggregation(c_user_graph_features, ptr=user_graph.ptr) 
        
        #conversations’ propagation trees (per user)
        tweet_features = []
        for graph in tweet_graph:
            c_tweet_features = graph.x
            for gcn_t in self.gcn_tweets:
                c_tweet_features = gcn_t(c_tweet_features, graph.edge_index)
                c_tweet_features = relu(c_tweet_features)
            part_tweets = self.tweets_aggregation(c_tweet_features, ptr=graph.ptr)
            part_tweets = self.tweets_gru(part_tweets)[1]
            tweet_features.append(part_tweets)
        tweet_features = torch.vstack(tweet_features)
        #Concat
        full_data = torch.concat([user_features,
                                  c_user_graph_features,  
                                  tweet_features], dim=1)
        for dense in self.dense:
            full_data = dense(full_data)
            full_data = relu(full_data)
        full_data = self.pred(full_data)
        if self.sigmoid:
            full_data = torch.sigmoid(full_data)
        return full_data