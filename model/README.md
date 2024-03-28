# Training

Call the following script for training the model: 

```bash
python train.py -g USER_GRAPH_FEATURES -u USER_FEATURES -ug USER_GRAPH -tg TWEET_GRAPH -p PARTITION -te TWEET_EMBEDINGS -tm TWEET_EMBEDINGS_MAP
```

Where:

```
  -g USER_GRAPH_FEATURES, --user_graph_features USER_GRAPH_FEATURES
                        Path to the feaures for the user graph
  -u USER_FEATURES, --user_features USER_FEATURES
                        Path to the feaures for the user
  -ug USER_GRAPH, --user_graph USER_GRAPH
                        Path to the user graph
  -tg TWEET_GRAPH, --tweet_graph TWEET_GRAPH
                        Path to the tweet graph
  -p PARTITION, --partition PARTITION
                        Path to the user partition file (train/test)
  -te TWEET_EMBEDINGS, --tweet_embedings TWEET_EMBEDINGS
                        Path to the tweet embedding matrix
  -tm TWEET_EMBEDINGS_MAP, --tweet_embedings_map TWEET_EMBEDINGS_MAP
                        Path to the tweet id -> position in the tweet embedding matrix map
```

The following is a call example: 

```bash
python train.py -g ../data/2024/feature_sets/score_final/fibvid_features_tree__nn_score_final.pickle -u ../data/2024/feature_sets/score_final/fibvid_features_all_features_score_final.pickle -p ../data/users_profile_split_temporal.pickle -ug ../data/gg_users_full.gpickle -tg ../data/gg_tweets_full.gpickle -te ../data/bert_embs/pooler_output.npz -tm ../data/bert_embs/tweetId_pos.pickle

```

# Evaluation

The evaluation script `eval.py` has the same parameters as the `train.py` script. It will output the results in the standard output (console by default).