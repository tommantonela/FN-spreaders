# Training

Call the following script for training the model: 

```bash
python train.py -g USER_GRAPH_FEATURES -u USER_FEATURES -ug USER_GRAPH -tg TWEET_GRAPH -p PARTITION -tp TEMPORAL_PARTITION -te TWEET_EMBEDINGS -tm TWEET_EMBEDINGS_MAP
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
  -tp TEMPORAL_PARTITION, --temporal_partition TEMPORAL_PARTITION
                        Path to the temporal partition of the users/tweets
  -te TWEET_EMBEDINGS, --tweet_embedings TWEET_EMBEDINGS
                        Path to the tweet embedding matrix
  -tm TWEET_EMBEDINGS_MAP, --tweet_embedings_map TWEET_EMBEDINGS_MAP
                        Path to the tweet id -> position in the tweet embedding matrix map
```

Notice that both, `-u` and `-g` receive a partial path name. For example, using the `tree__nn` features for the graph, and the features file are called:
1. **Slot 0**: `../data/feature_sets/score_slot/fibvid_features_tree__nn_user_n_split_temporal_10__0_score_slot.pickle` 
1. **Slot 1**: `../data/feature_sets/score_slot/fibvid_features_tree__nn_user_n_split_temporal_10__1_score_slot.pickle` 
1. **Slot 2**: `../data/feature_sets/score_slot/fibvid_features_tree__nn_user_n_split_temporal_10__2_score_slot.pickle` 
1. **Slot 3**: `../data/feature_sets/score_slot/fibvid_features_tree__nn_user_n_split_temporal_10__3_score_slot.pickle` 
1. **Slot 4**: `../data/feature_sets/score_slot/fibvid_features_tree__nn_user_n_split_temporal_10__4_score_slot.pickle` 
1. **Slot 5**: `../data/feature_sets/score_slot/fibvid_features_tree__nn_user_n_split_temporal_10__5_score_slot.pickle` 
1. **Slot 6**: `../data/feature_sets/score_slot/fibvid_features_tree__nn_user_n_split_temporal_10__6_score_slot.pickle` 
1. **Slot 7**: `../data/feature_sets/score_slot/fibvid_features_tree__nn_user_n_split_temporal_10__7_score_slot.pickle` 
1. **Slot 8**: `../data/feature_sets/score_slot/fibvid_features_tree__nn_user_n_split_temporal_10__8_score_slot.pickle` 
1. **Slot 9**: `../data/feature_sets/score_slot/fibvid_features_tree__nn_user_n_split_temporal_10__9_score_slot.pickle` 

`-g` parameter should be `../data/feature_sets/score_slot/fibvid_features_tree__nn`.

The following is a call example: 

```bash
python train.py -g ../data/feature_sets/score_slot/fibvid_features_tree__nn -u ../data/feature_sets/score_slot/fibvid_features_all_features -p ../data/users_profile_split_temporal.pickle -ug ../data/gg_users_full.gpickle -tg ../data/gg_tweets_full.gpickle -te ../data/bert_embs/pooler_output.npz -tm ../data/bert_embs/tweetId_pos.pickle -tp ../data/user_n_split_temporal_10.pickle
```


# Evaluation

The evaluation script `eval.py` has the same parameters as the `train.py` script. It will generate an output file called `temporal_results.csv`.