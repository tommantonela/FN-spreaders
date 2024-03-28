from itertools import chain

def process_user_edge_property(data, prop, valid_tweets):
    if prop not in data:
        return False
    prop_d = f'{prop}_date'
    dates = []
    tweets = []
    for t, d in zip(data[prop], data[prop_d]):
        if t in valid_tweets:
            tweets.append(t)
            dates.append(d)
    if len(tweets) == 0:
        del data[prop]
        del data[prop_d]
        return False
    data[prop] = tweets
    data[prop_d] = dates 
    return True

def filter_time_slots(time_step, time_slots, user_graph, tweet_graph, users_list, not_in=False, valid_users=None):
    assert time_step >= 0 and time_step < len(time_slots[1]) 

    users = set()
    tweets = set()
    
    for i in range(0, time_step + 1):
        users.update(time_slots[1][i].keys())
        tweets.update(chain(*time_slots[1][i].values()))

    u_g = user_graph.copy()
    g_users = set(u_g.nodes) - users
    u_g.remove_nodes_from(g_users)
    for u in u_g.nodes:
        if 'tweets' in u_g.nodes[u]:
            u_g.nodes[u]['tweets'] = [x for x in u_g.nodes[u]['tweets'] if x[0] in tweets]
        
    edges_to_remove = set()
    for e in u_g.edges:
        data = u_g.edges[e]
        parent = process_user_edge_property(data, 'parent', tweets)
        mentions = process_user_edge_property(data, 'mentions', tweets)
        replies = process_user_edge_property(data, 'replies', tweets)
        if not (parent or mentions or replies):
            edges_to_remove.add(e)
        else:
            props = (data['parent_date'] if 'parent_date' in data else []) +\
            (data['replies_date'] if 'replies_date' in data else []) +\
            (data['mentions_date'] if 'mentions_date' in data else [])
            data['weight'] = len(props)
            data['min_date'] = min(props)
            data['max_date'] = max(props)
    
    t_g = tweet_graph.copy()
    g_tweets = set(t_g.nodes) - tweets
    t_g.remove_nodes_from(g_tweets)

    if not_in:
        new_users_list = list((users - set(users_list)) & valid_users)
    else:
        new_users_list = [x for x in users_list if x in users]

    return u_g, t_g, new_users_list