"""
===================
create_vocab
===================
Author: Ruiqi Zhong
Date: 05/042018
This module implements a function that would take in the labeled and unlabeld corpus
and create a data.pkl that stores all the data
"""


from sentence_tokenizer import int_array_rep
import pandas as pd
from dateutil import parser
import pickle as pkl
from preprocess import extract_mentioned_user_name, extract_user_rt
from sklearn.model_selection import StratifiedKFold, train_test_split
import random
import numpy as np

user2property = pkl.load(open('../model/user.pkl', 'rb'))

# a function that checks whether the idx dictionary satisfies the criteria
def assert_idx_correctness(idx_dictionary):
    cv_dictionary = idx_dictionary['cross_val']
    test_idxes = set()
    fold = len(cv_dictionary)
    for fold_idx in range(fold):
        test_ind = set(cv_dictionary[fold_idx]['test_ind'])
        if len(test_ind & test_idxes) != 0:
            raise ValueError('Test indexes overlap between folds')
        test_idxes = test_idxes | test_ind
        train_ind, val_ind = set(cv_dictionary[fold_idx]['train_ind']), set(cv_dictionary[fold_idx]['val_ind'])
        if len(train_ind) + len(val_ind) + len(test_ind) != len(train_ind | val_ind | test_ind):
            raise ValueError('train, val, test overlaps within a fold')
        cv_idxes = train_ind | val_ind | test_ind
    heldout_test_ind = set(idx_dictionary['heldout_test_ind'])
    ensemble_ind = set(idx_dictionary['ensemble_ind'])
    if len(cv_idxes) + len(heldout_test_ind) + len(ensemble_ind) != len(cv_idxes | heldout_test_ind | ensemble_ind):
        raise ValueError('overlap between cross validation, ensember and heldout test')

# print the meta data of the index dictionary for the train-test split
def print_meta_info(idx_dictionary):
    cv_dictionary = idx_dictionary['cross_val']
    heldout_test_size = len(set(idx_dictionary['heldout_test_ind']))
    ensemble_size = len(set(idx_dictionary['ensemble_ind']))
    cv_train_size = len(cv_dictionary[0]['train_ind'])
    cv_val_size = len(cv_dictionary[0]['val_ind'])
    cv_test_size = len(cv_dictionary[0]['test_ind'])
    print('cross validation train set size %s.' % cv_train_size)
    print('cross validation val set size %s.' % cv_val_size)
    print('cross validation test size %s.' % cv_test_size)
    print('ensemble size %s.' % ensemble_size)
    print('heldout test size %s.' % heldout_test_size)

# "appending dict2 to dict1"
def append_dict(dict1, dict2):
    for key in dict2:
        if key not in dict1:
            dict1[key] = dict2[key]

# reading all data from the directories, regardless of where they are from
def retrieve_content(labeled_corpuses, unlabeled_corpuses, verbose):
    all_data = {}
    for corpus_dir in labeled_corpuses + unlabeled_corpuses:
        if verbose:
            print('reading data from %s ...' % corpus_dir)
        df = pd.read_json(corpus_dir)
        records = df.to_dict('records')
        tweetid2tweet = {}
        for record in records:
            tweet_id = record['tweet_id']
            tweetid2tweet[tweet_id] = record
        append_dict(all_data, tweetid2tweet)
    return all_data

# each tweet is now a dictionary of attributes
def process_data_entries(record):
    # tokenizing at different level
    record['word_int_arr'] = int_array_rep(str(record['text']))
    record['char_int_arr'] = int_array_rep(str(record['text']), option='char')

    # indexing the user who made the post
    user_name = record['user_name'].lower()
    if user2property.get(user_name) is None:
        record['user_post'] = 1
    else:
        record['user_post'] = user2property[user_name]['id']

    # indexing users being mentioned
    user_mentions = extract_mentioned_user_name(str(record['text']))
    record['user_mentions'] = []
    for user_name in user_mentions:
        user_name = user_name.lower()
        if user2property.get(user_name) is None:
            record['user_mentions'].append(1)
        else:
            record['user_mentions'].append(user2property[user_name]['id'])

    rt_user = extract_user_rt(str(record['text']))
    if rt_user is not None:
        rt_user = rt_user.lower()
        if user2property.get(rt_user) is None:
            record['user_retweet'] = 1
        else:
            record['user_retweet'] = user2property[rt_user]['id']

    # time posted
    time_str = record['created_at']
    date_time_created = parser.parse(str(time_str))
    record['created_at'] = date_time_created.replace(tzinfo=None)

    # the text is deliberately deleted to ensure coherence of indexing across implementation
    del record['text']
    del record['user_name']

def create_cv_idx(tr, val, idx_dictionary, fold):
    """
    Create train, val, test indexes for each fold

    Parameters
    ----------
    tr: training json file
    val: validation json file
    idx_dictionary: the index dictionary

    """
    idx_dictionary['cross_val'] = [{} for _ in range(fold)]
    df_tr, df_val = pd.read_json(tr), pd.read_json(val)

    tr_val_test_idx = np.array(df_tr['tweet_id'].values.tolist() + df_val['tweet_id'].values.tolist())
    labels = np.array(df_tr['label'].values.tolist() + df_val['label'].values.tolist())

    skf = StratifiedKFold(n_splits=fold)
    skf.get_n_splits(tr_val_test_idx, labels)
    skf.split(tr_val_test_idx, labels)
    fold_idx = 0
    for train_index, test_index in skf.split(tr_val_test_idx, labels):
        train_val_idx, test_idx = tr_val_test_idx[train_index], tr_val_test_idx[test_index]
        train_ind, val_ind = train_test_split(train_val_idx, test_size=0.2, stratify=labels[train_index])
        idx_dictionary['cross_val'][fold_idx]['train_ind'] = train_ind
        idx_dictionary['cross_val'][fold_idx]['val_ind'] = val_ind
        idx_dictionary['cross_val'][fold_idx]['test_ind'] = test_idx
        fold_idx += 1

def create_unlabeled_ind(data_dictionary):
    unlabeled_tweet_id = []
    for tweet_id in data_dictionary:
        if data_dictionary[tweet_id].get('label') is None:
            unlabeled_tweet_id.append(tweet_id)
    random.shuffle(unlabeled_tweet_id)
    train_size = int(len(unlabeled_tweet_id) * 0.8)
    return unlabeled_tweet_id[:train_size], unlabeled_tweet_id[train_size:]

def create_data_idx(labeled_corpuses, fold=5):
    """
    Creating a dictionary that assigns a tweet to a fold/set
    e.g. training set in the ith cross validation, ensemble, test, etc

    Parameters
    ----------
    labeled_corpuses: directories of tr, val, held_out_test, ensemble
    """
    # unpack the array
    tr, val, held_out_test, ensemble = labeled_corpuses
    #the index dictionary
    idx_dictionary = {}

    # read the ensemble and heldout test dataframe, which is the same
    df_ensemble = pd.read_json(ensemble)
    df_heldout_test = pd.read_json(held_out_test)
    idx_dictionary['ensemble_ind'] = set(df_ensemble['tweet_id'].values.tolist())
    idx_dictionary['heldout_test_ind'] = set(df_heldout_test['tweet_id'].values.tolist())

    create_cv_idx(tr, val, idx_dictionary, fold)
    return idx_dictionary

def user_time_indexing(data_dictionary):
    indexing = {}
    for tweet_id in data_dictionary:
        tweet_property = data_dictionary[tweet_id]
        user_post = tweet_property['user_post']
        if user_post == 1:
            continue
        else:
            if indexing.get(user_post) is None:
                indexing[user_post] = []
            indexing[user_post].append(tweet_id)
    for user_post in indexing:
        indexing[user_post] = sorted(indexing[user_post], key=lambda tid: data_dictionary[tid]['created_at'])
    return indexing

def create_dataset(labeled_corpuses, unlabeled_corpuses, verbose=False):
    data_dictionary = retrieve_content(labeled_corpuses, unlabeled_corpuses, verbose)

    if verbose:
        print('processing each tweet ...')
    for key in data_dictionary:
        process_data_entries(data_dictionary[key])

    if verbose:
        print('creating indexes for train test splitting')
    idx_dictionary = create_data_idx(labeled_corpuses)

    if verbose:
        print('checking correctness of the index dictionary')
        print_meta_info(idx_dictionary)
    assert_idx_correctness(idx_dictionary)

    if verbose:
        print('creating user-time indexing')
    user_time_ind = user_time_indexing(data_dictionary)

    unlabeled_tr, unlabeled_val = create_unlabeled_ind(data_dictionary)

    data = {'data': data_dictionary, 'classification_ind': idx_dictionary, 'user_time_ind': user_time_ind,
            'unlabeled_tr': unlabeled_tr, 'unlabeled_val': unlabeled_val}

    if verbose:
        print('dumping data')
    pkl.dump(data, open('../data/data.pkl', 'wb'))

    del data['user_time_ind']
    for tid in data['unlabeled_tr'] + data['unlabeled_val']:
        del data['data'][tid]
    del data['unlabeled_tr']
    del data['unlabeled_val']

    pkl.dump(data, open('../data/labeled_data.pkl', 'wb'))
