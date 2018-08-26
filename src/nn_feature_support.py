"""
===================
nn_feature_support.py
===================
Authors: Serina Chang
Date: 05/14/2018
Preparing features for neural net.
"""

import numpy as np
import pickle
from gensim.models import KeyedVectors
from data_loader import Data_loader
from represent_tweet_level import TweetLevel
from represent_context import Contextifier
from model_def import input_name_is_user_idx
from sklearn.preprocessing import StandardScaler

class ContextWrapper():
    def __init__(self, emb_type, num_days, dl, tweet_dict = None, user_ct_tweets = None, id_to_location = None):
        if emb_type == 'w2v':
            tl = TweetLevel(emb_file='../data/w2v_word_s300_w5_mc5_ep20.bin', tweet_dict=tweet_dict)
            context_hl_ratio = .5
            context_combine = 'avg'
            tl_combine = 'avg'
        else:
            tl = TweetLevel(emb_file='../data/splex_minmax_svd_word_s300_seeds_hc.pkl', tweet_dict=tweet_dict)
            context_hl_ratio = 1
            context_combine = 'sum'
            tl_combine = 'sum'
        context_size = num_days
        post_types = [Contextifier.SELF]

        self.cl = Contextifier(tl, post_types, context_size, context_hl_ratio, context_combine, tl_combine)
        if user_ct_tweets is None or id_to_location is None:
            user_ct_tweets, id_to_location = self.cl.assemble_context(dl.all_data())
        self.cl.set_context(user_ct_tweets, id_to_location)

        self.type = emb_type

    def get_representation(self, tid, modes):
        reps = []
        for mode in modes:
            self.cl.set_context_combine(mode)
            reps.append(self.cl.get_context_embedding(tid)[0])
        return np.concatenate(reps)

    def get_params(self):
        return self.cl.tweet_level.tweet_dict, self.cl.user_ct_tweets, self.cl.id_to_location

def init_tl(emb_type):
    if emb_type == 'w2v':
        tl = TweetLevel(emb_file='../data/w2v_word_s300_w5_mc5_ep20.bin')
    else:
        tl = TweetLevel(emb_file='../data/splex_standard_svd_word_s300_seeds_hc.pkl')
    return tl

def init_context(emb_type, num_days, dl, tweet_dict, user_ct_tweets, id_to_location):
    cl = ContextWrapper(emb_type, num_days, dl, tweet_dict, user_ct_tweets, id_to_location)
    return cl


'''Preparing embeddings'''
def make_word_embeds(include_w2v = True, include_splex = False, w2v_dim = 200):
    save_file, dim, w2v, splex = 'word_emb', 0, None, None
    if include_w2v:
        assert(w2v_dim == 300 or w2v_dim == 200)
        w2v = KeyedVectors.load_word2vec_format('../data/w2v_word_s' + str(w2v_dim) + '_w5_mc5_ep20.bin', binary=True)
        dim += w2v_dim
        save_file += '_w2v_' + str(w2v_dim)
    if include_splex:
        splex_w_sub = pickle.load(open('../data/splex_standard_svd_word_s300_seeds_hc.pkl', 'rb'))
        splex = dict((idx, splex_w_sub[idx][:2]) for idx in splex_w_sub)  # exclude substance scores
        dim += 2
        save_file += '_splex'
    save_file += '.np'

    vocab_size = 40000
    embeds = np.zeros((vocab_size, dim), dtype=np.float)
    for idx in range(1, vocab_size):
        str_idx = str(idx)
        if include_w2v:
            if str_idx in w2v.vocab:
                embeds[idx][:w2v_dim] = w2v[str_idx]
            else:
                embeds[idx][:w2v_dim] = w2v['1']
        if include_splex:
            if str_idx in splex:
                embeds[idx][-2:] = splex[str_idx]  # last two dims
            else:
                embeds[idx][-2:] = splex['1']

    np.savetxt(save_file, embeds)
    print('Saved embeddings in', save_file)

def check_embeds(fname):
    embeds = np.loadtxt(fname)
    print('Shape:', embeds.shape)

def make_user_embeds(num_users):
    dim = 300
    embeds = np.random.rand(num_users, dim)

    print('Initializing Data Loader...')
    dl = Data_loader()
    tl = init_tl('w2v')
    test_ids = [tweet['tweet_id'] for tweet in dl.test_data()]
    pretrained_count = 0
    for user_idx in range(2, num_users):  # reserve 0 for padding (i.e. no user), 1 for unknown user
        tweet_dicts = dl.tweets_by_user(user_idx)  # all tweets WRITTEN by this user
        if tweet_dicts is not None and len(tweet_dicts) > 0:
            tweet_count = 0
            all_tweets_sum = np.zeros(dim, dtype=np.float)
            for tweet_dict in tweet_dicts:
                tid = tweet_dict['tweet_id']
                if tid not in test_ids:
                    tweet_count += 1
                    tweet_avg = tl.get_representation(tid, mode='avg')
                    all_tweets_sum += tweet_avg
            if tweet_count > 0:
                pretrained_count += 1
                all_tweets_avg = all_tweets_sum / tweet_count
                embeds[user_idx] = all_tweets_avg
    print('Found tweets for {} out of {} users'.format(pretrained_count, num_users-2))

    embeds = StandardScaler().fit_transform(embeds)  # mean 0, variance 1
    embeds[0] = np.zeros(dim)  # make sure padding is all 0's

    save_file = str(num_users) + '_user_emb.np'
    np.savetxt(save_file, embeds)
    print('Saved embeddings in', save_file)


'''Preparing inputs'''
# splex_tl: splex tweet-level, summing word-level splex scores
# w2v_cl: w2v context-level, averaging word-level and tweet-level context scores, 60 days, .5 hl ratio
# splex_cl: splex context-level, summing word-level and tweet-level splex scores, 2 days, 1 hl ratio
# post_user_index: index of user who is posting, 1 if unknown user
# mention_user_index: index of the first user mentioned or 0 if no mentions
# retweet_user_index: index of the user being retweeted or 0 if no user retweet
# time: time features
def add_inputs():
    save_file = 'all_inputs.pkl'
    all_inputs = pickle.load(open(save_file, 'rb'))

    print('Initializing labeled Data Loader...')
    labeled_dl = Data_loader(labeled_only=True)
    labeled_tweets = labeled_dl.all_data()

    # TIME INPUT
    # add_time_input(all_inputs)
    # print('Added time input, shape =', np.array(list(all_inputs['time'].values())).shape)

    # TWEET-LEVEL INPUTS
    # add_tweet_level_input(all_inputs, labeled_tweets, emb_type='splex')
    # print('Added splex_tl input, shape =', np.array(list(all_inputs['splex_tl'].values())).shape)

    # CONTEXT-LEVEL INPUTS
    # emb_to_sizes = {'w2v': [30, 60], 'splex': [2, 30]}
    # add_context_level_inputs(all_inputs, labeled_tweets, emb_to_sizes=emb_to_sizes)
    # print('Added context inputs')
    # print('w2v shape =', np.array(list(all_inputs['30_w2v_cl'].values())).shape)
    # print('splex shape =', np.array(list(all_inputs['2_splex_cl'].values())).shape)

    # USER INPUTS
    # add_user_inputs(all_inputs, labeled_tweets, num_users=300)
    # add_user_inputs(all_inputs, labeled_tweets, num_users=50)
    # print('Added user inputs: 50 users shape =', np.array(list(all_inputs['50_post_user_index'].values())).shape)

    # # PAIRWISE INPUT
    # add_pairwise_input(all_inputs, labeled_tweets, cutoff=1)
    # add_pairwise_input(all_inputs, labeled_tweets, cutoff=2)
    # add_pairwise_input(all_inputs, labeled_tweets, cutoff=3)
    # print('Added pairwise inputs')
    # print('splex shape =', np.array(list(all_inputs['pairwise_c1_splex'].values())).shape)
    # print('w2v shape =', np.array(list(all_inputs['pairwise_c1_w2v'].values())).shape)

    pickle.dump(all_inputs, open(save_file, 'wb'))
    print('Saved', save_file)


def add_time_input(all_inputs):
    id2time = pickle.load(open('../data/id2time_feat.pkl', 'rb'))
    all_inputs['time'] = id2time

def add_tweet_level_input(all_inputs, labeled_tweets, emb_type):
    print('Pre-popping tl:', len(all_inputs))
    no_tl_inputs = {}
    for input_name, np in all_inputs.items():
        if not input_name.endswith('tl'):
            no_tl_inputs[input_name] = np
    all_inputs = no_tl_inputs
    print('Post-popping tl:', len(all_inputs))

    sorted_tids = sorted([tweet['tweet_id'] for tweet in labeled_tweets])
    tl = init_tl(emb_type)
    sorted_reps = [tl.get_representation(tid, mode='sum') for tid in sorted_tids]
    sorted_reps = StandardScaler().fit_transform(sorted_reps)
    all_inputs[emb_type + '_tl'] = dict((sorted_tids[i], sorted_reps[i]) for i in range(len(sorted_tids)))
    print('Post-adding tl:', len(all_inputs))

def add_context_level_inputs(all_inputs, labeled_tweets, emb_to_sizes):
    print('Pre-popping cl:', len(all_inputs))
    no_cl_inputs = {}
    for input_name, np in all_inputs.items():
        if not input_name.endswith('cl'):
            no_cl_inputs[input_name] = np
    all_inputs = no_cl_inputs
    print('Post-popping cl:', len(all_inputs))

    sorted_tids = sorted([tweet['tweet_id'] for tweet in labeled_tweets])
    print('Initializing complete Data Loader...')
    complete_dl = Data_loader()
    tweet_dict, user_ct_tweets, id_to_location = None, None, None
    for emb_type in emb_to_sizes:
        for size in emb_to_sizes[emb_type]:
            cl = init_context(emb_type, size, complete_dl, tweet_dict=tweet_dict,
                              user_ct_tweets=user_ct_tweets, id_to_location=id_to_location)
            combine_modes = ['avg'] if emb_type == 'w2v' else ['sum']
            sorted_reps = [cl.get_representation(tid, modes=combine_modes) for tid in sorted_tids]
            sorted_reps = StandardScaler().fit_transform(sorted_reps)
            all_inputs['{}_{}_cl'.format(emb_type, str(size))] = dict((sorted_tids[i], sorted_reps[i]) for i in range(len(sorted_tids)))
            if tweet_dict is None:
                tweet_dict, user_ct_tweets, id_to_location = cl.get_params()
    print('Post-adding cl:', len(all_inputs))

def add_user_inputs(all_inputs, tweets, num_users):
    print('Adding user inputs: {} tweets, {} users'.format(len(tweets), num_users))
    tid2post = {}
    tid2retweet = {}
    tid2mention = {}
    for tweet in tweets:
        tid = tweet['tweet_id']
        tid2post[tid] = tweet['user_post']
        tid2mention[tid] = tweet['user_mentions'][0] if 'user_mentions' in tweet and len(tweet['user_mentions']) > 0 else None
        tid2retweet[tid] = tweet['user_retweet'] if 'user_retweet' in tweet else None

    all_inputs[str(num_users) + '_post_user_index'] = tid2post
    all_inputs[str(num_users) + '_mention_user_index'] = tid2mention
    all_inputs[str(num_users) + '_retweet_user_index'] = tid2retweet
    edit_user_inputs(all_inputs, num_users)

# change None's to 0
# change user id's >= user_nums to 1
# change all user id's to nd arrays of length 1
def edit_user_inputs(inputs, num_users):
    for input_name in inputs:
        if input_name_is_user_idx(input_name) and input_name.startswith(str(num_users)):
            for id, user_idx in inputs[input_name].items():
                if user_idx is None:
                    user_idx = 0   # if there is no retweet or mention, user index is 0
                assert 'int' in str(type(user_idx))
                if user_idx >= num_users:
                    user_idx = 1  # if user index is not under num_users, user index is 1
                inputs[input_name][id] = np.array([user_idx])

def add_pairwise_input(all_inputs, tweets, cutoff):
    pair2w2v = pickle.load(open('pairwise_c{}_w2v.pkl'.format(str(cutoff)), 'rb'))
    w2v_dim = 300

    pair2splex = pickle.load(open('pairwise_c{}_splex.pkl'.format(str(cutoff)), 'rb'))
    splex_dim = 2

    tid2w2v = {}
    tid2splex = {}
    no_pair = no_embedding = 0
    for tweet in tweets:
        pw_w2v = None
        pw_splex = None
        u = int(tweet['user_post'])
        if 'user_retweet' not in tweet and ('user_mentions' not in tweet or len(tweet['user_mentions'])==0):
            no_pair += 1
            pw_w2v = np.zeros(w2v_dim)
            pw_splex = np.zeros(splex_dim)
        if pw_w2v is None and 'user_retweet' in tweet:
            ur = int(tweet['user_retweet'])
            pair = sorted([u, ur])
            pair_id = str(pair[0]) + '_' + str(pair[1])
            if pair_id in pair2w2v:
                pw_w2v = pair2w2v[pair_id]
                pw_splex = pair2splex[pair_id]
        if pw_w2v is None and 'user_mentions' in tweet:
            for um in tweet['user_mentions']:
                um = int(um)
                pair = sorted([u, um])
                pair_id = str(pair[0]) + '_' + str(pair[1])
                if pair_id in pair2w2v:
                    pw_w2v = pair2w2v[pair_id]
                    pw_splex = pair2splex[pair_id][:splex_dim]
                    break
        if pw_w2v is None:
            no_embedding += 1
            pw_w2v = pair2w2v['neutral']
            pw_splex = pair2splex['neutral']
        tid2w2v[tweet['tweet_id']] = pw_w2v
        tid2splex[tweet['tweet_id']] = pw_splex

    print('no pair:', no_pair)
    print('no embedding:', no_embedding)
    print('total tweets:', len(tweets))
    all_inputs['pairwise_c{}_splex'.format(str(cutoff))] = tid2splex
    all_inputs['pairwise_c{}_w2v'.format(str(cutoff))] = tid2w2v

    return all_inputs

# take unlabeled tids out of all_inputs.pkl to save space
def edit_inputs_pkl():
    save_file = 'all_inputs.pkl'
    inputs = pickle.load(open(save_file, 'rb'))
    labeled_tids = np.loadtxt('../data/labeled_tids.np', dtype='int')
    labeled_tids = set(labeled_tids.flatten())
    print('Size of tid set:', len(labeled_tids))
    for input_name in inputs:
        print(input_name)
        print('original size of id2np:', len(inputs[input_name]))
        only_labeled = {}
        for id,val in inputs[input_name].items():
            if id in labeled_tids:
                only_labeled[id] = val
        inputs[input_name] = only_labeled
        print('new size of id2np:', len(inputs[input_name]))
    pickle.dump(inputs, open(save_file, 'wb'))
    print('Edited inputs, saved in', save_file)

def check_pairwise():
    save_file = 'all_inputs.pkl'
    inputs = pickle.load(open(save_file, 'rb'))
    pw_splex = inputs['pairwise_splex']
    no_pair = 0
    for tweet in pw_splex:
        if np.array_equal(pw_splex[tweet], np.zeros(2)):
            no_pair += 1
    print('{} no pair out of {}'.format(no_pair, len(pw_splex)))

if __name__ == '__main__':
    # make_word_embeds(include_w2v=True, include_splex=False, w2v_dim=200)
    make_user_embeds(num_users=300)
    # check_embeds('300_user_emb.np')

    # add_inputs()

    # edit_inputs_pkl()
    # check_pairwise()
    # save_file = 'all_inputs.pkl'
    # inputs = pickle.load(open(save_file, 'rb'))
    # for input_name in inputs:
    #     if 'splex' in input_name:
    #         print(input_name)
    #         print(list(inputs[input_name].values())[:10])