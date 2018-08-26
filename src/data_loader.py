# coding=utf-8

"""
===================
data_loader
===================
Author: Ruiqi Zhong
Date: 04/21/2018
This module contains a class that will provide most functionalities needed for tokenizing and preprocessing
"""
import pickle as pkl
from collections import defaultdict
from sentence_tokenizer import int_array_rep
import numpy as np
from data_loader_utils import get_config, subsample, get_subsample_counts
from random import randint
from sklearn.model_selection import StratifiedShuffleSplit


class Data_loader:
    """
    A Data_loader class
    Read the corresponding processed files and
    return provides data needed for classification tasks
    each datapoint/tweet is a dictionary
    """

    def __init__(self, vocab_size = 40000, max_len = 50,
                 word_vocab_size = 40000, word_max_len = 50,
                 char_vocab_size = 1200, char_max_len = 150,
                 option = 'word', verbose = True, load_tweet = True,
                 labeled_only = False,
                 **kwargs):
        if verbose:
            print('Data loader ...')
        """
        Parameters
        ----------
        vocab_size: number of vocabularies to consider, including _PAD_ and _UNKNWON_
        max_len: the maximum length of a tweet
        option: the level of tokenization, "word" or "char"
        verbose: print progress while initializing
        """

        self.option, self.vocab_size, self.max_len = option, vocab_size, max_len
        assert(option in ['both', 'char', 'word'])
        if option in ['char', 'word']:
            self.word_vocab_size, self.word_max_len, self.char_vocab_size, self.char_max_len = [None] * 4
        else:
            self.word_vocab_size, self.word_max_len = word_vocab_size, word_max_len
            self.char_vocab_size, self.char_max_len = char_vocab_size, char_max_len
            self.vocab_size, self.max_len = None, None
            self.word2property = pkl.load(open('../model/word.pkl', 'rb'))
            self.char2property = pkl.load(open('../model/char.pkl', 'rb'))
            removed_words = []
            for word in self.word2property:
                if self.word2property[word]['id'] >= word_vocab_size:
                    removed_words.append(word)
            for word in removed_words:
                del self.word2property[word]
            removed_chars = []
            for char in self.char2property:
                if self.char2property[char]['id'] >= char_vocab_size:
                    removed_chars.append(char)
            for char in removed_chars:
                del self.char2property[char]

        if verbose:
            print('Loading vocabulary ...')
        if option != 'both':
            self.token2property = pkl.load(open('../model/' + option + '.pkl', 'rb'))  # loading the preprocessed token file
            removed_tokens = []
            for token in self.token2property:
                if self.token2property[token]['id'] >= vocab_size:
                    removed_tokens.append(token)
            for token in removed_tokens:
                del self.token2property[token]
            self.separator = ' ' if option == 'word' else ''  # chars are not seperated, words are by spaces
        if option == 'word':  # creating an id2wtoken dictionary
            self.id2token = dict([(self.token2property[word]['id'], word) for word in self.token2property])
            # self.token2id = {v.decode('utf8'):k for k, v in self.id2token.items()}
        elif option == 'char':
            self.id2token = dict([(self.token2property[c]['id'], chr(c) if bytes(c) < bytes(256) else c)
                                  for c in self.token2property])
            # self.token2id = {v.decode('utf8'):k for k, v in self.id2token.items()}

        if verbose and option != 'both':
            print('%d vocab is considered.' % min(len(self.id2token), self.vocab_size))
        else:
            print('%d word, max_len %d.' % (word_vocab_size, word_max_len))
            print('%d char, max_len %d.' % (char_vocab_size, char_max_len))

        '''
        # loading user information
        self.user2property = pkl.load(open('../model/user.pkl', 'rb'))
        self.id2user = dict([(self.user2property[user_name]['id'], user_name) for user_name in self.user2property])
        if verbose:
            print('Loading user information finished')
        '''
        if not load_tweet:
            return
        # loading tweet level data
        if verbose:
            print('Loading tweets ...')
        if not labeled_only:
            self.data = pkl.load(open('../data/data.pkl', 'rb'))
        else:
            self.data = pkl.load(open('../data/labeled_data.pkl', 'rb'))

        self.all_tid = set([tid for tid in self.data])
        # pad and trim the int representations of a tweet given the parameters of this Data_loader
        if verbose:
            print('Processing tweets ...')
        for tweet_id in self.data['data']:
            self.process_tweet_dictionary(self.data['data'][tweet_id])
        if verbose:
            print('Data loader initialization finishes')

    def cv_data(self, fold_idx):
        """
        Get the cross validation data. Each set is a list of dictionaries reprensenting a data point

        Parameters
        ----------
        fold_idx: the fold index of this 5-fold cross validation task

        Returns
        -------
        tr, val, test: train, val, test data for the current fold
        """
        cv_ind = self.data['classification_ind']['cross_val'][fold_idx]
        tr, val, test = (self.get_records_by_idxes(cv_ind['train_ind']),
                         self.get_records_by_idxes(cv_ind['val_ind']),
                         self.get_records_by_idxes(cv_ind['test_ind']))
        return tr, val, test

    def filter_by_length(self, tweets, min_len, padded, labeled):
        X = []
        y = []
        for tweet_dict in tweets:
            if len(tweet_dict['int_arr']) < min_len:
                continue
            if labeled:
                y.append(tweet_dict['label'])
            if padded:
                X.append(tweet_dict['padded_int_arr'])
            else:
                X.append(tweet_dict['int_arr'])
        return np.asarray(X), np.asarray(y)

    def cv_data_as_arrays(self, fold_idx, min_len, padded):
        tr, val, test = self.cv_data(fold_idx)
        X_tr, y_tr = self.filter_by_length(tr, min_len, padded, True)
        X_val, y_val = self.filter_by_length(val, min_len, padded, True)
        X_test, y_test = self.filter_by_length(test, min_len, padded, True)
        return X_tr, y_tr, X_val, y_val, X_test, y_test

    # retrieving data for ensemble
    # similar to cv_data function
    def ensemble_data(self):
        return self.get_records_by_idxes(self.data['classification_ind']['ensemble_ind'])

    # retrieving data for testing
    # similar to cv_data function
    def test_data(self):
        return self.get_records_by_idxes(self.data['classification_ind']['heldout_test_ind'])

    # return all the data for unsupervised learning
    def all_data(self):
        return self.get_records_by_idxes([tid for tid in self.data['data']])

    def unlabeled_tr_val(self):
        return [self.get_records_by_idxes(tids) for tids in [self.data['unlabeled_tr'],
                                                             self.data['unlabeled_val']]]

    def unlabeled_tr_val_as_arrays(self, min_len, padded):
        unld_tr, unld_val = self.unlabeled_tr_val()
        unld_tr_X, _ = self.filter_by_length(unld_tr, min_len, padded, False)
        unld_val_X, _ = self.filter_by_length(unld_val, min_len, padded, False)
        return unld_tr_X, unld_val_X

    def filter_by_emoji(self, x, emoji_ids):
        found_all = True
        for eid in emoji_ids:
            if eid not in x:
                found_all = False
        if not found_all:
            return None
        else:
            _x = [v for v in x if v not in emoji_ids]
            if len(_x) <= 0:
                return None
            _x = (_x + [0] * self.max_len)[:self.max_len]
            return _x

    def distant_supv_data(self, config_type = 'ACL', subsample_enabled = True, check_both = False):
        '''
        config_type = 'ACL' or 'EMNLP'
        It will generate distant supv dataset such that the label distribution is identical to that of labeled data.
        See method 'get_config' in data_loader_utils.py for more details.
        '''
        seed = 45345

        config = get_config(config_type = config_type)
        fh_e = u'\U0001f64f'.encode('utf8')
        fh_id = self.token2property[fh_e]['id']
        pf_e = u'\U0001f614'.encode('utf8')
        pf_id = self.token2property[pf_e]['id']
        top_two_loss = [fh_id, pf_id]
        g_e = u'\U0001f52b'.encode('utf8')
        g_id = self.token2property[g_e]['id']
        df_e = u'\U0001f608'.encode('utf8')
        df_id = self.token2property[df_e]['id']
        top_two_agg = [g_id, df_id]
        top_loss_ids = []
        for emoji in config['top_loss_emojis']:
            emoji_en = emoji.encode('utf8')
            if emoji_en in self.token2property:
                top_loss_ids.append(self.token2property[emoji_en]['id'])
        print ('len(top_loss_ids): ', len(top_loss_ids))
        top_agg_ids = []
        for emoji in config['top_agg_emojis']:
            emoji_en = emoji.encode('utf8')
            if emoji_en in self.token2property:
                top_agg_ids.append(self.token2property[emoji_en]['id'])
        print ('len(top_agg_ids): ', len(top_agg_ids))

        unld_tr, unld_val = self.unlabeled_tr_val()

        # mix unld_tr and unld_val
        unld_tr = unld_tr + unld_val

        print('Unlabeled data size: ', len(unld_tr))

        X_a = []
        X_l = []
        X_o = []
        for tweet_dict in unld_tr:
            x = tweet_dict['int_arr']
            # check for loss
            if check_both:
                _x = self.filter_by_emoji(x, top_two_loss)
            else:
                _x = self.filter_by_emoji(x, top_two_loss[:1])
            if _x is not None:
                X_l.append(_x)
                continue

            # check for aggression
            if check_both:
                _x = self.filter_by_emoji(x, top_two_agg)
            else:
                _x = self.filter_by_emoji(x, top_two_agg[:1])
            if _x is not None:
                X_a.append(_x)
                continue

            # consider for other
            get_in = True
            for eid in top_loss_ids:
                if eid in x:
                    get_in = False
                    break
            if not get_in:
                continue
            for eid in top_agg_ids:
                if eid in x:
                    get_in = False
                    break
            if get_in:
                # select 5% of other tweets to reduce the training time and avoid too much skew
                rv = randint(0, 99)
                if rv in [0, 33, 65, 78, 99]:
                    # text = re.sub(r'[^\x00-\x7F]+', '', text.encode('utf8'))
                    X_o.append(tweet_dict['padded_int_arr'])

        X_a = np.asarray(X_a)
        X_l = np.asarray(X_l)
        X_o = np.asarray(X_o)

        print ('Count(Aggression): ', X_a.shape)
        print ('Count(Loss): ', X_l.shape)
        print ('Count(Other): ', X_o.shape)

        if subsample_enabled:
            print ('Subsampling is enabled by default . . .')
            actual_counts = {'Loss': X_l.shape[0], 'Aggression': X_a.shape[0], 'Other': X_o.shape[0]}
            ss_counts = get_subsample_counts(actual_counts, config['desired_dist'])

            X_a = subsample(X_a, keep_num = ss_counts['Aggression'], seed = seed)
            X_l = subsample(X_l, keep_num = ss_counts['Loss'], seed = seed)
            X_o = subsample(X_o, keep_num = ss_counts['Other'], seed = seed)

            print ('After sampling:')
            print ('Count(Aggression): ', X_a.shape)
            print ('Count(Loss): ', X_l.shape)
            print ('Count(Other): ', X_o.shape)

        X = np.concatenate((X_a, X_l, X_o), axis = 0)
        y = np.asarray([0] * X_a.shape[0] + [1] * X_l.shape[0] + [2] * X_o.shape[0])

        sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = seed)
        for train_index, test_index in sss.split(X, y):
            X_train = X[train_index]
            y_train = y[train_index]
            X_test = X[test_index]
            y_test = y[test_index]

        return X_train, y_train, X_test, y_test

    # tokenize a string and convert it to int representation given the parameters of this data loader
    def convert2int_arr(self, s):
        assert(self.option in ['char', 'word'])
        int_arr = int_array_rep(str(s), option = self.option, vocab_count = self.vocab_size)
        # int_arr = self.pad_int_arr(int_arr)
        return int_arr

    def tweets_by_user(self, user_id):
        """
        Given a user id, return a list of tweets posted by that user, sorted by time

        Parameters
        ----------
        user_id: the user_id of interest

        Returns
        -------
        a list of tweet dictionaries, None if that user id does not have a list
        """
        tweet_ids = self.data['user_time_ind'].get(user_id)
        if tweet_ids is None:
            return None
        return self.get_records_by_idxes(self.data['user_time_ind'][user_id])

    # convert an int array to the unicode representation
    def convert2unicode(self, int_arr):
        return self.separator.join([self.id2token[id].decode() if type(self.id2token[id]) != str
                                    else self.id2token[id]
                                    for id in int_arr])

    def print_recovered_tweet(self, tweet_property):
        for key in tweet_property:
            print("%s: %s" % (key, tweet_property[key]))
        '''
        print('User %s posted the tweet.' % self.id2user[tweet_property['user_post']])
        print('Users being mentioned: ' + ', '.join([self.id2user[user_id] for user_id in tweet_property['user_mentions']]))
        if tweet_property.get('user_retweet') is not None:
            print('Retweet from %s.' % self.id2user[tweet_property['user_retweet']])
        '''
        print('original tweet content: ' + self.convert2unicode(tweet_property['int_arr']))

    # get the user name of an id
    def id2user_name(self, id):
        return self.id2user[id]

    # ========== Below are the helper functions of the class ==========

    def process_tweet_dictionary(self, record):
        if self.option != 'both':
            record['int_arr'] = record[self.option + '_int_arr']
            del record['word_int_arr']
            del record['char_int_arr']
            record['padded_int_arr'] = self.pad_int_arr(record['int_arr'])
            self.trim2vocab_size(record['int_arr'])
            self.trim2vocab_size(record['padded_int_arr'])
        else:
            record['word_padded_int_arr'] = self.pad_int_arr(record['word_int_arr'], self.word_max_len)[:]
            self.trim2vocab_size(record['word_padded_int_arr'], self.word_vocab_size)
            record['char_padded_int_arr'] = self.pad_int_arr(record['char_int_arr'], self.char_max_len)[:]
            self.trim2vocab_size(record['char_padded_int_arr'], self.char_vocab_size)

    def pad_int_arr(self, int_arr, max_len = None):
        int_arr = int_arr[:]
        if max_len is None:
            max_len = self.max_len
        int_arr += [0] * max_len
        return int_arr[:max_len]

    def trim2vocab_size(self, int_arr, vocab_size = None):
        if vocab_size is None:
            vocab_size = self.vocab_size
        for idx in range(len(int_arr)):
            if int_arr[idx] >= vocab_size:
                int_arr[idx] = 1

    def get_records_by_idxes(self, idxes):
        return [self.data['data'][idx] for idx in idxes]

    def get_label2idx(self):
        fold_idx = 0
        tr, _, _ = self.cv_data(fold_idx)
        labels = set()
        for tweet in tr:
            labels.add(tweet['label'])
        labels = list(labels)
        labels.sort()
        label2idx = {}
        for idx, label in enumerate(labels):
            label2idx[label] = idx
        return label2idx

    def get_class_weights(self):

        fold_idx = 0
        tr, _, _ = self.cv_data(fold_idx)

        label2count = defaultdict(int)
        for tweet in tr:
            label2count[tweet['label']] += 1

        class_weights = {}
        total = 0.0
        for cls in label2count.keys():
            total += (float(1) / label2count[cls])

        if total <= 0.0:
            return None

        K = float(1) / total

        for cls in label2count.keys():
            class_weights[cls] = (K / label2count[cls])

        return class_weights

    def get_freq_counts(self):
        counts = defaultdict(float)
        for token in self.token2property:
            counts[token.decode('utf8')] += self.token2property[token]['occurence_in_labeled'] + self.token2property[token]['occurence_in_unlabeled']
        return counts

    def __getitem__(self, tweet_id):
        return self.data['data'][tweet_id]


if __name__ == '__main__':
    debug_option = 'labeled_only'

    if debug_option == 'tokenizer_only':
        dl = Data_loader(vocab_size = 40000, max_len = 150, option = 'word', load_tweet = False)
        print(dl.convert2int_arr('fake niggas get extorted 💯'))

    elif debug_option == 'data_loader':
        dl = Data_loader(vocab_size = 40000, max_len = 150, option = 'word')
        fold_idx = 0
        tr, val, test = dl.cv_data(fold_idx)
        for idx in range(10):
            print('-------------')
            dl.print_recovered_tweet(tr[idx])

        user_tweets = dl.tweets_by_user(2)
        for idx in range(10):
            print('-------------')
            dl.print_recovered_tweet(user_tweets[idx])

        print(dl[423325236696580096])

    elif debug_option == 'labeled_only':
        dl = Data_loader(vocab_size = 40000, max_len = 150, option = 'both', labeled_only = True)
        fold = 5
        for fold_idx in range(fold):
            tr, val, test = dl.cv_data(fold_idx)
            print(len([d for d in val if d['label'] == 'Aggression']))

        ensemble_data = dl.ensemble_data()
        print(ensemble_data[0])

        test_data = dl.test_data()
        print(test_data[0])
