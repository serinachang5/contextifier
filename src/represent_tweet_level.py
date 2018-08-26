"""
===================
represent_tweet_level.py
===================
Authors: Serina Chang
Date: 5/01/2018
Generate and visualize tweet-level embeddings, write them to file.
"""

from data_loader import Data_loader
from gensim.models import KeyedVectors, Doc2Vec
import numpy as np
import pickle
import statsmodels.api as sm
from scipy.stats.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer

class TweetLevel:
    '''
    Returns representations of tweets as fixed-length vectors. Takes an
    SPLex, Word2Vec, or Doc2Vec embedding file to initialize. If SPLex or Word2Vec,
    TweetLevel combines the word-level embeddings when get_representation is
    called. If Doc2Vec, a vector is inferred from the Doc2Vec model.
    ---------------------
    Parameters:
        emb_file: the name of the embedding file. Must be an SPLex pkl file,
            Word2Vec KeyedVectors, or Doc2Vec model.
        w2v_weight_by_idf: whether to weight the embeddings by their inverse
            document frequency. Only applied to Word2Vec embeddings.
        splex_include_sub: whether to include substance scores of SPLex embeddings.
        tweet_dict: a dictionary of tweet id to index sequences. If this is
            None, tweet_dict is built upon initialization.
    '''
    def __init__(self, emb_file, w2v_weight_by_idf = False, splex_include_sub = False, tweet_dict = None, verbose = False):
        print('Initializing TweetLevel...')
        self.verbose = verbose

        # store embeddings
        if 'splex' in emb_file:
            self.emb_type = 'splex'
            splex_w_sub = pickle.load(open(emb_file, 'rb'))
            if splex_include_sub:
                self.idx2emb = splex_w_sub
            else:
                self.idx2emb = dict((idx, splex_w_sub[idx][:2]) for idx in splex_w_sub)
            self.neut = self.idx2emb['1']
            if self.verbose: print('Number of word vectors in {}: {}'.format(emb_file, len(self.idx2emb)))

        elif 'w2v' in emb_file:
            self.emb_type = 'w2v'
            wv = KeyedVectors.load_word2vec_format(emb_file, binary=True)
            self.idx2emb = dict((idx, wv[idx]) for idx in wv.vocab)
            self.neut = self.idx2emb['1']  # pre-weighted
            if self.verbose: print('Number of word vectors in {}: {}'.format(emb_file, len(self.idx2emb)))

        elif 'd2v' in emb_file:
            self.emb_type = 'd2v'
            self.d2v = Doc2Vec.load(emb_file)
            self.neut = None
            if self.verbose: print('Number of doc vectors in {}: {}'.format(emb_file, len(self.d2v.docvecs)))

        else:
            raise ValueError('Cannot init TweetLevel with', emb_file)

        # dictionary of tweet_id to word_int_arr
        if tweet_dict is None:
            complete_dict = pickle.load(open('../data/data.pkl', 'rb'))['data']
            tweet_dict = dict((tweet_id, complete_dict[tweet_id]['word_int_arr']) for tweet_id in complete_dict)
            if self.verbose: print('Sample tweet_dict item:', next(iter(tweet_dict.items())))
        self.tweet_dict = tweet_dict

        if self.emb_type == 'w2v' and w2v_weight_by_idf:
            self._weight_embs_by_idf()

        print('Built TweetLevel. Type {}, storing {} tweets'.format(self.emb_type, len(self.tweet_dict)))

    # if Doc2Vec, infer vector; else, combine word-level embeddings
    def get_representation(self, tweet_id, mode = None):
        if type(tweet_id) is str:
            tweet_id = int(tweet_id)
        assert(tweet_id in self.tweet_dict)

        seq = self.tweet_dict[tweet_id]
        seq = [str(idx) for idx in seq]

        if self.emb_type == 'd2v':
            return self._get_docvec(seq)

        # get word-level embeddings
        if len(seq) == 0:
            return self.get_neutral_word_level()
        found_embeddings = []
        for idx in seq:
            if idx in self.idx2emb:
                found_embeddings.append(self.idx2emb[idx])
        if len(found_embeddings) == 0:
            return self.get_neutral_word_level()

        # combine word-level embeddings
        if mode is None:
            # defaults
            mode = 'avg' if self.emb_type == 'w2v' else 'sum'

        if mode == 'avg':
            return self._get_average(found_embeddings)
        elif mode == 'sum':
            return self._get_sum(found_embeddings)
        elif mode == 'max':
            return self._get_max(found_embeddings)
        elif mode == 'min':
            return self._get_min(found_embeddings)
        else:
            raise ValueError('Invalid word-level mode:', mode)

    # yield tweet-level reps for all tweets in data.pkl
    def get_all_representations(self, mode = 'avg'):
        for tweet_id in self.tweet_dict:
            yield tweet_id, self.get_representation(tweet_id, mode=mode)

    # get dimension of tweet-level representation
    def get_dimension(self):
        if self.emb_type == 'd2v':
            sample_vec = self._get_docvec(['1'])
        else:
            sample_vec = self.get_neutral_word_level()
        return sample_vec.shape[0]

    # get representation of neutral word - only exists for Word2Vec or SPLex
    def get_neutral_word_level(self):
        assert(self.neut is not None)
        return self.neut

    '''HELPER FUNCTIONS'''
    # weight word-level embeddings by their idf, only called if w2v
    def _weight_embs_by_idf(self):
        sentences = []
        for seq in self.tweet_dict.values():  # idx sequences
            sentences.append(' '.join([str(x) for x in seq]))

        tfidf_model = TfidfVectorizer(token_pattern='\d+')  # get all digits
        tfidf_model.fit(sentences)
        vocab = tfidf_model.vocabulary_  # dict of corpus_idx to tfidf_idx (idx in tfidf_model)
        idf_arr = tfidf_model.idf_
        idx2idf = {}
        for corpus_idx in vocab:
            tfidf_idx = vocab[corpus_idx]
            idf = idf_arr[tfidf_idx]
            idx2idf[corpus_idx] = idf

        # print('idfs for first 10 corpus indices:')
        # for i in range(1,11):
        #     print(i, idx2idf[str(i)])
        # print('idfs for last 10 corpus indices:')
        # for i in range(len(idx2idf)-10, len(idx2idf)):
        #     print(i, idx2idf[str(i)])

        for idx in self.idx2emb:
            self.idx2emb[idx] = self.idx2emb[idx] * idx2idf[idx]

        print('Weighted w2v embeddings by inverse document frequency.')

    # inferred vector from doc2vec model, given list of indices in doc
    def _get_docvec(self, seq):
        return self.d2v.infer_vector(seq)

    # average of embeddings in list
    def _get_average(self, elist):
        return np.mean(elist, axis=0)

    # sum of embeddings in list
    def _get_sum(self, elist):
        return np.sum(elist, axis=0)

    # max per dimension of embeddings in list
    def _get_max(self, elist):
        embs = np.array(elist)
        embs_by_dim = embs.T  # one dim per row
        max_per_dim = np.max(embs_by_dim, axis=1)
        return max_per_dim

    # min per dimension of embeddings in list
    def _get_min(self, elist):
        embs = np.array(elist)
        embs_by_dim = embs.T  # one dim per row
        min_per_dim = np.min(embs_by_dim, axis=1)
        return min_per_dim

# test TL functionalities
def test_TL(emb_type):
    assert(emb_type == 'w2v' or emb_type == 'splex' or emb_type == 'd2v')
    if emb_type == 'w2v':
        tl = TweetLevel(emb_file='../data/w2v_word_s300_w5_mc5_ep20.bin')
    elif emb_type == 'splex':
        tl = TweetLevel(emb_file='../data/splex_minmax_svd_word_s300_seeds_hc.pkl')
    else:
        tl = TweetLevel(emb_file='../data/d2v_word_s300_w5_mc5_ep20.mdl')

    print(tl.get_dimension())
    tweet_dict = tl.tweet_dict
    sample_id = list(tweet_dict.keys())[0]
    print(sample_id)
    print(tl.get_representation(sample_id))

    tl2 = TweetLevel(emb_file='../data/w2v_word_s300_w5_mc5_ep20.bin', tweet_dict=tweet_dict)
    for mode in ['avg', 'sum', 'min', 'max']:
        print(mode, sum(tl2.get_representation(sample_id, mode=mode)))

# write tweet-level representations to file
def write_reps_to_file(emb_type, rep_modes = None):
    assert(emb_type == 'w2v' or emb_type == 'splex' or emb_type == 'd2v')
    if emb_type == 'w2v':
        tl = TweetLevel(emb_file='../data/w2v_word_s300_w5_mc5_ep20.bin')
    elif emb_type == 'splex':
        tl = TweetLevel(emb_file='../data/splex_minmax_svd_word_s300_seeds_hc.pkl')
    else:
        tl = TweetLevel(emb_file='../data/d2v_word_s300_w5_mc5_ep20.mdl')

    if emb_type == 'w2v' or emb_type == 'splex':
        assert(rep_modes is not None)
        for rm in rep_modes:
            fname = '../reps/' + emb_type + '_' + rm + '.txt'
            print('\nWriting embeddings to', fname)
            with open(fname, 'w') as f:
                count = 0
                for id,rep in tl.get_all_representations(mode=rm):
                    if count % 50000 == 0: print(count)
                    f.write(str(id) + '\t')
                    rep = [str(x) for x in rep]
                    f.write(','.join(rep) + '\n')
                    count += 1
            print('Done. Wrote {} embeddings'.format(count))
    else:
        fname = '../reps/d2v.txt'
        print('\nWriting embeddings to', fname)
        with open(fname, 'w') as f:
            count = 0
            for id,rep in tl.get_all_representations():  # no mode to specify
                if count % 50000 == 0: print(count)
                f.write(str(id) + '\t')
                rep = [str(x) for x in rep]
                f.write(','.join(rep) + '\n')
                count += 1
        print('Done. Wrote {} embeddings'.format(count))

# check first 100 written embeddings
def check_written_embeddings(emb_type, rep_mode = None):
    assert(emb_type == 'w2v' or emb_type == 'splex' or emb_type == 'd2v')
    if emb_type == 'w2v':
        tl = TweetLevel(emb_file='../data/w2v_word_s300_w5_mc5_ep20.bin')
    elif emb_type == 'splex':
        tl = TweetLevel(emb_file='../data/splex_minmax_svd_word_s300_seeds_hc.pkl')
    else:
        tl = TweetLevel(emb_file='../data/d2v_word_s300_w5_mc5_ep20.mdl')

    if emb_type == 'w2v' or emb_type == 'splex':
        assert(rep_mode is not None)
        fname = '../reps/' + emb_type + '_' + rep_mode + '.txt'
    else:
        rep_mode = 'd2v'
        fname = '../reps/d2v.txt'

    print('Checking', fname)
    with open(fname, 'r') as f:
        count = 0
        for line in f:
            id, written_emb = line.split('\t')
            written_emb = [float(x) for x in written_emb.split(',')]
            real_emb = tl.get_representation(id, mode=rep_mode)
            assert(np.allclose(written_emb, real_emb))
            count += 1
            if count == 100:
                return

# check correlation between SPLex scores at tweet level
def check_corr():
    tl = TweetLevel(emb_file='../data/splex_balanced_minmax_svd_word_s300_seeds_hc.pkl')
    reps = np.array([rep for id, rep in tl.get_all_representations(mode='sum')])
    all_loss = reps.T[0]
    all_agg = reps.T[1]
    all_sub = reps.T[2]

    print('Pearson for loss-agg:', pearsonr(all_loss, all_agg))
    ols = sm.OLS(all_loss, all_agg).fit()
    print(ols.summary())

    print('Pearson for loss-sub:', pearsonr(all_loss, all_sub))
    ols = sm.OLS(all_loss, all_sub).fit()
    print(ols.summary())

    print('Pearson from agg-sub:', pearsonr(all_agg, all_sub))
    ols = sm.OLS(all_agg, all_sub).fit()
    print(ols.summary())


if __name__ == '__main__':
    # modes = ['max', 'min']
    # write_reps_to_file(emb_type='w2v', rep_modes=modes)
    # write_reps_to_file(emb_type='splex', rep_modes=modes)
    # check_written_embeddings(emb_type='splex', rep_mode='avg')

    test_TL('w2v')