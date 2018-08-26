"""
===================
svm_pipeline.py
===================
Authors: Ethan Adams & Serina Chang
Date: 05/01/2018
Pipeline for cross-validating SVM.
"""

import argparse
from collections import Counter
from data_loader import Data_loader
from represent_context import Contextifier
from represent_tweet_level import TweetLevel
import numpy as np
import pickle
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC
from sentence_tokenizer import int_array_rep

print('Initializing Data Loader...')
dl = Data_loader(labeled_only=True)

def init_models():
    models = {}

    if args['include_unigrams']:
        models['unigrams'] = init_uni_model()

    if args['include_char_ngrams']:
        models['char_ngrams'] = init_char_model()

    # store so it doesn't need to be generated every time
    tweet_dict, user_ct_tweets, id_to_location = None, None, None

    # initialize TweetLevel models
    if args['include_w2v_tl']:
        models['w2v_tl'] = init_TL('w2v', tweet_dict)
        tweet_dict = models['w2v_tl'].tweet_dict
    if args['include_splex_tl']:
        models['splex_tl'] = init_TL('splex', tweet_dict)
        tweet_dict = models['splex_tl'].tweet_dict

    # initialize context level Contextifier models
    if args['include_w2v_cl']:
        models['w2v_cl'] = init_CL('w2v', tweet_dict, user_ct_tweets, id_to_location)
        tweet_dict = models['w2v_cl'].tweet_level.tweet_dict
        user_ct_tweets = models['w2v_cl'].user_ct_tweets
        id_to_location = models['w2v_cl'].id_to_location
    if args['include_splex_cl']:
        models['splex_cl'] = init_CL('splex', tweet_dict, user_ct_tweets, id_to_location)

    return models

# initialize unigrams model
def init_uni_model():
    vocab = [str(idx) for idx in range(1,args['unigram_size']+1)]  # get top <unigram_size> indices
    uni_model = CountVectorizer(vocabulary=vocab, token_pattern='\d+')
    return uni_model

# initialize charngram model
def init_char_model():
    min_n = args['char_min_gram']
    max_n = args['char_max_gram']
    assert(min_n <= max_n)
    char_model = CountVectorizer(token_pattern='\d+', ngram_range=(min_n, max_n), max_features=args['char_size'])
    return char_model

# initialize TweetLevel model
def init_TL(emb_type, tweet_dict):
    if emb_type == 'w2v':
        if args['use_d2v']:
            tl = TweetLevel(emb_file='../data/d2v_word_s300_w5_mc5_ep20.mdl', tweet_dict=tweet_dict)
        else:
            tl = TweetLevel(emb_file='../data/w2v_word_s200_w5_mc5_ep20.bin', tweet_dict=tweet_dict)
    else:
        scaling = args['splex_scale']
        include_sub = args['include_sub_splex_tl']
        valid_scaling = ['minmax', 'standard']
        assert(scaling in valid_scaling)
        tl = TweetLevel(emb_file='../data/splex_' + scaling + '_svd_word_s300_seeds_hc.pkl',
                        splex_include_sub=include_sub, tweet_dict=tweet_dict)
    return tl

def parse_post_types(emb_type):
    post_types = [Contextifier.SELF]
    if args[emb_type + '_use_rt']:
        post_types.append(Contextifier.RETWEET)
    if args[emb_type + '_use_mentions']:
        post_types.append(Contextifier.MENTION)
    if args[emb_type + '_use_rt_mentions']:
        post_types.append(Contextifier.RETWEET_MENTION)
    return post_types

def init_CL(emb_type, tl_tweet_dict, user_ct_tweets, id_to_location):
    # intialize internal TweetLevel
    if emb_type == 'w2v':
        tl = TweetLevel(emb_file='../data/w2v_word_s200_w5_mc5_ep20.bin', tweet_dict=tl_tweet_dict)
        tl_combine = 'avg'
    else:
        # default - will not include substance scores
        tl = TweetLevel(emb_file='../data/splex_minmax_svd_word_s300_seeds_hc.pkl', tweet_dict=tl_tweet_dict)
        tl_combine = 'sum'

    # set params
    context_combine = args[emb_type + '_cl_mode']
    context_size = args[emb_type + '_size']
    context_hl_ratio = args[emb_type + '_hl']
    post_types = parse_post_types(emb_type)

    # initialize
    cl = Contextifier(tl, post_types, context_size, context_hl_ratio, context_combine, tl_combine)

    # configure
    if user_ct_tweets is None or id_to_location is None:
        user_ct_tweets, id_to_location = cl.assemble_context(dl.all_data())
    cl.set_context(user_ct_tweets, id_to_location)

    return cl

def update_models(models):
    for emb_type in ['w2v', 'splex']:
        cl_name = emb_type + '_cl'
        if cl_name in models:
            models[cl_name].set_context_combine(args[emb_type + '_cl_mode'])
            models[cl_name].set_context_size(args[emb_type + '_size'])
            models[cl_name].set_context_hl_ratio(args[emb_type + '_hl'])
            models[cl_name].set_post_types(parse_post_types(emb_type))
    if 'char_ngrams' in models:
        models['char_ngrams'] = init_char_model()

def parse_reps_to_include():
    possible_reps = ['unigrams', 'char_ngrams', 'w2v_tl', 'splex_tl', 'w2v_cl', 'splex_cl']
    reps_to_include = []
    for rep_type in possible_reps:
        if args['include_' + rep_type]:
            reps_to_include.append(rep_type)
    return reps_to_include

def transform_data(data, models, tid2np, fitted):
    label_to_idx = {'Loss':0, 'Aggression':1, 'Other':2}
    y = np.array([label_to_idx[t['label']] for t in data])

    # rep_type mapped to representation matrix
    reps_to_include = parse_reps_to_include()
    reps = dict((rep_type, []) for rep_type in reps_to_include)
    for i in range(len(tid2np)):
        reps['add'+str(i+1)] = []  # additional features

    # build unigram representation matrix
    if 'unigrams' in reps:
        sentences = []  # tweets in index form but as strings
        for tweet in data:
            sentences.append(' '.join([str(x) for x in tweet['int_arr']]))
        # no need to check if fitted because vocab is set
        reps['unigrams'] = models['unigrams'].transform(sentences)
    if 'char_ngrams' in reps:
        sentences = []
        for tweet in data:
            unicode = dl.convert2unicode(tweet['int_arr'])
            char_arr = int_array_rep(unicode, option='char', debug=False)
            sentences.append(' '.join([str(x) for x in char_arr]))
        if fitted:  # already fitted - must be transforming test
            reps['char_ngrams'] = models['char_ngrams'].transform(sentences)
        else:  # hasn't been fitted yet - must be transforming train -> fit on train
            reps['char_ngrams'] = models['char_ngrams'].fit_transform(sentences)

    # build tweet and context representation matrices
    for tweet in data:
        tid = tweet['tweet_id']
        if 'w2v_tl' in reps:
            reps['w2v_tl'].append(models['w2v_tl'].get_representation(tid, args['w2v_tl_mode']))
        if 'splex_tl' in reps:
            splex_tl_rep = models['splex_tl'].get_representation(tid, args['splex_tl_mode'])
            reps['splex_tl'].append(splex_tl_rep)
        if 'w2v_cl' in reps:
            reps['w2v_cl'].append(models['w2v_cl'].get_context_embedding(tid, keep_stats=False)[0])  # ignore ct_tweets
        if 'splex_cl' in reps:
            reps['splex_cl'].append(models['splex_cl'].get_context_embedding(tid, keep_stats=False)[0])
        for i in range(len(tid2np)):
            reps['add'+str(i+1)].append(tid2np[i][tid])

    to_stack = []
    # standardize order (alphabetical): char_ngrams, splex_cl, splex_tl, unigrams, w2v_cl, w2v_tl,
    for rep_type, rep in sorted(reps.items(), key=lambda x: x[0]):
        if rep_type == 'unigrams' or rep_type == 'char_ngrams':
            to_stack.append(rep)
        else:
            to_stack.append(csr_matrix(np.array(rep)))
    X = hstack(to_stack)

    return X, y

def cross_validate(dl, models, add_feats):
    preds = []
    scores = []
    total_f1 = 0

    tid2np = []
    if len(add_feats) > 0:
        inputs = pickle.load(open('all_inputs.pkl', 'rb'))
        if 'pairwise' in add_feats:
            if add_feats['pairwise'] == 'both' or add_feats['pairwise'] == 'splex':
                tid2np.append(inputs['pairwise_splex'])
            if add_feats['pairwise'] == 'both' or add_feats['pairwise'] == 'w2v':
                tid2np.append(inputs['pairwise_w2v'])

    for fold_i in range(5):
        print('Fold:', fold_i)
        tr,val,tst = dl.cv_data(fold_i)

        # if tuning parameters, test on val; else, test on test
        if args['tuning']:
            tst = val

        print('Transforming training data...')
        X, y = transform_data(tr, models, tid2np, fitted=False)
        print('Training dimensions:', X.shape, y.shape)

        if args['weights'] == 'static':
            weights = {0: 0.35, 1: 0.5, 2: 0.15}
        else: # weights == dynamic
            weights = 'balanced'
        clf = SVC(kernel='linear', class_weight=weights)
        clf.fit(X, y)

        print('Transforming testing data...')
        X, y = transform_data(tst, models, tid2np, fitted=True)
        print('Testing dimensions:', X.shape, y.shape)

        pred = clf.predict(X)
        per_class = precision_recall_fscore_support(y, pred, average=None)
        macros = precision_recall_fscore_support(y, pred, average='macro')
        preds.extend(pred)
        scores.append(per_class)
        print('Loss F1: {}. Agg F1: {}. Other F1: {}. Macro F1: {}.'.format(round(per_class[2][0],5), round(per_class[2][1],5),
                                                                            round(per_class[2][2],5), round(macros[2],5)))
        total_f1 += macros[2]

    print('AVERAGE F1:', round(total_f1/5, 5))
    return preds, scores

def get_specs():
    specs = []
    if args['include_unigrams']:
        specs.append('UNI')
        specs.append(str(args['unigram_size']))

    if args['include_char_ngrams']:
        specs.append('CH')
        specs.append(str(args['char_size']))
        specs.append(str(args['char_min_gram']))
        specs.append(str(args['char_max_gram']))

    if args['include_w2v_tl']:
        if args['use_d2v']:
            specs.append('DT')
        else:
            specs.append('WT')

    if args['include_splex_tl']:
        specs.append('ST')
        specs.append(args['splex_scale'])
        if args['include_sub_splex_tl']:
            specs.append('wsub')

    if args['include_w2v_cl']:
        specs.append('WC')
        specs.append(str(args['w2v_size']))
        specs.append(str(args['w2v_hl']))
        if args['w2v_use_rt']:
            specs.append('rt')
        if args['w2v_use_mentions']:
            specs.append('mn')
        if args['w2v_use_rt_mentions']:
            specs.append('rtm')

    if args['include_splex_cl']:
        specs.append('SC')
        specs.append(args['splex_cl_mode'])
        specs.append(str(args['splex_size']))
        specs.append(str(args['splex_hl']))
        if args['splex_use_rt']:
            specs.append('rt')
        if args['splex_use_mentions']:
            specs.append('mn')
        if args['splex_use_rt_mentions']:
            specs.append('rtm')

    assert(len(specs) > 0)

    if 'pairwise' in args:
        specs.append('PW')
        specs.append(args['pairwise'])

    if args['weights'] == 'static':
        specs.append('STAT')
    else:
        specs.append('DYN')

    if args['tuning']:
        specs.append('TUN')
    else:
        specs.append('TST')

    return specs

def run_experiment(models = None):
    specs = get_specs()

    # initialize representation models
    if models is None:
        models = init_models()
    else:
        update_models(models)

    # check context level settings
    if args['include_w2v_cl']:
        w2v_cl = models['w2v_cl']
        print('W2V CONTEXT settings:')
        w2v_cl.print_settings()
    if args['include_splex_cl']:
        splex_cl = models['splex_cl']
        print('SPLEX CONTEXT settings:')
        splex_cl.print_settings()

    # get tid2np feats
    add_feats = {}
    if 'pairwise' in args:
        add_feats['pairwise'] = args['pairwise']

    # run cv experiment using these representations
    cv_preds, cv_scores = cross_validate(dl, models, add_feats)

    # save results
    out_file = '../cv_results/' + '_'.join(specs) + '.pkl'
    with open(out_file, 'wb') as f:
        pickle.dump((args, cv_preds, cv_scores), f)
    print('Args and cross-val scores saved to', out_file)

# print precision, recall, and F1 per class in each fold if verbose
# print macro F1 in each fold
# print final average of macro F1's
def print_scores(per_class, verbose=True):
    avg_loss_f = avg_agg_f = avg_oth_f = avg_mac_f = 0.0
    for fold_i, fold in enumerate(per_class):
        print('Fold:', fold_i)
        if verbose:
            for metric, metric_results in zip(['Precision', 'Recall', 'F1'], fold):
                l, a, o = metric_results
                if metric == 'F1':
                    avg_loss_f += l
                    avg_agg_f += a
                    avg_oth_f += o
                print(metric, '- Loss: {}. Agg: {}. Other: {}.'.format(round(l, 5), round(a, 5), round(o, 5)))
        print('Macro F1:', round(np.mean(fold[2]), 5))
        avg_mac_f += np.mean(fold[2])
        print()
    print('AVG LOSS F1:', round(avg_loss_f/len(per_class), 5))
    print('AVG AGG F1:', round(avg_agg_f/len(per_class), 5))
    print('AVG OTH F1:', round(avg_oth_f/len(per_class), 5))
    print('AVG MACRO F1:', round(avg_mac_f/len(per_class), 5))

# test combos for tweet level
def test_tl_combos():
    args['include_unigrams'] = True
    args['include_w2v_tl'] = False
    args['include_splex_tl'] = False
    args['include_w2v_cl'] = False
    args['include_splex_cl'] = False
    run_experiment()  # only unigrams

    args['include_splex_tl'] = True
    args['splex_scale'] = 'minmax'
    run_experiment()  # unigrams + splex minmax
    args['splex_scale'] = 'standard'
    run_experiment()  # unigrams + splex standard

    args['include_unigrams'] = False
    args['include_w2v_tl'] = True
    args['include_splex_tl'] = False
    run_experiment()  # only w2v

    args['include_splex_tl'] = True
    args['splex_scale'] = 'minmax'
    run_experiment()  # w2v + splex minmax
    args['splex_scale'] = 'standard'
    run_experiment()  # w2v + splex standard

    # args['include_unigrams'] = True
    # args['include_w2v_tl'] = True
    # args['include_splex_tl'] = False
    # run_experiment()  # unigrams and w2v
    #
    # args['include_splex_tl'] = True
    # args['splex_scale'] = 'minmax'
    # run_experiment()  # unigrams + w2v + splex minmax
    # args['splex_scale'] = 'standard'
    # run_experiment()  # unigrams+  w2v + splex standard

# find best w2v context
def test_w2v_context():
    # initialize all models that will be needed
    print('Testing context-level combos.')
    args['include_unigrams'] = False
    args['include_w2v_tl'] = True
    args['include_splex_tl'] = True
    args['splex_scale'] = 'standard'
    args['include_w2v_cl'] = True
    args['include_splex_cl'] = False
    models = init_models()

    args['w2v_use_rt'] = False
    args['w2v_use_mentions'] = False
    args['w2v_use_rt_mentions'] = False

    # test optimal size
    # args['w2v_size'] = 7
    # run_experiment(models=models)
    # args['w2v_size'] = 30
    # run_experiment(models=models)
    # args['w2v_size'] = 60
    # run_experiment(models=models)

    # test with retweets, mentions, and both
    args['w2v_size'] = 30
    args['w2v_use_rt'] = True  # only self + rt
    run_experiment(models=models)

    args['w2v_use_rt'] = False
    args['w2v_use_mentions'] = True  # only self + mentions
    run_experiment(models=models)

    args['w2v_use_rt'] = True
    run_experiment(models=models)  # self + rt + mentions

# find best w2v context
def test_splex_context():
    # initialize all models that will be needed
    print('Testing context-level combos.')
    args['include_unigrams'] = False
    args['include_w2v_tl'] = True
    args['include_splex_tl'] = True
    args['splex_scale'] = 'standard'
    args['include_w2v_cl'] = False
    args['include_splex_cl'] = True
    models = init_models()

    args['splex_use_rt'] = False
    args['splex_use_mentions'] = False
    args['splex_use_rt_mentions'] = False

    # test optimal size
    # args['splex_size'] = 2
    # run_experiment(models=models)
    # args['splex_size'] = 7
    # run_experiment(models=models)
    # args['splex_size'] = 30
    # run_experiment(models=models)
    # args['splex_size'] = 60
    # run_experiment(models=models)

    # test with retweets, mentions, and both
    args['splex_size'] = 30
    args['splex_use_rt'] = True  # only self + rt
    run_experiment(models=models)

    args['splex_use_mentions'] = True  # self + rt + mentions
    run_experiment(models=models)

def test_pairwise():
    # previous optimal
    args['include_unigrams'] = False
    args['include_w2v_tl'] = True
    args['include_splex_tl'] = True
    args['splex_scale'] = 'standard'
    args['include_w2v_cl'] = True
    args['w2v_size'] = 30
    args['w2v_use_rt'] = False
    args['w2v_use_mentions'] = False
    args['w2v_use_rt_mentions'] = False
    args['include_splex_cl'] = False
    models = init_models()

    args['pairwise'] = 'w2v'
    run_experiment(models=models)

    args['pairwise'] = 'splex'
    run_experiment(models=models)

    args['pairwise'] = 'both'
    run_experiment(models=models)

def test_char_ngrams():
    args['include_unigrams'] = False
    args['include_char_ngrams'] = True
    args['include_w2v_tl'] = True
    args['include_splex_tl'] = True
    args['include_w2v_cl'] = False
    args['include_splex_cl'] = False
    models = init_models()

    args['char_size'] = 10000

    args['char_min_gram'] = 1
    args['char_max_gram'] = 2
    run_experiment(models=models)

    args['char_min_gram'] = 1
    args['char_max_gram'] = 4
    run_experiment(models=models)

# test combos for context level
def test_cl_combo():
    # initialize all models that will be needed
    print('Testing context-level combos.')
    args['include_unigrams'] = False
    args['include_w2v_tl'] = True
    args['include_splex_tl'] = True
    args['splex_scale'] = 'standard'
    args['include_w2v_cl'] = True
    args['include_splex_cl'] = True
    models = init_models()

    # optimal w2v
    args['w2v_size'] = 30
    args['w2v_use_rt'] = False
    args['w2v_use_mentions'] = False
    args['w2v_use_rt_mentions'] = False

    # optimal splex
    args['splex_size'] = 30
    args['splex_use_rt'] = False
    args['splex_use_mentions'] = False
    args['splex_use_rt_mentions'] = False

    run_experiment(models=models)

def save_preds_for_stat_sig():
    path_to_best = '../cv_results/WT_ST_standard_WC_30_0.5_STAT_TST.pkl'
    args, preds, scores = pickle.load(open(path_to_best, 'rb'))
    my_idx_to_class = {0:'Loss', 1:'Aggression', 2:'Other'}
    ruiqi_class_to_idx = {'Aggression':0, 'Loss':1, 'Other':2}
    translated_preds = np.array([ruiqi_class_to_idx[my_idx_to_class[pred]] for pred in preds])
    print(Counter(translated_preds))
    np.savetxt('serina_svm_pred.np', translated_preds)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('-w', '--weights', type = str, default = 'static', help = 'weights for SVM: \'dynamic\' or \'static\'')

    parser.add_argument('-iu', '--include_unigrams', type = bool, default = False, help = 'whether to include unigrams')
    parser.add_argument('-usize', '--unigram_size', type = int, default = 10000, help = 'number of unigrams to include')

    parser.add_argument('-ic', '--include_char_ngrams', type = bool, default = False, help = 'whether to include char n-grams')
    parser.add_argument('-csize', '--char_size', type = int, default = 20000, help = 'number of char ngrams to include')
    parser.add_argument('-cmin', '--char_min_gram', type = int, default = 1, help = 'minimum n for char n-grams')
    parser.add_argument('-cmax', '--char_max_gram', type = int, default = 5, help = 'maximum n for char n-grams')

    parser.add_argument('-iwt', '--include_w2v_tl', type = bool, default = True, help = 'whether to include w2v embeddings at tweet-level; if false, w2v-tweet params are ignored')
    parser.add_argument('-d2v', '--use_d2v', type = bool, default = False, help = 'use doc2vec instead of aggregated w2v embedding for tweet-level')
    parser.add_argument('-wtmode', '--w2v_tl_mode', type = str, default = 'avg', help = 'how to combine w2v embeddings at tweet-level')

    parser.add_argument('-ist', '--include_splex_tl', type = bool, default = True, help = 'whether to include splex at tweet-level')
    parser.add_argument('-stscale', '--splex_scale', type = str, default = 'standard', help = 'which scaling of splex to use')
    parser.add_argument('-isub', '--include_sub_splex_tl', type = bool, default = False, help = 'whether to include splex substance use scores at tweet-level')
    parser.add_argument('-stmode', '--splex_tl_mode', type = str, default = 'sum', help = 'how to combine splex scores into tweet-level')

    parser.add_argument('-iwc', '--include_w2v_cl', type = bool, default = True, help = 'whether to include w2v embeddings in context; if false, w2v-context params are ignored')
    parser.add_argument('-wcmode', '--w2v_cl_mode', type = str, default = 'avg', help = 'how to combine tweet-level w2v embeddings into context-level')
    parser.add_argument('-wcsize', '--w2v_size', type = int, default = 30, help = 'w2v-context: number of days to look back')
    parser.add_argument('-wchl', '--w2v_hl', type = float, default = .5, help = 'w2v-context: half-life ratio')
    parser.add_argument('-wcrt', '--w2v_use_rt', type = bool, default = False, help = 'w2v-context: User A retweets User B\'s tweet -- if true,this tweet will be counted in User A and User B\'s context')
    parser.add_argument('-wcmen', '--w2v_use_mentions', type = bool, default = False, help = 'w2v-context: User A tweets, mentioning User B -- if true, this tweet will be in User A and User B\'s context')
    parser.add_argument('-wcrtmen', '--w2v_use_rt_mentions', type = bool, default = False, help = 'w2v-context: User A retweets User B\'s tweet, which mentioned User C -- if true,this tweet will counted in User A and User C\'s history')

    parser.add_argument('-isc', '--include_splex_cl', type = bool, default = False, help = 'whether to include splex in context; if false, splex-context params are ignored')
    parser.add_argument('-scmode', '--splex_cl_mode', type = str, default = 'sum', help = 'how to combine tweet-level splex scores into context-level')
    parser.add_argument('-scsize', '--splex_size', type = int, default = 60, help = 'splex-context: number of days to look back')
    parser.add_argument('-schl', '--splex_hl', type = float, default = 1, help = 'splex-context: half-life ratio')
    parser.add_argument('-scrt', '--splex_use_rt', type = bool, default = False, help = 'splex-context: User A retweets User B\'s tweet -- if true,this tweet will be counted in User A and User B\'s context')
    parser.add_argument('-scmen', '--splex_use_mentions', type = bool, default = False, help = 'splex-context: User A tweets, mentioning User B -- if true, this tweet will be in User A and User B\'s context')
    parser.add_argument('-scrtmen', '--splex_use_rt_mentions', type = bool, default = False, help = 'splex-context: User A retweets User B\'s tweet, which mentioned User C -- if true,this tweet will counted in User A and User C\'s history')

    parser.add_argument('-tn', '--tuning', type = bool, default = False, help = 'whether parameters are being tuned -- if true, cross-val will train on train and test on val per folder; if false, cross-val will train on train+val and test on test')

    args = vars(parser.parse_args())
    print(args)

    test_char_ngrams()
    # test_tl_combos()
    # test_w2v_context()
    # test_splex_context()
    # test_cl_combo()
    # run_experiment()

    # test_pairwise()

    # per_class = pickle.load(open('../cv_results/UNI_10000_STAT_TST.pkl', 'rb'))[1]
    # print_scores(per_class)

    # save_preds_for_stat_sig()
