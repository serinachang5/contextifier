from data_loader import Data_loader
from nn_feature_support import init_tl
import numpy as np
import pickle

def get_pair_to_tids():
    print('Initializing Data Loader...')
    dl = Data_loader()
    test_ids = [tweet['tweet_id'] for tweet in dl.test_data()]
    pair2tids = {}
    for record in dl.all_data():
        if record['tweet_id'] not in test_ids:
            involved = set()
            involved.add(record['user_post'])
            if 'user_retweet' in record:
                involved.add(record['user_retweet'])
            if 'user_mentions' in record:
                for user in record['user_mentions']:
                    involved.add(user)
            involved = sorted(list(involved))

            for i, u1 in enumerate(involved):
                for u2 in involved[i+1:]:
                    pair_id = str(u1) + '_' + str(u2)
                    if pair_id in pair2tids:
                        pair2tids[pair_id].append(record['tweet_id'])
                    else:
                        pair2tids[pair_id] = [record['tweet_id']]

    return pair2tids

def get_pairwise_embeddings(pair2tids, cutoff=None):
    splex_tl = init_tl('splex')
    splex_dim = 2
    pair2splex = {}

    w2v_tl = init_tl('w2v')
    pair2w2v = {}
    w2v_dim = 200

    pair2splex['neutral'] = splex_tl.get_neutral_word_level()
    pair2w2v['neutral'] = w2v_tl.get_neutral_word_level()
    for pair, tids in pair2tids.items():
        if cutoff is not None and len(tids) < cutoff:
            pair2splex[pair] = pair2splex['neutral']
            pair2w2v[pair] = pair2w2v['neutral']
        else:
            splex_sum = np.zeros(splex_dim, dtype=np.float)
            w2v_sum = np.zeros(w2v_dim, dtype=np.float)
            for tid in pair2tids[pair]:
                splex_sum += splex_tl.get_representation(tid, mode='sum')
                w2v_sum += w2v_tl.get_representation(tid, mode='avg')
            pair2splex[pair] = splex_sum
            pair2w2v[pair] = w2v_sum / len(pair2tids[pair])

    return pair2splex, pair2w2v

def check_pairwise_embeddings(fname):
    pair2embs = pickle.load(open(fname, 'rb'))
    embs = np.array(list(pair2embs.values()))
    print(embs.shape)

    neutral = pair2embs['neutral']
    count_real = 0
    for emb in embs:
        if emb[0] != neutral[0] and emb[1] != neutral[1]:
            count_real += 1

    print('Real embs:', count_real)
    print('Under cutoff:', len(pair2embs)-count_real-1)

# pair2tids = get_pair_to_tids()
# cutoff=2
# pair2splex, pair2w2v = get_pairwise_embeddings(pair2tids, cutoff=cutoff)
# pickle.dump(pair2splex, open('pairwise_c{}_splex.pkl'.format(str(cutoff)), 'wb'))
# pickle.dump(pair2w2v, open('pairwise_c{}_w2v.pkl'.format(str(cutoff)), 'wb'))

check_pairwise_embeddings('pairwise_c2_splex.pkl')