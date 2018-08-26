"""
===================
viz.py
===================
Authors: Serina Chang
Date: 5/01/2018
Various visualizations of tweet_level representations.
"""

from data_loader import Data_loader
from represent_tweet_level import TweetLevel
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.preprocessing import MinMaxScaler

'''Functions to visualize labeled dataset using tweet-level representations, color-coded by Loss and Aggression.'''

# labeled_tweets is a list of (tweet_id, label) tuples
# include_sub only applies to emb_type 'splex' -- whether to include the Substance score or not
def plot_tweets(labeled_tweets, emb_type, want_labels = None, rep_mode = None, include_sub = True, force_TSNE = False):
    assert(emb_type == 'w2v' or emb_type == 'splex' or emb_type == 'd2v')
    if emb_type == 'w2v':
        tl = TweetLevel(emb_file='../data/w2v_word_s200_w5_mc5_ep20.bin')
        assert(rep_mode is not None)
    elif emb_type == 'splex':
        tl = TweetLevel(emb_file='../data/splex_minmax_svd_word_s300_seeds_hc.pkl')
        assert(rep_mode is not None)
    else:
        tl = TweetLevel(emb_file='../data/d2v_word_s300_w5_mc5_ep20.mdl')

    label_to_color = {'Aggression':'r', 'Loss':'b', 'Other':'g'}
    X = []
    color_map = []
    if want_labels is None:
        want_labels = ['Loss', 'Aggression']
    for (tweet_id, label) in labeled_tweets:
        if label in want_labels:
            rep = tl.get_representation(tweet_id, rep_mode)
            if emb_type == 'splex' and include_sub is False:
                rep = rep[:2]
            X.append(rep)
            color_map.append(label_to_color[label])
    X = np.array(X)
    print('Built tweet by embedding matrix of shape', X.shape)

    if X.shape[1] > 2 or force_TSNE:
        print('Transforming with TSNE...')
        X = TSNE(n_components=2).fit_transform(X)

    print('Plotting X with dimensions', X.shape)
    plt.figure(figsize=(6,6))  # make sure figure is square
    plt.scatter(X[:, 0], X[:, 1], c=color_map)

    specs = emb_type
    if emb_type != 'd2v':
        specs += '_' + rep_mode
    title = 'Visualization of tweet-level embeddings ({})'.format(specs)
    plt.title(title)

    plt.show()

def visualize_labeled_dataset():
    print('Initializing Data Loader')
    dl = Data_loader()

    tr, val, tst = dl.cv_data(fold_idx=0)
    labeled_tweets = tr + val + tst
    labeled_tweets = [(x['tweet_id'], x['label']) for x in labeled_tweets]
    print('Number of labeled tweets:', len(labeled_tweets))

    # plot_tweets(labeled_tweets, emb_type='splex', rep_mode='sum', include_sub=False, force_TSNE=True)
    plot_tweets(labeled_tweets, emb_type='w2v', rep_mode='avg')


'''Functions to visualize how Loss and Aggression interact for a given user over time, using SPLex tweet-level scores.'''

# get Loss/Agg scores for an optimized window of time
def get_window_scores(tl, tweets, min_tweets, max_days, agg_level):
    print('Searching for window of at least {} tweets within {} days...'.format(str(min_tweets), str(max_days)))
    loss_scores = {}
    agg_scores = {}

    # search until at least <min_tweet> tweets are found within <max_days> days
    i = num_tweets = start_i = 0
    while i < len(tweets):
        loss_scores = {}
        agg_scores = {}
        start_i = i
        start_time = tweets[start_i]['created_at']
        num_tweets = 0
        while i < len(tweets):
            tweet = tweets[i]
            curr_time = tweet['created_at']
            diff_in_sec = (curr_time - start_time).total_seconds()
            diff_in_min = diff_in_sec / 60
            diff_in_hour = diff_in_min / 60
            diff_in_day = diff_in_hour / 24

            if diff_in_day > max_days:  # greater than max days
                break

            assert(agg_level == 'day' or agg_level == 'hr' or agg_level == 'min' or agg_level == 'sec')
            if agg_level == 'day':
                time_marker = int(diff_in_day)
            elif agg_level == 'hr':
                time_marker = int(diff_in_hour)
            elif agg_level == 'min':
                time_marker = int(diff_in_min)
            else:  # sec
                time_marker = int(diff_in_sec)

            tweet_id = tweet['tweet_id']
            loss, agg, _ = tl.get_representation(tweet_id, mode='sum')
            if time_marker in loss_scores:
                loss_scores[time_marker].append(loss)
                agg_scores[time_marker].append(agg)
            else:
                loss_scores[time_marker] = [loss]
                agg_scores[time_marker] = [agg]
            num_tweets += 1
            i += 1

        if num_tweets < min_tweets:
            i += int(min_tweets/3)
        else:
            break

    return num_tweets, start_i, loss_scores, agg_scores

def plot_scores(loss_scores, agg_scores, num_tweets, user_id, tweet_id, agg_level):
    sorted_time = sorted(loss_scores.keys())
    y_loss = []
    y_agg = []
    for time in sorted_time:
        y_loss.append(np.mean(loss_scores[time]))
        y_agg.append(np.mean(agg_scores[time]))

    scaled = MinMaxScaler().fit_transform(list(zip(y_loss, y_agg)))
    y_loss = scaled.T[0]
    y_agg = scaled.T[1]

    time_covered = int(sorted_time[-1])
    print('Number of tweets:', num_tweets)

    plt.figure(figsize=(6,6))  # make sure figure is square
    loss_plot = '.b' if agg_level == 'sec' or agg_level == 'min' else '.b-'
    plt.plot(sorted_time, y_loss, loss_plot)  # blue dots connected by line
    agg_plot = '+r' if agg_level == 'sec' or agg_level == 'min' else '+r-'
    plt.plot(sorted_time, y_agg, agg_plot)  # red pluses connected by line
    plt.title('User {}, Tweet #{}: SPLex scores over {} {}s'.format(user_id, tweet_id, str(time_covered), agg_level))
    plt.show()

def visualize_user_over_time():
    data_pkl = pickle.load(open('../data/data.pkl', 'rb'))
    tweet_dict = data_pkl['data']
    uti = data_pkl['user_time_ind']
    print('Got data: {} tweets, {} users'.format(len(tweet_dict), len(uti)))

    tl = TweetLevel(emb_file='../data/splex_balanced_standard_svd_word_s300_seeds_hc.pkl')

    # parameters for finding and plotting scores
    min_tweets = 800
    max_days = 30
    agg_level = 'day'
    num_tweets = start_i = user_id = 0
    loss = None
    agg = None

    # if user unknown
    shuffled_users = list(uti.items())
    random.shuffle(shuffled_users)
    for user_i, user_tweets in shuffled_users:
        if len(user_tweets) > min_tweets:
            tweets = [tweet_dict[t_id] for t_id in user_tweets]
            scores = get_window_scores(tl, tweets, min_tweets=min_tweets, max_days=max_days, agg_level=agg_level)
            if scores[0] > min_tweets:
                num_tweets, start_i, loss, agg = scores
                user_id = user_i
                print('Found good window for User', user_id, 'starting with their tweet #', start_i)
                break

    # if user known
    # user_id = 10
    # user_tweets = uti[user_id]
    # tweets = [tweet_dict[t_id] for t_id in user_tweets]
    # scores = get_window_scores(tl, tweets, min_tweets=min_tweets, max_days=max_days, agg_level=agg_level)
    # if scores[0] > min_tweets:
    #     num_tweets, start_i, loss, agg = scores
    #     print('Found good window for User', user_id, 'starting with their tweet #', start_i)

    if num_tweets != 0:
        plot_scores(loss, agg, num_tweets, user_id=user_id, tweet_id=start_i, agg_level=agg_level)

if __name__ == '__main__':
    visualize_labeled_dataset()
   # visualize_user_over_time()
