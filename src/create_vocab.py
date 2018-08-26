"""
===================
create_vocab
===================
Author: Ruiqi Zhong
Date: 04/20/2018
This module implements a function that would take in the labeled and unlabeld corpus
and create a word dictionary in json
"""
from preprocess import preprocess, isemoji, to_char_array, extract_user_rt, extract_mentioned_user_name
import pandas as pd
import pickle as pkl

special_tokens = [b'_PAD_', b'_UNKNOWN_']

# count the number of ocurrences of char and word in a list of data directory
def count_occurrence(corpus_dirs):
    corpus_word_count, corpus_char_count, user_post_count, user_rt_count, user_mentioned_count = \
        [{} for _ in range(5)]
    tweet_id_read = set()
    for corpus_dir in corpus_dirs:
        df = pd.read_json(corpus_dir)
        texts = df['text'].values
        users = df['user_name'].values
        tweet_ids = df['tweet_id'].values
        for idx in range(len(texts)):
            # avoid reading the same tweet from two files twice
            text, tweet_id, user = texts[idx], tweet_ids[idx], users[idx].lower()
            if tweet_id in tweet_id_read:
                continue
            else:
                tweet_id_read.add(tweet_id)

            # preprocess the text
            text_bytes = preprocess(str(text))
            char_array = to_char_array(preprocess(str(text), char_level=True))

            # count the occurence of words
            for word in text_bytes.split(b' '):
                if corpus_word_count.get(word) is None:
                    corpus_word_count[word] = 0
                corpus_word_count[word] += 1

            # count the occurence of chars
            for c in char_array:
                if corpus_char_count.get(c) is None:
                    corpus_char_count[c] = 0
                corpus_char_count[c] += 1

            # count the ocurrence of users
            if user_post_count.get(user) is None:
                user_post_count[user] = 0
            user_post_count[user] += 1

            # count the ocurrence of a user being retweeted
            retweeted_user = extract_user_rt(text)
            if retweeted_user is not None:
                if user_rt_count.get(retweeted_user) is None:
                    user_rt_count[retweeted_user] = 0
                user_rt_count[retweeted_user] += 1

            # user mentions
            mentioned_users = extract_mentioned_user_name(text)
            for mentioned_user in mentioned_users:
                if user_mentioned_count.get(mentioned_user) is None:
                    user_mentioned_count[mentioned_user] = 0
                user_mentioned_count[mentioned_user] += 1

    return corpus_word_count, corpus_char_count, user_post_count, user_rt_count, user_mentioned_count

# adding a list of dictionary counts
def add_dictionary(dict_list):
    result = {}
    for dictionary in dict_list:
        for key in dictionary:
            if result.get(key) is None:
                result[key] = 0
            result[key] += dictionary[key]
    return result

# set the count to 0 if one entry is in another dictionary
def merge_dict(dict1, dict2):
    for key in dict1:
        if key not in dict2:
            dict2[key] = 0

    for key in dict2:
        if key not in dict1:
            dict1[key] = 0

# each word is mapped to a dictionary that describes its property
def new_property_dict(id):
    return {'id': id}


def get_token_properties(labeled_corpus_token_count, unlabeled_corpus_token_count, user_name=False):
    merge_dict(labeled_corpus_token_count, unlabeled_corpus_token_count)
    token2property = {b'_PAD_': new_property_dict(0), b'_UNKNOWN_': new_property_dict(1)}
    offset = len(token2property)
    # only consider tokens that occur more than once
    # ranked tokens first by number of occurence in labeled corpus
    # then in unlabeled corpus
    token_rank = sorted([w for w in labeled_corpus_token_count if
                         labeled_corpus_token_count[w] + unlabeled_corpus_token_count[w] >= 2],
                        key=lambda w: (-labeled_corpus_token_count[w],
                                       -unlabeled_corpus_token_count[w],
                                       str(w)))
    for idx in range(len(token_rank)):
        w = token_rank[idx]
        token2property[w] = new_property_dict(idx + offset)
        token2property[w]['occurence_in_labeled'] = labeled_corpus_token_count[w]
        token2property[w]['occurence_in_unlabeled'] = unlabeled_corpus_token_count[w]
        if not user_name:
            token2property[w]['isemoji'] = isemoji(w)

    for special_token in special_tokens:
        if not user_name:
            token2property[special_token]['isemoji'] = False
        token2property[special_token]['occurence_in_labeled'] = 0
        token2property[special_token]['occurence_in_unlabeled'] = 0

    return token2property

def merge_token2property(main_dict, dict_dict):
    for key in dict_dict:
        dictionary = dict_dict[key]
        for token in special_tokens:
            main_dict[token][key] = 0
        for token in main_dict:
            if token not in special_tokens:
                if dictionary.get(token) is None:
                    main_dict[token][key] = 0
                else:
                    main_dict[token][key] = dictionary[token]


def create_vocab(labeled_corpuses, unlabeled_corpuses, word_file_dir, char_file_dir, user_file_dir, verbose=False):
    """
    A function that takes in labeled and unlabeld corpuses and create vocabulary-index lookup

    Parameters
    ----------
    labeled_corpuses: a list of labeled corpus file directory in json format
    unlabeled_corpuses: a list of unlabeled corpus file directory in json format
    word_file_dir: output directory of the vocab-index lookup
    char_file_dir: output directory of the char-index lookup

    Returns
    -------
    word_dict: a word-dictionary that contains the information of a vocab
    char_dict: a char-dictionary that contains the information of a char
    """

    # counting the token occurrences in the labeled and unlabeled corpus
    if verbose:
        print('reading labeled corpus')
    labeled_corpus_word_count, labeled_corpus_char_count, labeled_corpus_user_post_count, \
       labeled_corpus_user_rt_count, labeled_corpus_user_mentioned_count = count_occurrence(labeled_corpuses)

    if verbose:
        print('reading unlabeled corpus')
    unlabeled_corpus_word_count, unlabeled_corpus_char_count, unlabeled_corpus_user_post_count, \
       unlabeled_corpus_user_rt_count, unlabeled_corpus_user_mentioned_count = count_occurrence(unlabeled_corpuses)

    # based on the word count, create a property dictionary for each word
    if verbose:
        print('calculating properties for words')
    word2property = get_token_properties(labeled_corpus_word_count, unlabeled_corpus_word_count)

    if verbose:
        print('calculating properties for characters')
    char2property = get_token_properties(labeled_corpus_char_count, unlabeled_corpus_char_count)

    labeled_user_count = add_dictionary([labeled_corpus_user_post_count,
                                         labeled_corpus_user_rt_count, labeled_corpus_user_mentioned_count])

    unlabeled_user_count = add_dictionary([unlabeled_corpus_user_post_count,
                                           unlabeled_corpus_user_rt_count, unlabeled_corpus_user_mentioned_count])
    if verbose:
        print('calculating properties for users')
    user2property = get_token_properties(labeled_user_count, unlabeled_user_count, user_name=True)

    merge_token2property(user2property, {'labeled_user_post': labeled_corpus_user_post_count,
                                         'labeled_user_rt': labeled_corpus_user_rt_count,
                                         'labeled_user_mentioned': labeled_corpus_user_mentioned_count,
                                         'unlabeled_user_post': unlabeled_corpus_user_post_count,
                                         'unlabeled_user_rt': unlabeled_corpus_user_rt_count,
                                         'unlabeled_user_mentioned': unlabeled_corpus_user_mentioned_count
                                         })
    del user2property[b'_PAD_']
    user2property['UNKNOWN_USER'] = user2property[b'_UNKNOWN_']
    del user2property[b'_UNKNOWN_']

    with open(word_file_dir, 'wb') as out_word_file:
       pkl.dump(word2property, out_word_file)

    with open(char_file_dir, 'wb') as out_char_file:
        pkl.dump(char2property, out_char_file)


    if '' in user2property:
        del user2property['']
        
    with open(user_file_dir, 'wb') as out_user_file:
        pkl.dump(user2property, out_user_file)
