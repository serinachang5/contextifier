import numpy as np
import random

def helper_generator(X, y, batch_size, sample):
    counter = 0
    for key in X:
        num_data = len(X[key])
    order = [i for i in range(num_data)]
    while True:
        if sample and counter % (num_data / batch_size) == 0:
            random.shuffle(order)
        if sample:
            idxes = [order[idx % num_data] for idx in range(counter, counter + batch_size)]
        else:
            idxes = [idx % num_data for idx in range(counter, counter + batch_size)]
        counter += batch_size
        yield dict([(key, X[key][idxes]) for key in X]), y[idxes]

def create_data(input_name2id2np, tweet_dicts, return_generators=False,
                batch_size=32, sample=False):
    """
    A map that takes in tweet dictionaries and return data points readable for keras fit/fit_generator

    Parameters
    ----------

    tweet_dicts: a list of tweets dictionary
    return_generators: whether (generator, step_size) is returned or (X, y) is returned

    Returns
    -------
    X: key-worded inputs
    y: one-hot labels
        OR
    generator: a generator that will generate
    step_size: number of times for a generator to complete one epoch

    """
    data, keys = [], None

    # convert each tweet_dict to a dictionary that only contains field that is recognizable and useulf
    # for the model
    for tweet_dict in tweet_dicts:
        result = {}
        one_hot_labels = np.eye(3)
        if tweet_dict['label'] == 'Aggression':
            result['y'] = one_hot_labels[0]
        elif tweet_dict['label'] == 'Loss':
            result['y'] = one_hot_labels[1]
        else:
            result['y'] = one_hot_labels[2]
        result['word_content_input'] = tweet_dict['word_padded_int_arr']
        result['char_content_input'] = tweet_dict['char_padded_int_arr']
        for input_name in input_name2id2np:
            result[input_name + '_input'] = input_name2id2np[input_name][tweet_dict['tweet_id']]
        if keys is None:
            keys = [key for key in result]
        data.append(result)

    X = dict([(key, np.array([d[key] for d in data])) for key in keys])
    y = np.array([d['y'] for d in data])

    # return the entire datapoints and labels in one single array
    if not return_generators:
        return X, y

    generator = helper_generator(X, y, batch_size, sample)
    step_size = len(X) / batch_size
    return generator, step_size


# takes in tr, val, test, each of it a list of dictionaries
# besides basic y and word/char level input
# sets the input field by input_name2id2np
# create the cv fold of data for tr, val, test
def create_clf_data(input_name2id2np, tr_test_val_dicts, return_generators=False, batch_size=32):
    tr, val, test = tr_test_val_dicts
    return (create_data(input_name2id2np, tr, return_generators=return_generators, batch_size=batch_size, sample=True),
            create_data(input_name2id2np, val, return_generators=return_generators, batch_size=batch_size, sample=False),
            create_data(input_name2id2np, test, return_generators=return_generators, batch_size=batch_size, sample=False))

if __name__ == '__main__':
    from data_loader import Data_loader
    option = 'word'
    max_len = 50
    vocab_size = 40000
    dl = Data_loader(vocab_size=vocab_size, max_len=max_len, option=option)
    fold_idx = 0
    data_fold = dl.cv_data(fold_idx)
    tr, val, test = data_fold
    print(tr[0])
    '''
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_clf_data(simplest_tweet2data,
                                                                           data_fold)
    for key in X_train:
        print(X_train[key])
    '''
