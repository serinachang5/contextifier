# coding=utf-8

import pickle as pkl
from preprocess import preprocess, to_char_array

# loading the preprocessed dictionary
word2property = pkl.load(open('../model/word.pkl', 'rb'))
char2property = pkl.load(open('../model/char.pkl', 'rb'))

#print char2property.keys()

# creating an id to token map to debug
id2word = dict([(word2property[word]['id'], word) for word in word2property])
id2char = dict([(char2property[c]['id'], str(chr(c)) if bytes(c) < bytes(256) else c.decode()) for c in char2property])

# Given an array of integer, return a unicode representation
def unicode_rep(arr, option='word'):
    if option == 'char':
        return ''.join([id2char[id] for id in arr])
    elif option == 'word':
        return ' '.join([id2word[id] for id in arr])
    else:
        raise ValueError('option %s is not implemented.' % option)


def int_array_rep(s, option='word', vocab_count=50000, debug=False):
    """
    Given a unicode string and the vocab2property file (preprocessed)
    Return an int array representation of the sentence

    Parameters
    ----------
    s: the string to be processed, unicode encoding
    option: tokenize in word or character level. 'word' or 'char'
    vocab_count: the maximum vocabulary size; including PAD and UNKNOWN, #vocab_count tokens considered
    debug: if debug, print the result mapping int back to unicode string

    Returns
    -------
    result: an int array
    """
    if type(s) != str:
        raise ValueError('input to this tokenizer function must be a unicode string')
    if debug:
        print('tokenization at level: %s.' % option)

    if option == 'word':
        utf8encode = preprocess(s)
    else:
        utf8encode = preprocess(s, char_level=True)
    result = []
    if option == 'word':
        for token in utf8encode.split(b' '):
            if token in word2property and word2property[token]['id'] < vocab_count:
                result.append(word2property[token]['id'])
            else:
                result.append(1)
        if debug:
            print('mapping back to unicode string: %s' % ' '.join([id2word[id] for id in result]))
        return result
    elif option == 'char':
        print(utf8encode)
        char_array = to_char_array(utf8encode)
        for c in char_array:
            if c in char2property and char2property[c]['id'] < vocab_count:
                result.append(char2property[c]['id'])
            else:
                result.append(1)
        if debug:
            print('mapping back to unicode string: %s' % (''.join([id2char[id] for id in result])))
        return result
    else:
        raise ValueError('Option %s not implemented' % option)

if __name__ == '__main__':
    s = 'FREE 🔓🔓 BRO @ReesemoneySODMG Shit is FU 😤😤👿 .....👮🏽👮🏽💥💥💥🔫 #ICLR https://dd'
    arr = int_array_rep(s, option='char', debug=True)
    print(arr)
    print(unicode_rep(arr, option='char'))

