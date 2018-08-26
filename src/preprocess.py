# coding=utf-8

"""
===================
preprocessor
===================

Original Author: Siddharth Varia
Author: Ruiqi Zhong
Date: 04/19/2018
This module contains a function that perform preprocessing on strings.
I modified the preprocess as follows:
i) splitting emojis that were not splitted by spaces
ii) removing all attribute emojis (e.g. skin colors)
iii) return a utf-8 encoding rather than a string

"""
import re
from emoji import UNICODE_EMOJI
import unicodedata
from nltk.tokenize import TweetTokenizer

# a hand-curated set of emojis (modifiers)
# that would be excluded
excluded = set([b'\xf0\x9f\x8f\xbc',
                b'\xe2\x99\x80',
                b'\xe2\x83\xa3',
                b'\xef\xb8\x8f',
                b'\xe2\x99\x82',
                b'\xe2\x80\x8d'])

regex_punc = re.compile('[\\!\\"\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^\\`\\{\\|\\}\\~]')

def extract_mentioned_user_name(s):
    s = str(s)
    if len(s) <= 2:
        return set()
    tokens = s.split(' ')
    mentioned_users = set()
    for i, token in enumerate(tokens):
        if len(token) > 2 and token[0] == '@':
            # Confirm previous token is not 'RT', as that would be a retweet
            if i <= 1 and tokens[i-1] == 'RT':
                continue
            mentioned_user = token[1:]
            mentioned_users.add(mentioned_user.lower())
    return mentioned_users

def extract_user_rt(s):
    s = str(s)
    if len(s) <= 2 or s[:2] != 'RT':
        return None
    tokens = s.split(' ')
    if len(tokens) > 2 and len(tokens[1]) >= 2:
        return tokens[1][1:-1].lower()
    return None


def to_char_array(s):
    """
    making a binary string a list of chars
    take special care of emojis

    Parameters
    ----------
    s: a binary string of interest

    Returns
    -------
    char_array: a list of i) int if char has ascii 2) bytes if char is an emoji
    """
    if type(s) != bytes:
        raise ValueError('Input to to_char_array function should be in bytes')
    char_array = []
    tokens = s.split(b' ')
    for token in tokens:
        char_array.append(ord(' '))
        if isemoji(token):
            char_array.append(token)
            continue
        if token in [b'!url', b'@user']:
            char_array.append(token)
            continue
        for c in token:
            char_array.append(c)
    print(char_array)
    return char_array[1:]


# check whether an emoji is an emoji modifier
# (e.g. skin colors, etc)
def checkEmojiType(strEmo):
    if strEmo.encode('utf-8') in excluded:
        return False
    try:
        if unicodedata.name(strEmo).startswith("EMOJI MODIFIER"):
            return False
    except:
        return True
    return True

def isemoji(c):
    """
    Checking whether a character string c is an emoji

    Parameters
    ----------
    c: a character in byte representation

    Returns
    -------
    a boolean indicating whether the character is an emoji
    """
    if type(c) == str:
        c = c.encode('utf-8')
    c = bytes(c)
    return c.decode() in UNICODE_EMOJI

# a list of regular expressions that perform the preprocessing
regex_str = [
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)+',  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)

# returns a list of tokens after applying the regex
def tokenize(s):
    return tokens_re.findall(s)


# preprocess a raw string and returns its utf-8 representation
def preprocess(s, char_level=False, debug=False):

    # assert type of the input: must be a unicode string
    if type(s) != str:
        raise ValueError('Input to the preprocess function must be a unicode string')
    
    # make a copy of the original string
    original_string = s[:]
    # put a space after every emoji
    s = ''.join([' ' + c + ' ' if c in UNICODE_EMOJI else c for c in s if checkEmojiType(c)])
    # eliminate consecutive spaces
    s = s.replace('  ', ' ')

    # put a space after qoute because otherwise "@user instances will not be handled below
    s = re.sub('"', '" ', s)

    # replace ::emoji:: with empty string
    s = re.sub('::emoji::', '', s)
    tokens = tokenize(s)
    if not char_level:
        tokens = [token.lower() for token in tokens]

    html_regex = re.compile('<[^>]+>')
    tokens = [token for token in tokens if not html_regex.match(token)]

    mention_regex = re.compile('(?:@[\w_]+)')
    tokens = ['@user' if mention_regex.match(token) else token for token in tokens]

    url_regex = re.compile('http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+')
    tokens = ['!url' if url_regex.match(token) else token for token in tokens]

    s = ' '.join([t for t in tokens if t]).replace('rt @user : ', '')
    utf8encoded = s.encode('utf-8')
    if debug:
        print('----')
        print('The string being processed is:')
        print(original_string)
        print('----')
        print('The string after processing is, in unicode: ')
        print(s)
        print('----')
        print('The string being returned is:')
        print(utf8encoded)
        print('----')
        print('After splitting by spaces: ')
        print([c for c in utf8encoded.split(b' ')])
        print('----')
        print('Whether each of the token is an emoji')
        print([isemoji(c) for c in utf8encoded.split(b' ')])
    return utf8encoded

# a test case
if __name__ == '__main__':
    binary_str = preprocess(s, debug=True, char_level=True)
    print(binary_str)
    print([s.decode() for s in binary_str.split(b' ')])
    print(to_char_array(binary_str))
    print(extract_mentioned_user_name(s))
    print(extract_user_rt(s))
