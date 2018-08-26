"""
===================
represent_tweet_level.py
===================
Authors: Ethan Adams and Serina Chang
Date: 04/27/2018
Generate and write context embeddings.
"""
import csv
import numpy as np
from data_loader import Data_loader
from represent_tweet_level import TweetLevel

class Contextifier:
    '''
        Creates the context for tweets.
    '''
    # Magic strings to determine relationship between user and tweet
    # From User A's perspective:
    SELF = 'SELF'               # User A tweets
    RETWEET = 'RETWEET'         # User A's tweet is retweted
    MENTION = 'MENTION'         # User A is mentioned in User B's tweet
    RETWEET_MENTION = 'RETWEET_MENTION'     # User C retweets user B's tweet,
                                                # in which user A was mentioned.
    POST_TYPES = [SELF, RETWEET, MENTION, RETWEET_MENTION]
    
    
    def __init__(self, tweet_level, post_types, context_size, context_hl_ratio,
                    context_combine, tl_combine):
        '''
        Create it!
        Args:
            tweet_level (TweetLevel): instance of TweetLevel class. See
                represent_tweet_level.py.
            post_types (list (str)): post types, must be within self.POST_TYPES
            context_size (float): Number of days to look back or 'ALL'
            context_hl_ratio (float): Ratio of half life to context size. 
                    Tweet embeddings will be weighed according to
                    self.decay_rate)^(t/x) where t is the number of days the 
                    previous tweet is from the current one, and x is 
                    context_size * context_hl Set to 0 for no weighting/decay.
            context_combine (str): method of combining tweet embeddings,
                currently 'sum', 'avg', 'max'
            tl_combine (str): method of combining embeddings into a tweet embedding,
                currently 'sum', 'avg', 'max'

        '''
        # Save variables
        self.tweet_level = tweet_level
        self.set_post_types(post_types)
        self.set_context_size(context_size)
        self.set_context_hl_ratio(context_hl_ratio)
        self.set_context_combine(context_combine)
        self.set_tl_combine(tl_combine)

        # To allow for loading from files
        self.tweet_to_ct = {}

        # Storage for quick access, provided settings stay the same - good for cross-val
        # These are not static contexts like those from files, so cache needs be emptied
        #     every time settings are changed
        self.cache = {}


    def empty_cache(self):
        self.cache = {}

    def set_post_types(self, post_types):
        if not hasattr(self, 'post_types') or post_types != self.post_types:
            self.empty_cache()
            self.post_types = []
            for p in post_types:
                if p not in self.POST_TYPES:
                    raise ValueError('Unrecognized post type:', p)
                self.post_types.append(p)

    def set_context_size(self, context_size):
        if not hasattr(self, 'context_size') or context_size != self.context_size:
            self.empty_cache()
            self.context_size = context_size

    def set_context_hl_ratio(self, context_hl_ratio):
        if not hasattr(self, 'context_hl_ratio') or context_hl_ratio != self.context_hl_ratio:
            self.empty_cache()
            self.context_hl_ratio = context_hl_ratio
            self.decay_rate = 0.5 # hardcoded!

    def set_context_combine(self, context_combine):
        if not hasattr(self, 'context_combine') or context_combine != self.context_combine:
            self.empty_cache()
            self.context_combine = context_combine

    def set_tl_combine(self, tl_combine):
        if not hasattr(self, 'tl_combine') or tl_combine != self.tl_combine:
            self.empty_cache()
            self.tl_combine = tl_combine

    def print_settings(self):
        print('Context size:', self.context_size)
        print('Half-life ratio:', self.context_hl_ratio)
        print('Context combine:', self.context_combine)
        print('Tweet-level combine:', self.tl_combine)
        print('Post types:', self.post_types)


    def assemble_context(self, all_data):
        '''
        Sorts the tweets into self.user_ct_tweets, based on the variables
            self.use_rt_user, self.use_rt_mentions, and self.use_mentions.
        Args:
            all_data (list): result of Data_loader.all_data(). All the tweets in the corpus.
        Returns:
            user_ct_tweets (dict (str -> list((int, str))): map from user_id to
                list of (tweet, type) where tweet is the tweet and type is one
                of self.SELF, self.RETWEET, self.MENTION, self.RETWEET_MENTION.
            id_to_location (int -> (str, int)): map from tweet id to
                (username, index in user_ct_tweets).
        '''

        # Tweets in a user's "context"
        user_ct_tweets = {}
        
        # Map from tweet id to tuple of (user, idx in sorted list)
        # Note that "user" is user_post, the user who posted the tweet
        id_to_location = {}
        
        # For every tweet in the dataset (labled and unlabeled)
        for tweet in all_data:
            incl_users = []
            # Always include poster
            incl_users.append((tweet['user_post'], self.SELF))
            # Check if tweet is a retweet
            if 'user_retweet' in tweet:
                incl_users.append((tweet['user_retweet'], self.RETWEET))
                # Include users mentioned in retweet
                rt_mentions = [(u, self.RETWEET_MENTION) for u in tweet['user_mentions']]
                incl_users.extend(rt_mentions)
            else:
                # Include users mentioned (not retweet)
                mentions = [(u, self.MENTION) for u in tweet['user_mentions']]
                incl_users.extend(mentions)
            
            # Add tweets to users' context tweets
            for u, post_type in incl_users:
                if u in user_ct_tweets:
                    user_ct_tweets[u].append((tweet, post_type))
                else:
                    user_ct_tweets[u] = [(tweet, post_type)]
        
        # Sort context tweets chronologically
        for u in user_ct_tweets:
            user_ct_tweets[u] = sorted(user_ct_tweets[u], key=lambda t: t[0]['created_at'])
            
        # Go through the tweets to save their location
        for u, tweets in user_ct_tweets.items():
            for idx, t in enumerate(tweets):
                if u == t[0]['user_post']:
                    id_to_location[t[0]['tweet_id']] = (u, idx)

        return user_ct_tweets, id_to_location


    def set_context(self, user_ct_tweets, id_to_location):
        '''
        Set the context.
        Args:
            user_ct_tweets: see self.assemble_context().
            id_to_location: see self.assemble_context().
        Returns:
            None
        '''
        self.user_ct_tweets = user_ct_tweets
        self.id_to_location = id_to_location
    
    
    def get_tweet_embedding(self, tweet_id, mode):
        '''
        Get the tweet embedding for the given tweet.
        Args:
            tweet_id (int): the id of the tweet, according to twitter's ID system.
            mode (str): the mode of combining embeddings at the tweet level.
                See TweetLevel.get_representation()
        Returns:
            the tweet embedding
        '''
        return self.tweet_level.get_representation(tweet_id, mode)

    def get_neutral_embedding(self):
        '''
        Get a neutral embedding. Note: this could eventually be changed
            to get a more accurate neutral embedding, by using the
            method of combining at the tweet level.
        Args:
            None
        Returns:
            A neutral embedding
        '''
        return self.tweet_level.get_neutral_word_level()

    def get_dimension(self):
        '''
        Get the dimension of tweet level representation.
        Args:
            None
        Returns:
            (int): the dimension. See TweetLevel.get_dimension().
        '''
        return self.tweet_level.get_dimension()


    def _combine_embeddings(self, embeddings, mode):
        '''
        Combine embeddings, according the given mode.
        Args:
            embeddings (list): a list of embeddings
            mode (str): 'avg', 'sum', or 'max'
        Returns:
            the combined embeddings (dims equivalent to size of one embedding
                in embeddings)
        '''
        result = None
        if mode == 'avg':
            result = np.mean(np.array(embeddings), axis=0)
        elif mode == 'sum':
                result = sum(embeddings)
        elif mode == 'max':
            result = np.max(np.array(embeddings), axis=0)
        else:
            raise ValueError('Unknown combination method:', mode)
        return result
    
    
    def _create_context_embedding(self, user_id, tweet_idx):
        '''
        Get the context embedding for the given tweet, determined by user and index.
        Args:
            user_id (int): the id of the user, according to data_loader's user ids
            tweet_idx (int): the index of the tweet in self.user_ct_tweets[user_id]
        Returns:
            (list (np.array(int))): the context embeddings
            (list (int)): the ids of tweets in the context window
        '''
        
        # Return difference in days, as a float
        def days_diff(d1, d2):
            return (d1 - d2).total_seconds() / 60 / 60 / 24
        
        embs = [] # embeddings
        context_hl = self.context_size * self.context_hl_ratio # set half life
        
        today = self.user_ct_tweets[user_id][tweet_idx][0]['created_at']
        i = tweet_idx-1
        while i >= 0 and days_diff(today, self.user_ct_tweets[user_id][i][0]['created_at']) \
                                     < self.context_size:
            
            # Confirm post type is one we want to include
            post_type = self.user_ct_tweets[user_id][i][1]
            if post_type not in self.post_types:
                i -= 1
                continue 
            
            # Save tweet ids
            if keep_ids:
                tweet_ids.append(self.user_ct_tweets[user_id][i][0]['tweet_id'])

            # Get embedding
            emb = self.get_tweet_embedding(self.user_ct_tweets[user_id][i][0]['tweet_id'],
                                           self.tl_combine)

            # Weigh embedding
            if context_hl not in [0, 1]:
                diff = days_diff(today, self.user_ct_tweets[user_id][i][0]['created_at'])
                weight = self.decay_rate ** (diff/context_hl)
                emb = emb * weight

            # Save
            embs.append(emb)
            i -= 1

        # Combine word embeddings
        result = None
        if len(embs) == 0:
            result = self.get_neutral_embedding()
        else:
            result = self._combine_embeddings(embs, self.context_combine)

        return result, tweet_ids


    def get_context_embedding(self, tweet_id):
        '''
        Get the context embedding for the specified tweet, determined by tweet_id
        Args:
            tweet_id (int): the id of the tweet, according to the twitter tweet ids.
        Returns:
            (np.array(int)): the context embedding
        '''
        if tweet_id in self.tweet_to_ct: # Allows for reading in from files
            return self.tweet_to_ct[tweet_id]
        if tweet_id in self.cache:
            return self.cache[tweet_id]
        context_embedding = self._create_context_embedding(self.id_to_location[tweet_id][0],
                                                           self.id_to_location[tweet_id][1])
        self.cache[tweet_id] = context_embedding
        return context_embedding
    
    def from_file(self, in_file):
        '''
        Reads the context embeddings in from a file.
        Args:
            in_file (str): the path to the file, in csv format, <tweet_id>, <embedding>
        Returns:
            None
        '''
        with open(in_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.tweet_to_ct[int(row['tweet_id'])] = np.fromstring(row['context_embedding'],
                                                                       dtype=float,
                                                                       sep=' ')
        

    def write_context_embeddings(self, dl, out_file=None):
        '''
        Writes the embeddings to a file.
        Args:
            out_file (str): the path of the file to write to
        Returns:
            None
        '''

        if not out_file:
            out_file = 'context_embeddings.csv' # should add in default name here
        tweet_to_ct = {}
        for fold_idx in range(0, 1):
            tr, val, test = dl.cv_data(fold_idx)
            all_tweets = [t for l in [tr, val, test] for t in l ]
            for tweet in all_tweets: 
                tweet_to_ct[tweet['tweet_id']] = self.get_context_embedding(tweet['tweet_id'])

        with open(out_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['tweet_id', 'context_embedding'])
            for tweet_id, ct_emb in tweet_to_ct.items():
                ct_emb_str = ' '.join([str(x) for x in ct_emb])
                writer.writerow([tweet_id, ct_emb_str])


if __name__ == '__main__':

    # Tester/usage
    tweet_level = TweetLevel('../data/splex_minmax_svd_word_s300_seeds_hc.pkl')
    post_types = [Contextifier.SELF,
                  Contextifier.RETWEET,
                  Contextifier.MENTION,
                  Contextifier.RETWEET_MENTION]
    context_size = 2
    context_hl_ratio = 0.5
    context_combine = 'avg'
    tl_combine = 'sum'
    
    print('Initializing Contextifier...')
    contextifier = Contextifier(tweet_level,
                                post_types,
                                context_size,
                                context_hl_ratio,
                                context_combine,
                                tl_combine)

    print('Loading Data...')
    option = 'word'
    max_len = 53
    vocab_size = 30000
    dl = Data_loader(vocab_size=vocab_size, max_len=max_len, option=option)
    dl.all_data()

    print('Creating contexts...')
    context = contextifier.assemble_context(dl.all_data())
    contextifier.set_context(*context)

    # Only necessary if you want to write them all to a file.
    # Can be done "on-demand" with .get_context_embedding()

    print('Writing context embeddings...')
    contextifier.write_context_embeddings(dl)

    # Alternatively, to load from a file, do:
    # contextifier.from_file('../data/'context_emb_5_avg_rtFalse_menTrue_rtmenFalse_hl1.0_.csv')