"""
===================
model_def
===================
Author: Ruiqi Zhong
Date: 05/04/2018
This module includes a model class s.t. each component is exactly the same as the previous ACL paper
nevertheless, it allows combination of different models (concatenated at the last layer)
"""
import numpy as np
from keras.layers import Input, Dense, Conv1D, Embedding, concatenate, \
    GlobalMaxPooling1D, Dropout, Flatten
from keras.models import Model

# returns two tensors
# one for input_content, the other for tensor before final classification
def content2rep(option='word', vocab_size=40000, max_len=50, drop_out=0.5,
                filter=200, dense_size=256, embed_dim=300,
                kernel_range=(1,3), prefix='general'):

    # input layer
    # input will not have a prefix in its name
    input_content = Input(shape=(max_len,),
                          name= option + '_content_input')

    # embedding layer
    embed_layer = Embedding(vocab_size, embed_dim, input_length=max_len,
                            name= prefix + '_' + option + '_embed')
    e_i = embed_layer(input_content)
    embed_drop_out = Dropout(drop_out, name=prefix + '_' + option + '_embed_dropout')
    e_i = embed_drop_out(e_i)

    # convolutional layers
    conv_out = []
    for kernel_size in kernel_range:
        c = Conv1D(filter, kernel_size, activation='relu',
                   name= prefix + '_' + option + '_conv_' + str(kernel_size))(e_i)
        c = GlobalMaxPooling1D(name= prefix + '_' + option + '_max_pooling_' + str(kernel_size))(c)
        c = Dropout(drop_out, name= prefix + '_' + option + '_drop_out_' + str(kernel_size))(c)
        conv_out.append(c)
    agg = concatenate(conv_out)

    dense_layer = Dense(dense_size, activation='relu',
                        name= prefix + '_' + option + '_last')
    content_rep = dense_layer(agg)

    return input_content, content_rep

# return a boolean checking whether an input name is in user index format
def input_name_is_user_idx(input_name):
    if (('usr' in input_name or 'user' in input_name) # capturing various personal spelling habbits to prevent bug
        and ('idx' in input_name or 'index' in input_name or 'idex' in input_name)):
        return True
    return False

class NN_architecture:

    def __init__(self,
                 options,
                 input_dim_map=None,
                 word_vocab_size=40000, word_max_len=50,
                 char_vocab_size=1200, char_max_len=150,
                 drop_out=0.5,
                 filter=200, dense_size=256, embed_dim=300, kernel_range=range(1,3),
                 pretrained_weight_dirs=None, weight_in_keras=None,
                 num_users=50, user_embed_dim=32, user_embed_dropout=0,
                 interaction_layer_dim=-1, interaction_layer_drop_out=0.5,
                 mode='cascade',
                 prefix='general'):
        """
        Initilizing a neural network architecture according to the specification
        access the actual model by self.model

        Parameters
        ----------
        options: an array containing all the options considered in the neural network model ['char', 'word']
                    (probably splex in the future)
                    for each option, the input is mapped to a lower dimension,
                    then the lower dimension representation of each option is concatenated
                    and is followed by the final classification layer
        input_dim_map: a map from additional input name to its dimension
        word_vocab_size: number of word level vocabs to be considered
        word_max_len: number of words in a tweet sentence
        char_vocab_size: number of char level vocabs to be considered
        char_max_len: number of chars in a tweet sentence
        drop_out: dropout rate for regularization
        filter: number of filters for each kernel size
        dense_size: the size of the dense layer following the max pooling layer
        embed_dim: embedding dimension for character and word level
        kernel_range: range of kernel sizes
        pretrained_weight_dirs: a dictionary containing the pretrained weight.
                    e.g. {'char': '../weights/char_ds.weights'} means that the pretrained weight for character level model
                    is in ../weights/char_ds.weights
        weight_in_keras: whether the weight is in Keras
        """
        self.options, self.prefix = options, prefix
        print(self.prefix)
        if input_dim_map is None:
            input_dim_map = {}
        self.input_dim_map = input_dim_map

        # changeable hyper parameter
        self.drop_out = drop_out
        self.word_vocab_size, self.word_max_len = word_vocab_size, word_max_len
        self.char_vocab_size, self.char_max_len = char_vocab_size, char_max_len
        self.num_users, self.user_embed_dim, self.user_embed_dropout = num_users, user_embed_dim, user_embed_dropout
        self.interaction_layer_dim, self.interaction_layer_drop_out = interaction_layer_dim, interaction_layer_drop_out

        # hyper parameters that is mostly fixed
        self.filter, self.dense_size, self.embed_dim, self.kernel_range = filter, dense_size, embed_dim, kernel_range

        self.mode = mode

        # pretrained_weight directory
        self.pretrained_weight_dirs, self.weight_in_keras = pretrained_weight_dirs, weight_in_keras
        if self.pretrained_weight_dirs is None:
            self.pretrained_weight_dirs = {}
        if self.weight_in_keras is None:
            self.weight_in_keras = {}
        self.create_model()

    def create_model(self):
        # for each option, create computational graph and load weights
        inputs, last_tensors = [], []
        for option in self.options:

            # how to map char input to the last layer
            if option in ['char', 'word']:
                if option == 'char':
                    input_content, content_rep = content2rep(option,
                                                             self.char_vocab_size, self.char_max_len, self.drop_out,
                                                             self.filter, self.dense_size, self.embed_dim, self.kernel_range,
                                                             self.prefix)
                elif option == 'word':
                    input_content, content_rep = content2rep(option,
                                                             self.word_vocab_size, self.word_max_len, self.drop_out,
                                                             self.filter, self.dense_size, self.embed_dim, self.kernel_range,
                                                             self.prefix)
                inputs.append(input_content)
                last_tensors.append(content_rep)

        # the user name needs to have "user_idx" suffix to be considered user idx
        need_user_embedding = False
        for input_name in self.input_dim_map:
            if input_name_is_user_idx(input_name):
                need_user_embedding = True
        if need_user_embedding:
            user_embedding = Embedding(self.num_users, self.user_embed_dim, input_length=1,
                                       name=self.prefix + '_user_embed')
            user_embed_dropout_layer = Dropout(self.user_embed_dropout,
                                               name=self.prefix + '_user_embed_dropout')

        # directly concatenate addtional inputs (such as splex scores and context representations)
        # to the last layer
        for input_name in self.input_dim_map:
            if input_name_is_user_idx(input_name):
                input = Input(shape=(1,),
                              name=input_name + '_input')
                inputs.append(input)
                # flatten the user embedding (after dropout)
                input_embed = Flatten()(user_embed_dropout_layer(user_embedding(input)))
                last_tensors.append(input_embed)
            else:
                input = Input(shape=(self.input_dim_map[input_name],),
                                      name=input_name + '_input')
                inputs.append(input)
                last_tensors.append(input)

        # concatenate all the representations
        if len(last_tensors) >= 2:
            concatenated_rep = concatenate(last_tensors)
        else:
            concatenated_rep = last_tensors[0]

        if self.interaction_layer_dim != -1:
            interaction_layer = Dense(self.interaction_layer_dim, activation='relu',
                                      name=self.prefix + '_interaction_layer')
            concatenated_rep = interaction_layer(concatenated_rep)
            interaction_drop_out_layer = Dropout(self.interaction_layer_drop_out,
                                                 name = self.prefix + '_interaction_dropout')
            concatenated_rep = interaction_drop_out_layer(concatenated_rep)

        # out layer
        if self.mode == 'ternary':
            self.out_layer = Dense(3, activation='softmax',
                                   name=self.prefix + '_classification')
        elif self.mode == 'cascade':
            self.out_layer = Dense(1, activation='sigmoid',
                                  name=self.prefix+ '_classification')
        else:
            print('Error: mode %s not implemented' % self.mode)
            exit(0)
        out = self.out_layer(concatenated_rep)
        
        self.model = Model(inputs=inputs, outputs=out)

        layers = self.model.layers
        layer_dict = dict([(layer.name, layer) for layer in layers])
        self.model.summary()
        for layer_name in self.pretrained_weight_dirs:
            if layer_name in layer_dict:
                layer_dict[layer_name].set_weights([np.loadtxt(weight_dir) if type(weight_dir) == str else weight_dir
                                                    for weight_dir in self.pretrained_weight_dirs[layer_name]])
                print('weight of layer %s successfully loaded.' % layer_name)

if __name__ == '__main__':
    options = ['char', 'word']
    nn = NN_architecture(options,
                         word_vocab_size=40000, word_max_len=50,
                         char_vocab_size=1200, char_max_len=150,
                         pretrained_weight_dirs=None, weight_in_keras=None,
                         prefix='shabi')
