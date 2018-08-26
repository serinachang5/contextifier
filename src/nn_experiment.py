"""
===================
nn_experiment
Author: Ruiqi Zhong
Date: 05/04/2018
contains the Experiment class
A high level API that enables different experiment configurations
that require the minimum implementation
===================
"""

from keras.callbacks import EarlyStopping, ModelCheckpoint
from model_def import NN_architecture, input_name_is_user_idx
from data_loader import Data_loader
from generator_util import create_clf_data, create_data
import numpy as np
import subprocess
from sklearn.metrics import precision_recall_fscore_support, f1_score
from keras import backend as K
from keras import optimizers
import pickle as pkl
import time

# number of classes we are performing classification task
# currently 3
nb_classes = 3

# all the labeled ids
labeled_tids = np.loadtxt('../data/labeled_tids.np', dtype='int')

# extracting the dimension for each input name
# also assert that all the input from the same input name has the same dimension
# and that for each key input_name2id2np contains mapping from every labeled tweet
def extract_dim_input_name2id2np(input_name2id2np):
    dim_map = {}
    for input_name in input_name2id2np:
        
        # other input_name
        id2np = input_name2id2np[input_name]
        dim = None
        for id in id2np:
            if dim is None:
                dim = id2np[id].shape[0]
            # asserting the dimension for the same key across all tweets is the same
            assert(id2np[id].shape[0] == dim)
        dim_map[input_name] = dim

        # asserting that labeled tweets has a corresponding input
        for tid in labeled_tids:
            assert(tid in id2np)
    return dim_map

# making the predicted label one hot
def make_onehot(y):
    y_cat = K.argmax(y, axis=-1)
    return K.one_hot(y_cat, nb_classes)

# f1-score implementation for keras (for binary classification)
# copy pasted from stackoverflow
# https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# calculate the f score based on the Keras implementation
def macro_f1(y_true, y_pred):
    y_true, y_pred = make_onehot(y_true), make_onehot(y_pred)
    f = 0
    for class_idx in range(nb_classes):
        f = f + f1(y_true[:,class_idx], y_pred[:,class_idx])
    return f / nb_classes


# calculating the classweight given the y_train
# the weight will be inversely proprotional to the label
# belonging to that class
def calculate_class_weight(y_train):
    class_count = np.sum(y_train, axis=0)
    class_count = 1 / class_count
    class_count /= np.sum(class_count)
    class_weight = dict([(idx, class_count[idx]) for idx in range(len(class_count))])
    return class_weight

# for supervised learning without any pre-training
# we need to adjust the vocabulary size for all the content input
# mapping every word that occurs less than twice in the training set to 1
def adapt_vocab(X_train, X_list):
    threshold = 2

    for key in X_train:
        if key in [option + '_content_input' for option in ['char', 'word']] or input_name_is_user_idx(key):
            # count the number of occurence for each word
            wc = {}
            for xs in X_train[key]:
                for x in xs:
                    if wc.get(x) is None:
                        wc[x] = 0
                    wc[x] += 1

            # define a filter here
            def f(x):
                return x if (wc.get(x) is not None and wc[x] >= threshold) else 1

            # applying the filter to all x
            X_train[key] = np.array([[f(x) for x in xs] for xs in X_train[key]])
            for X in X_list:
                X[key] = np.array([[f(x) for x in xs] for xs in X[key]])

# an experiment class that runs cross validation
class Experiment:

    def __init__(self, experiment_dir, input_name2id2np=None, adapt_train_vocab=False,
                 comments='', epochs=100, patience=4, noise_function=None, filter_function=None, fold=5,
                 predict_ens_test=True, by_fold=False,
                 **kwargs):
        """
        an experiment class that runs cross validation
        designed to enable easy experiments with combinations of:
        1) context representation:
            handled by input_name2id2np
        2) pre-training methods:
            handled by pretrained_weight_dir in the kwargs argument
            None if there is no pretraining weight available
        3) char vs. word:
            specified in "options"
            options = ['char', 'word'] if you want to include both
            implement the value for key "word_content_input"
        options = ['char', 'word'] if you want to include everything

        Parameters
        ----------
        input_name2id2np:
        experiment_dir: the directory that the experiment weights and results will be saved
        adapt_train_vocab: under supervised training without pretraining,
                            some vocab will not be seen (twice) in the training set.
                            if set to True, then vocab occuring less than twice will be removed.
        comments: the comments that will be written to the README
        epochs: number of epochs of training during cross validation
        patience: number of epochs allowable for not having any improvement on the validation set
        kwargs: arguments that will be passed to initializing the neural network model (shown below)

        ========== below is the parameters needed by the neural network model ==========

        options: an array containing all the options considered in the neural network model ['char', 'word']
                    (probably splex in the future)
                    for each option, the input is mapped to a lower dimension,
                    then the lower dimension representation of each option is concatenated
                    and is followed by the final classification layer
        word_vocab_size: number of word level vocabs to be considered
        word_max_len: number of words in a tweet sentence
        char_vocab_size: number of char level vocabs to be considered
        char_max_len: number of chars in a tweet sentence
        drop_out: dropout rate for regularization
        filter: number of filters for each kernel size
        dense_size: the size of the dense layer following the max pooling layer
        embed_dim: embedding dimension for character and word level
        kernel_range: range of kernel sizes
        pretrained_weight_dir: a dictionary containing the pretrained weight.
                    e.g. {'char': '../weights/char_ds.weights'} means that the pretrained weight for character level model
                    is in ../weights/char_ds.weights
        weight_in_keras: whether the weight is in Keras
        context_dim: the dimension of context representation
        context_dense_size: the dense layer size right before the context representation
        splex_dense_size: dense layer size right before the splex reps
        """
        # creating the experiment dir
        # automatically generate a README
        if experiment_dir[:-1] != '/':
            experiment_dir += '/'
        experiment_dir = '../experiments/' + experiment_dir
        self.experiment_dir, self.kwargs = experiment_dir, kwargs
        subprocess.call(['rm', '-rf', experiment_dir])
        subprocess.call(['mkdir', experiment_dir])
        self.adapt_train_vocab = adapt_train_vocab
        self.predict_ens_test = predict_ens_test
        '''
        with open(self.experiment_dir + 'README', 'w') as readme:
            
            readme.write(comments + '\n')
            for key in kwargs:
                readme.write("%s: %s\n" % (str(key), str(kwargs[key])))
        '''
        if input_name2id2np is None:
            input_name2id2np = {}
        self.input_name2id2np = input_name2id2np
        self.fold = fold
        self.dl = Data_loader(option='both', labeled_only=True, **kwargs)
        self.epochs, self.patience = epochs, patience
        self.noise_function, self.filter_function = noise_function, filter_function
        self.pretrained_weight_dirs = self.kwargs.get('pretrained_weight_dirs')
        self.by_fold = by_fold

    # cross validation
    # write all results to the directory
    # see read_results for retrieving the performance
    def cv(self):
        results = []

        for fold_idx in range(self.fold):
            print('cross validation fold %d.' % (fold_idx + 1))

            # retriving cross validataion data
            fold_data = self.dl.cv_data(fold_idx)
            ((X_train, y_train), (X_val, y_val), (X_test, y_test)) = \
                create_clf_data(self.input_name2id2np, fold_data, return_generators=False)
        
            if self.predict_ens_test:
                # retrieving the ensemble data
                ensemble_data = self.dl.ensemble_data()
                X_ensemble, y_ensemble = create_data(self.input_name2id2np, ensemble_data)
            
                # retrieving the held-out test data
                held_out_data = self.dl.test_data()
                X_held_out, y_held_out = create_data(self.input_name2id2np, held_out_data)
            
            if self.filter_function is not None:
                def apply_filter(X):
                    X_filtered = {}
                    for key in X:
                        if 'char' in key and 'input' in key:
                            X_filtered[key] = np.array([self.filter_function(x[:])
                                                        for x in X[key]])
                        else:
                            X_filtered[key] = X[key]
                    return X_filtered
            
                X_train, X_val, X_test = apply_filter(X_train), apply_filter(X_val), apply_filter(X_test)
                if self.predict_ens_test:
                    X_ensemble, X_held_out = apply_filter(X_ensemble), apply_filter(X_held_out)

            # if no pretrained weights, adapting vocabulary so that those who appear in
            # X_train less than twice would not be counted
            if self.adapt_train_vocab:
                if self.predict_ens_test:
                    adapt_vocab(X_train, (X_val, X_test, X_ensemble, X_held_out))
                else:
                    adapt_vocab(X_train, (X_val, X_test))

            class_weight = calculate_class_weight(y_train)

            # initializing model, train and predict
            K.clear_session()
            self.kwargs['input_dim_map'] = extract_dim_input_name2id2np(self.input_name2id2np)

            # cross validation test data in categorical form
            y_test = np.argmax(y_test, axis=-1)
            
            # OBSOLETE
            if self.kwargs.get('mode') == 'ternary':
                if not self.by_fold:
                    self.model = NN_architecture(**self.kwargs).model
                else:
                    self.kwargs['pretrained_weight_dirs'] = self.pretrained_weight_dirs[fold_idx]
                    self.model = NN_architecture(**self.kwargs).model
                self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[macro_f1])

                # call backs
                es = EarlyStopping(patience=self.patience, monitor='val_loss', verbose=1)
                weight_dir = self.experiment_dir + str(fold_idx) + '.weight'
                mc = ModelCheckpoint(weight_dir, save_best_only=True, save_weights_only=True)
                callbacks = [es, mc]

                # fit for at least 1 epoch
                self.model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), class_weight=class_weight)

                # training
                self.model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), callbacks=callbacks,
                               epochs=self.epochs, class_weight=class_weight)
                self.model.load_weights(weight_dir)

                # prediction
                y_pred = self.model.predict(x=X_test)
                y_pred_val = self.model.predict(x=X_val)

                # saving predictions for ensembles
                np.savetxt(self.experiment_dir + 'pred_test' + str(fold_idx) + '.np', y_pred)
                np.savetxt(self.experiment_dir + 'pred_val' + str(fold_idx) + '.np', y_pred_val)
                np.savetxt(self.experiment_dir + 'truth_test' + str(fold_idx) + '.np', y_test)
                np.savetxt(self.experiment_dir + 'truth_val' + str(fold_idx) + '.np', y_test)

                # make y categorical
                y_pred = np.argmax(y_pred, axis=-1)

            # only cascade model is used and it is the default
            elif self.kwargs.get('mode') is None or self.kwargs.get('mode') == 'cascade':
                # initialize the predictions
                num_train, num_val, num_test = y_train.shape[0], y_val.shape[0], y_test.shape[0]
                y_pred_val, y_pred_test = [None] * num_val, [None] * num_test

                for class_idx in range(2):
                    # time the training
                    start = int(round(time.time() * 1000))

                    # create layer name that has prefix
                    # since for each fodl we train model for aggression and loss models separately
                    if class_idx  == 0:
                        self.kwargs['prefix'] = 'aggression'
                    else:
                        self.kwargs['prefix'] = 'loss'
                
                    # initialize a model
                    if not self.by_fold:
                        self.model = NN_architecture(**self.kwargs).model
                    else:
                        self.kwargs['pretrained_weight_dirs'] = self.pretrained_weight_dirs[fold_idx]
                        self.model = NN_architecture(**self.kwargs).model

                    self.model.compile(optimizer='adam', loss='binary_crossentropy')

                    # create the label for this binary classification task
                    _y_train_, _y_val_ = y_train[:,class_idx], y_val[:,class_idx]
                    num_positive_train = sum(_y_train_)

                    # call backs
                    es = EarlyStopping(patience=self.patience, monitor='val_loss', verbose=1)
                    weight_dir = self.experiment_dir + str(fold_idx) + '_' + str(class_idx) + '.weight'
                    mc = ModelCheckpoint(weight_dir, save_best_only=True, save_weights_only=True)
                    callbacks = [es, mc]

                    # training
                    if self.noise_function is None:
                        self.model.fit(x=X_train, y=_y_train_,
                                       validation_data=(X_val, _y_val_))
                        history = self.model.fit(x=X_train, y=_y_train_,
                                                 validation_data=(X_val, _y_val_),
                                                 callbacks=callbacks, epochs=self.epochs)
                    else:
                        def add_noise2data(X_train):
                            X_train_noised = {}
                            for key in X_train:
                                if 'char' in key and 'input' in key:
                                    X_train_noised[key] = np.array([self.noise_function(x[:])
                                                                    for x in X_train[key]])
                                else:
                                    X_train_noised[key] = X_train[key]
                            return X_train_noised
        
                        self.model.fit(x=add_noise2data(X_train), y=_y_train_)
                                       
                        best_val_loss, best_epoch = float('inf'), 0
                        for epoch_idx in range(1, self.epochs + 1):
                            self.model.fit(x=add_noise2data(X_train), y=_y_train_)
                            val_loss = self.model.evaluate(x=X_val, y=_y_val_)
                            print('validation loss for epoch %d: %.3f' % (epoch_idx, val_loss))
                            if val_loss < best_val_loss:
                                best_epoch = epoch_idx
                                best_val_loss = val_loss
                                self.model.save_weights(weight_dir)

                            if epoch_idx - best_epoch >= self.patience:
                                break
            
                    self.model.load_weights(weight_dir)

                    _y_pred_val_score, _y_pred_test_score = (self.model.predict(X_val).flatten(),
                                                             self.model.predict(X_test).flatten())
                    if self.predict_ens_test:
                        _y_pred_ensemble_score, _y_pred_held_out_score = (self.model.predict(X_ensemble).flatten(),
                                                                          self.model.predict(X_held_out).flatten())
                    
                    prefix = self.experiment_dir + 'fold_%d_class_%d_' % (fold_idx, class_idx)
                    
                    np.savetxt(prefix + 'pred_val.np', _y_pred_val_score)
                    np.savetxt(prefix + 'pred_test.np', _y_pred_test_score)
                    
                    if self.predict_ens_test:
                        np.savetxt(prefix + 'pred_ensemble.np', _y_pred_ensemble_score)
                        np.savetxt(prefix + 'pred_held_out.np', _y_pred_held_out_score)

                    # threshold tuning
                    best_t, best_f_val = 0, -1
                    for t in np.arange(0.01, 1, 0.01):
                        y_val_pred_ = [0] * num_val
                        for idx in range(num_val):
                            if y_pred_val[idx] is None and _y_pred_val_score[idx] >= t:
                                y_val_pred_[idx] = 1
                        f = f1_score(_y_val_, y_val_pred_)
                        if f > best_f_val:
                            best_f_val = f
                            best_t = t
                        # a temp variable that we do not want its value
                        # to be accidentally accessed by outside code
                        y_val_pred_ = None

                    # predictions made only when predictions not made by the previous model
                    # and larger than the best threshold
                    # true for both val_pred and test_pred
                    for idx in range(num_val):
                        if y_pred_val[idx] is None and _y_pred_val_score[idx] >= best_t:
                            y_pred_val[idx] = class_idx


                    for idx in range(num_test):
                        if y_pred_test[idx] is None and _y_pred_test_score[idx] >= best_t:
                            y_pred_test[idx] = class_idx
                    
                    end = int(round(time.time()))

                    # write how many time it takes for a run into the readme
                    duration = end - start
                    with open(self.experiment_dir + 'README', 'a') as readme:
                        readme.write('fold %d class %d takes %d seconds\n'
                                     % (fold_idx, class_idx, duration))

                # predict the rest as the "Other" class
                for idx in range(num_test):
                    if y_pred_test[idx] is None:
                        y_pred_test[idx] = 2
                
                for idx in range(num_val):
                    if y_pred_val[idx] is None:
                        y_pred_val[idx] = 2

            np.savetxt(self.experiment_dir + 'fold_%d_pred_val.np' % fold_idx, y_pred_val)
            np.savetxt(self.experiment_dir + 'fold_%d_pred_test.np' % fold_idx, y_pred_test)
            np.savetxt(self.experiment_dir + 'fold_%d_truth_val.np' % fold_idx, y_val)
            np.savetxt(self.experiment_dir + 'fold_%d_truth_test.np' % fold_idx, y_test)
            
            # append the result on this fold to results
            results.append(precision_recall_fscore_support(y_test, y_pred_test))

        # saving results
        results = np.array(results)
        np.savetxt(self.experiment_dir + 'result_by_fold.np', results.flatten())
        np.savetxt(self.experiment_dir + 'result_averaged.np', np.mean(results, axis=0))
        np.savetxt(self.experiment_dir + 'result_std.np', np.std(results, axis=0))

        avg_macro_f = np.mean(np.mean(results, axis=0)[2])
        with open(self.experiment_dir + 'README', 'a') as readme:
            readme.write('macro F-score: %.4f\n' % avg_macro_f)

if __name__ == '__main__':
    options = ['word']
    pretrained_weight_dirs = {'aggression_word_embed': ['../weights/w2v_word_s300_w5_mc5_ep20.np'], \
                                'loss_word_embed': ['../weights/w2v_word_s300_w5_mc5_ep20.np']}
    experiment = Experiment(experiment_dir='test',
                            pretrained_weight_dirs=pretrained_weight_dirs,
                            options=options)
    experiment.cv()
