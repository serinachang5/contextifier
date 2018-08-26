from nn_experiment import Experiment
import pickle as pkl

if __name__ == '__main__':
    # run on each fold 5 times to achieve a stable performance estimate
    num_runs = 5
    options = ['word']

    # load the word embedding from the weights directory
    # w2v.np in weights directory is the word embedding trained on the data
    # here we train two models, one for aggression and another for loss
    pretrained_weight_dirs = ({
                              'aggression_word_embed': ['../weights/w2v.np'],
                              'loss_word_embed': ['../weights/w2v.np'],
                              })
                              
    # this is a map from feature_name to maps(id2np)
    # where the "id2np" is a map from tweet id to feature vector
    # the feature_names are: 1) user history embedding (90 days) 2) user splex (2 days) 3) pairwise embedding
    # For example, input_name2id2np['splex_2days'][12345] is the "splex 2 days" feature vector (in numpy)
    # for tweet with tweet id 12345
    input_name2id2np = pkl.load(open('../data/emnlp2017_name2id2np.pkl', 'rb'))
    
    # run experiment 5 times
    for run_idx in range(num_runs):
        # experiment class will automatically load the 5 fold data and perform cross validation
        # with the weighted loaded, context features specified
        # and then perform 5 fold cross validation
        # implementation details can be seen nn_experiment.py
        experiment = Experiment(experiment_dir='emnlp2017_' + str(run_idx),
                                options=options, pretrained_weight_dirs=pretrained_weight_dirs,
                                input_name2id2np=input_name2id2np)
        # cross validation method for the experiment class
        experiment.cv()
