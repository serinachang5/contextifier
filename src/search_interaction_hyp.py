from nn_experiment import Experiment

if __name__ == '__main__':
    epochs, patience = 20, 5
    pretrained_weight_dirs = ({
                              # word embedding can be loaded here
                              'aggression_word_embed': ['../weights/w2v_word_s300_w5_mc5_ep20.np'],
                              'loss_word_embed': ['../weights/w2v_word_s300_w5_mc5_ep20.np'],
                              })
    interaction_dims = [32, 64, 128]
    drop_outs = [0.1, 0.25, 0.5, 0.6]
    num_runs = 5
    options = ['word']
    for run_idx in range(num_runs):
        for interaction_dim in interaction_dims:
            for drop_out in drop_outs:
                dir_name = 'search_interaction_hyp_size_%d_dropout_%.3f' % (interaction_dim, drop_out)
                experiment = Experiment(experiment_dir=dir_name,
                                        pretrained_weight_dirs=pretrained_weight_dirs,
                                        options=options,
                                        interaction_layer_dim=100, interaction_layer_drop_out=0.5)
                experiment.cv()


