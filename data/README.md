We release the tweet id used for experiments on different cross validation fold.
The file names are formatted as fold<fold_index><set_description>.txt; each line of the file is a tweet id.
<set_description>: train means "trainig set", val "validation set", test "test set".
For example, each id (each line) in file fold0val.txt corresponds to a tweet we use for the validation set in fold 0.
unlabeled.txt contains all the tweet_id by which we used to train the word embedding, splex scores and generate the context feautres.
fold 0-4 is used to for all experiments and fold 0-19 is used for calculating the statistical significance. 
