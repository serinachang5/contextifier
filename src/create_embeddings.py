"""
===================
create_embeddings
===================
Authors: Ethan Adams & Serina Chang
Date: 04/22/2018
Train embeddings on the tweets in Data Loader (labeled + unlabeled data).
"""

import argparse
from data_loader import Data_loader
from gensim.models import Word2Vec, KeyedVectors, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import numpy as np
import pickle
from sentence_tokenizer import unicode_rep
from sklearn.feature_extraction.text import CountVectorizer
from csv import DictReader


def generate_w2v_embs(sentences, option):
	size = args['dim']
	window = args['window']
	min_count = args['min_count']
	epochs = args['epochs']

	print('Training Word2Vec...')
	model = Word2Vec(sentences, size=size, window=window,
					 min_count=min_count, iter=epochs)
	wv = model.wv
	print('Finished. Vocab size:', len(wv.vocab))
	vocab = list(sorted([w for w in wv.vocab], key=lambda x: int(x)))  # sort by idx
	print('First 10 words in vocab:', vocab[:10])
	print('Last 10 words in vocab:', vocab[-10:])

	# save word vectors (as binary)
	out_file = '../data/w2v_{0}_s{1}_w{2}_mc{3}_ep{4}.bin'.format(option, size, window, min_count, epochs)
	wv.save_word2vec_format(out_file, binary=True)
	print('Word2Vec vectors saved to', out_file)

def generate_svd_embs(sentences, option):
	size = args['dim']

	# get positive pointwise mutual info matrix
	mat, vocab = get_ppmi(sentences)

	# singular value decomposition - find most important eigenvalues
	u,s,v = np.linalg.svd(mat)
	print('Computed SVD')
	u = np.array(u)  # convert from matrixlib

	# make dictionary of unigram : embedding (truncated)
	embs = {}
	for i, word in enumerate(vocab):
		embs[word] = u[i, :size]
	print('Embedding dim:', len(list(embs.values())[0]))

	# save as pickle file
	out_file = '../data/svd_{0}_s{1}.pkl'.format(option, size)
	with open(out_file, 'wb') as f:
		pickle.dump(embs, f)
	print('SVD embeddings saved to', out_file)

def get_ppmi(sentences):
	vocab = [str(idx) for idx in range(1,20001)]
	count_model = CountVectorizer(vocabulary=vocab, token_pattern='\d+')  # get top 20k indices
	counts = count_model.fit_transform(sentences)
	counts.data = np.fmin(np.ones(counts.data.shape), counts.data)  # want occurence, not count
	n,v = counts.shape  # n is num of docs, v is vocab size
	print('n = {}, v = {}'.format(n,v))
	vocab = list(sorted([w for w in count_model.vocabulary_], key=lambda x: int(x)))  # sort by idx
	print('First 10 words in vocab:', vocab[:10])
	print('Last 10 words in vocab:', vocab[-10:])

	coo = (counts.T).dot(counts)  # co-occurence matrix is v by v
	coo.setdiag(0)  # set same-word to 0
	coo = coo + np.full(coo.shape, .01)  # smoothing

	marginalized = coo.sum(axis=0)  # smoothed num of coo per word
	prob_norm = coo.sum()  # smoothed sum of all coo
	print('Prob_norm:', prob_norm)
	row_mat = np.ones((v, v), dtype=np.float)
	for i in range(v):
		prob = marginalized[0,i] / prob_norm
		row_mat[i,:] = prob
	col_mat = row_mat.T
	joint = coo / prob_norm

	P = joint / (row_mat * col_mat)  # elementwise
	P = np.fmax(np.zeros((v, v), dtype=np.float), np.log(P))  # all elements >= 0
	print('Computed PPMI:', P.shape)
	return P, vocab

def generate_d2v_embs(sentences, tags, option):
	size = args['dim']
	window = args['window']
	min_count = args['min_count']
	epochs = args['epochs']

	docs = []
	for s,t in zip(sentences, tags):
		docs.append(TaggedDocument(s,t))

	print('Initializing Doc2Vec')
	model = Doc2Vec(documents=docs, dm=1,
					size=size, window=window,
					min_count=min_count)
	print('Training Doc2Vec on', len(docs), 'examples...')
	model.train(docs, total_examples=len(docs), epochs=epochs)
	print('Finished training.')
	model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

	# need to save the whole model to keep
	out_file = '../data/d2v_{0}_s{1}_w{2}_mc{3}_ep{4}.mdl'.format(option, size, window, min_count, epochs)
	model.save(out_file)
	print('Doc2Vec model saved to', out_file)

def sample_usage(fname, mode):
	assert(mode == 'w2v' or mode == 'svd' or mode == 'd2v')

	if mode == 'w2v':
		test_indices = [str(idx) for idx in range(10)]
		wv = KeyedVectors.load_word2vec_format(fname, binary=True)
		print('Number of embeddings in {}: {}'.format(fname, len(wv.vocab)))
		for idx in test_indices:
			if idx in wv.vocab:
				print(unicode_rep([int(idx)]), idx, wv[idx][:10])
			else:
				print('No embedding found for', unicode_rep([int(idx)]),  idx)

	elif mode == 'svd':
		test_indices = [str(idx) for idx in range(10)]
		svd_embs = pickle.load(open(fname, 'rb'))
		print('Number of embeddings in {}: {}'.format(fname, len(svd_embs)))
		for idx in test_indices:
			if idx in svd_embs:
				print(unicode_rep([int(idx)]), idx, svd_embs[idx][:10])
			else:
				print('No embedding found for', unicode_rep([int(idx)]), idx)

	else:  # mode == 'd2v'
		test_seqs = [['2', '254', '440', '192', '94', '57', '72', '77'],
				      ['2', '16', '60', '10', '219', '259', '16', '142', '538'],
				      ['6', '132', '130', '11646', '47', '6', '25', '4', '132', '130', '3934', '73', '12', '163', '3035', '545', '221', '545']]
		test_tags = [['740043438788345856'], ['258662084089368576'], ['842801723001487360']]
		model = Doc2Vec.load(fname)
		for seq,tag in zip(test_seqs, test_tags):
			inferred = model.infer_vector(seq)
			print(inferred[:10])


def main(args):
	# params for data loader
	option = args['option']
	print('Initializing Data Loader')
	dl = Data_loader(option=option)
	all_data = dl.all_data()
	print('Len of all data:', len(all_data))
	test_ids = set([tweet['tweet_id'] for tweet in dl.test_data()])
	print('Len of test data:', len(test_ids))
	ensemble_ids = get_ensemble_tids()
	print('Len of ensemble data:', len(ensemble_ids))

	mode = args['mode']
	assert(mode == 'w2v' or mode == 'svd' or mode == 'd2v')
	if mode == 'w2v':
		sentences = []
		for tweet in all_data:
			# need indices split
			if tweet['tweet_id'] not in test_ids and tweet['tweet_id'] not in ensemble_ids:
				sentences.append([str(x) for x in tweet['int_arr']])
		print('Num sentences:', len(sentences))
		print('Check sentence0:', sentences[0])
		generate_w2v_embs(sentences, option)
	elif mode == 'svd':
		sentences = []
		for i, tweet in enumerate(all_data):
			# need indices joined
			if tweet['tweet_id'] not in test_ids and tweet['tweet_id'] not in ensemble_ids:
				sentences.append(' '.join([str(x) for x in tweet['int_arr']]))
		print('Num sentences:', len(sentences))
		print('Check sentence0:', sentences[0])
		generate_svd_embs(sentences, option)
	else:  # mode == d2v
		sentences = []
		tags = []
		for tweet in all_data:
			if tweet['tweet_id'] not in test_ids and tweet['tweet_id'] not in ensemble_ids:
				# need indices split and use id's as tags
				sentences.append([str(x) for x in tweet['int_arr']])
				tags.append([str(tweet['tweet_id'])])
		print('Num sentences:', len(sentences))
		print('Check sentence0:', sentences[0])
		print('Check tag0:', tags[0])
		generate_d2v_embs(sentences, tags, option)

def get_ensemble_tids():
	tids = set()
	with open('../../datasets/2017_11_27/ensemble.csv') as f:
		reader = DictReader(f)
		for row in reader:
			tids.add(row['tweet_id'])
	return tids

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = '')
	parser.add_argument('-opt', '--option', type = str, default = 'word', help = 'embedding option: {\'word\', \'char\'}')
	parser.add_argument('-md', '--mode', type = str, default = 'w2v', help = 'mode of embedding: {\'w2v\', \'svd\'}')

	parser.add_argument('-dim', '--dim', type = int, default = 300, help = 'dimension of embeddings')
	parser.add_argument('-w', '--window', type = int, default = 5, help = 'window for word2vec; ignored if svd')
	parser.add_argument('-mc', '--min_count', type = int, default = 5, help = 'min_count for word2vec; ignored if svd')
	parser.add_argument('-ep', '--epochs', type = int, default = 20, help = 'iterations for word2vec; ignored if svd')

	args = vars(parser.parse_args())
	print(args)

	# main(args)
	option = args['option']
	print('Initializing Data Loader')
	dl = Data_loader(option=option)
	all_data = dl.all_data()
	all_tids = set([str(tweet['tweet_id']) for tweet in all_data])
	print(list(all_tids)[:10])
	print('Len of all data:', len(all_data))
	test_ids = set([tweet['tweet_id'] for tweet in dl.test_data()])
	print('Len of test data:', len(test_ids))
	ensemble_ids = get_ensemble_tids()
	print('Len of ensemble data:', len(ensemble_ids))
	print(list(ensemble_ids)[:10])
	assert(len(ensemble_ids.intersection(all_tids)) == 0)

	# w2v_file = '../data/w2v_word_s300_w5_mc5_ep20.bin'
	# svd_file = '../data/svd_word_s300.pkl'
	# sample_usage(w2v_file, svd_file)

	# test_sents = [['2', '254', '440', '192', '94', '57', '72', '77'],
	# 			  ['2', '16', '60', '10', '219', '259', '16', '142', '538'],
	# 			  ['6', '132', '130', '11646', '47', '6', '25', '4', '132', '130', '3934', '73', '12', '163', '3035', '545', '221', '545']]
	# test_tags = [['740043438788345856'], ['258662084089368576'], ['842801723001487360']]
	# generate_d2v_embs(test_sents, test_tags, 'word')
	# d2v_file = 'd2v_word_s300_w5_mc1_ep20.mdl'
	# sample_usage(d2v_file, mode='d2v')


