import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

def plt_roc(label, prediction):
	tp, fp, _ = roc_curve(label, prediction)
	auc = roc_auc_score(label, prediction)
	plt.plot(tp, fp, label=f'Auc is {auc}')
	plt.show()


def timer(fn_to_measture):
	def func(*args):
		start = time.time()
		res = fn_to_measture(*args)
		time_cost = time.time() - start
		print(f'Spent {time_cost} s to run {fn_to_measture}')

		return res

	return func
    

def load_embeddings(file_path, word_count, num_words):
	print(f'load_embeddings: {file_path}')
	embeddings_index = {}

	with open(file_path, encoding='utf-8') as file:
		for line in file:
			values = line.split(' ')
			embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')

	all_embs = np.stack(embeddings_index.values())
	embedding_mean, embedding_std = all_embs.mean(), all_embs.std()
	embedding_size = all_embs.shape[1]

	embedding_matrix = np.random.normal(embedding_mean, embedding_std, (num_words + 1, embedding_size))

	for idx, word in tqdm(enumerate(word_count)):
		embedding_vect = embeddings_index[word]

		if embedding_vect is not None:
			embedding_matrix[idx] = embedding_vect

	return embedding_matrix
