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
    

def load_embeddings(file_path, word_index, num_words):
    print(f'load_embeddings: {file_path}')
    embeddings_index = {}

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        for line in tqdm(file, disable=False):
            values = line.split(' ')

            if 'glove' not in file_path and len(values) > 100:
                continue

            embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')

    all_embs = np.stack(embeddings_index.values())
    embedding_mean, embedding_std = all_embs.mean(), all_embs.std()
    # Need to nomornalize the size otherwise got "raise ValueError('all input arrays must have the same shape')."
    embedding_size = all_embs.shape[1]
    print(f'Embedding size: {embedding_size}')

    embedding_matrix = np.random.normal(embedding_mean, embedding_std, (num_words, embedding_size))
    word_not_fit = 0

    for word in tqdm(word_index, disable=False):
        if word not in embeddings_index:
            word_not_fit += 1
            continue

        embedding_matrix[word_index[word]] = embeddings_index[word]

    print(f'Word cover rate in the embedding is: {1 - (word_not_fit / num_words)}')
    return embedding_matrix
