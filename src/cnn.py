# System import
import collections

# External import
import torch
import psutil
import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Internal import
from data_pre_processing import load_data
from utils import timer
from cnn_model import *

TRAIN_PATH = 'preprocess.csv'
TEST_PATH = 'test.csv'

CUDA = True if torch.cuda.is_available() else False


print(f'CUDA available: {CUDA}')
print(f'Total cpus: {psutil.cpu_count()}')

def get_data():
	return load_data(TRAIN_PATH)


@timer
def build_word_dict(pd_data):
	word_dict = collections.defaultdict(int)

	for setences in tqdm(pd_data['question_text'].values, disable=False):
		for word in setences.split(' '):
			word_dict[word] += 1
		
	return word_dict


def add_features(pd):
	# TODO: Come up with more features.
	# import pdb; pdb.set_trace()
	pd['total_length'] = pd['question_text'].apply(len)
	pd['capitals'] = pd['question_text'].apply(lambda x: sum(1 for c in x if c.isupper()))
	pd['caps_ratio'] = pd.apply(lambda x: float(x['capitals'])/float(x['total_length']), axis=1)
	pd['num_words'] = pd['question_text'].str.count('\S+')
	pd['num_unique_words'] = pd['question_text'].apply(lambda x: len(set(x.split(' '))))
	pd['unique_word_ratio'] = 1.0 * pd['num_unique_words'] / pd['num_words'] 


def prepare_data():
	pd_data = get_data()
	add_features(pd_data)
	train_data, test_data, train_label, test_label = train_test_split(pd_data['question_text'], pd_data['target'], test_size=0.2, shuffle=False)
	
	all_features = pd_data[['total_length', 'num_words', 'caps_ratio', 'unique_word_ratio']].fillna(0)
	train_features, test_features = train_test_split(all_features, test_size=0.2, shuffle=False)

	ss = StandardScaler()
	ss.fit(np.vstack((train_features, test_features)))
	train_features = ss.transform(train_features)
	test_features = ss.transform(test_features)

	word_dict = build_word_dict(pd_data)

	return train_data, test_data, train_label, test_label, train_features, test_features, len(word_dict)


@timer
def fit_and_predict(load_test_data, 
					train_data, 
					test_data, 
					train_label, 
					test_label,
					train_features,
					test_features,
					word_count,
					if_plt_roc):

	tokenizer = Tokenizer(num_words=120000)
	tokenizer.fit_on_texts(list(train_data))
	train_data = tokenizer.texts_to_sequences(train_data)
	test_data = tokenizer.texts_to_sequences(test_data)

	# Pad the train and test data. This is important otherwise the dimention of the inputs won't match!
	train_data = pad_sequences(train_data, maxlen=64)
	test_data = pad_sequences(test_data, maxlen=64)
	# import pdb; pdb.set_trace()
	embedding_data = EmbeddingLayer(tokenizer.word_index, word_count)
	cnn_model = CNNModel(embedding_size=300, hidden_size_1=96, num_layers_1=1, hidden_size_2=96, 
						 num_layers_2=1, dense_size_1=388, dense_size_2=24, output_size=1, embeddings=embedding_data)

	if CUDA:
		cnn_model.cuda()

	print(f'Num of params to train: {len(list(cnn_model.parameters()))}')

	# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/
	# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/recurrent_neural_network/main.py
	batch_size = 100
	n_iters = 3000
	num_epochs = 8
	print(f'Num of epochs: {num_epochs}')
	# import pdb; pdb.set_trace()
	train_data_cuda = torch.tensor(train_data, dtype=torch.long).cuda() if CUDA else torch.tensor(train_data, dtype=torch.long)
	train_features_cuda = torch.tensor(train_features, dtype=torch.float).cuda() if CUDA else torch.tensor(train_features, dtype=torch.float)
	test_data_cuda = torch.tensor(test_data, dtype=torch.long).cuda() if CUDA else torch.tensor(test_data, dtype=torch.long)
	test_features_cuda = torch.tensor(test_features, dtype=torch.float).cuda() if CUDA else torch.tensor(test_features, dtype=torch.float)

	train = torch.utils.data.TensorDataset(train_data_cuda, train_features_cuda)
	train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)

	test = torch.utils.data.TensorDataset(test_data_cuda, test_features_cuda)
	test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

	train_label_cuda = torch.tensor(train_label, dtype=torch.float).cuda() if CUDA else torch.tensor(train_label, dtype=torch.float)
	# test_label_cuda = torch.tensor(test_label, dtype=torch.float).cuda() if CUDA else torch.tensor(test_label, dtype=torch.float)

	optimizer = torch.optim.SGD(cnn_model.parameters(), lr=0.01)
	criterion = nn.BCEWithLogitsLoss()

	iter = 0
	should_stop = False
	previous_loss = None

	for epoch in range(num_epochs):
		if should_stop:
			break

		print(f'Start epoch: {epoch}')

		for my_train_data, my_train_features in train_loader:
			optimizer.zero_grad()
			outputs = cnn_model(my_train_data, my_train_features)
			# import pdb; pdb.set_trace()
			loss = criterion(outputs[:, 0], train_label_cuda)
			loss.backward()
			optimizer.step()

			iter += 1

			if iter % 100 == 0:
				# Calculate Accuracy  
				# import pdb; pdb.set_trace()       
				correct = 0
				total = 0
				# Iterate through test dataset
				for my_test_data, my_test_features in test_loader:
					outputs = cnn_model(my_test_data, my_test_features)
					_, predicted = torch.max(outputs.data, 1)

					# Total number of labels
					total += len(test_label)
					# import pdb; pdb.set_trace()
					correct += (predicted.numpy() == test_label).sum()

				accuracy = 1.0 * 100 * correct / total

				if abs(loss.item() - previous_loss) < 1e-3:
					print(f'Converged previous_loss:{previous_loss}, stopping...')
					should_stop = True

				# Print Loss
				print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))

			previous_loss = loss.item()
					
			
	# if load_test_data:
	#   del test_label_OR_test_data['question_text']
		
	#   test_label_OR_test_data.insert(1, 'prediction', prediction)
	#   test_label_OR_test_data.to_csv('submission.csv', index=False)
	#   return prediction
	# else:
	#   if if_plt_roc:
	#       plt_roc(test_label_OR_test_data, prediction)

	#   print(f'accuracy is: {accuracy_score(test_label_OR_test_data, prediction)}')
	#   print(f'f1 score is: {f1_score(test_label_OR_test_data, prediction)}')
	#   print(f'confusion_matrix score is: {confusion_matrix(test_label_OR_test_data, prediction)}')


if __name__ == "__main__":
	load_test_data = False

	train_data, test_data, train_label, test_label, train_features, test_features, word_count = prepare_data()

	fit_and_predict(load_test_data, 
					train_data, 
					test_data, 
					train_label, 
					test_label,
					train_features,
					test_features,
					word_count,
					True)
