# System import

# External import
import torch
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split


# Internal import
from data_pre_processing import build_word_dict, load_data
from utils import timer
from cnn_models import *

TRAIN_PATH = 'preprocess.csv'
TEST_PATH = 'test.csv'

CUDA = False

print(psutil.cpu_count())

def get_data():
	return load_data(TRAIN_PATH)


@timer
def build_word_dict(pd_data):
	for word in tqdm(pd_data['question_text'].values, disable=False):
		word_dict[word] += 1
		
	return word_dict


def add_features(pd):
    # TODO: Come up with more features.
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
    train_features, test_features = train_test_split(pd_data, test_size=0.2, shuffle=False)

	ss = StandardScaler()
    ss.fit(np.vstack((train_features, test_features)))
    train_features = ss.transform(train_features)
    test_features = ss.transform(test_features)

	return train_data, test_data, train_label, test_label, train_features, test_features


@timer
def fit_and_predict(load_test_data, 
					train_data, 
					test_data, 
					train_label, 
					test_label,
					train_features,
					test_features,
					embedding_data,
					if_plt_roc):
	cnn_model = CNNModel(len(embedding_data), hidden_size_1=1, num_layers_1=1, hidden_size_2=1, 
             			 num_layers_2=1, dense_size_1=1*2*4+3, dense_size_2=0, 1, embedding_data)

	if CUDA:
		cnn_model.cuda()

	print(f'Num of params to train: {len(list(cnn_model.parameters()))}')

	# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/
	# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/recurrent_neural_network/main.py
	batch_size = 100
	n_iters = 3000
	num_epochs = int(1.0 * n_iters / (1.0 * len(train_dataset) / batch_size))

	train_loader = torch.utils.data.DataLoader(dataset=train_data, 
	                                           batch_size=batch_size, 
	                                           shuffle=False)

	test_loader = torch.utils.data.DataLoader(dataset=test_data, 
	                                          batch_size=batch_size, 
	                                          shuffle=False)

	optimizer = torch.optim.SGD(cnn_model.parameters(), lr=0.01)
	criterion = nn.CrossEntropyLoss()

	iter = 0

	for epoch in range(num_epochs):
		print(f'Start epoch: {epoch}')

	    for i, questions in enumerate(train_loader):
	        optimizer.zero_grad()
	        outputs = cnn_model(questions)
	        loss = criterion(outputs, train_label)
	        loss.backward()
	        optimizer.step()

	        iter += 1

	        if iter % 500 == 0:
	            # Calculate Accuracy         
	            correct = 0
	            total = 0
	            # Iterate through test dataset
	            for i, questions in enumerate(test_loader):
	                outputs = cnn_model(questions)
	                _, predicted = torch.max(outputs.data, 1)

	                # Total number of labels
	                total += test_label.size(0)
	                correct += (predicted == test_label).sum()

	            accuracy = 100 * correct / total

	            # Print Loss
	            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))
                    
            
	# if load_test_data:
	# 	del test_label_OR_test_data['question_text']
		
	# 	test_label_OR_test_data.insert(1, 'prediction', prediction)
	# 	test_label_OR_test_data.to_csv('submission.csv', index=False)
	# 	return prediction
	# else:
	# 	if if_plt_roc:
	# 		plt_roc(test_label_OR_test_data, prediction)

	# 	print(f'accuracy is: {accuracy_score(test_label_OR_test_data, prediction)}')
	# 	print(f'f1 score is: {f1_score(test_label_OR_test_data, prediction)}')
	# 	print(f'confusion_matrix score is: {confusion_matrix(test_label_OR_test_data, prediction)}')


if __name__ == "__main__":
	load_test_data = False

	train_data, test_data, train_label, test_label, train_features, test_features = prepare_data()
	embedding_data = EmbeddingLayer()

	fit_and_predict(load_test_data, 
					train_data, 
					test_data, 
					train_label, 
					test_label,
					train_features,
					test_features,
					embedding_data,
					True)
