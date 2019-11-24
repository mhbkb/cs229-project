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
from cnn_model import *

TRAIN_PATH = 'preprocess.csv'
TEST_PATH = 'test.csv'

print(psutil.cpu_count())

def get_data():
	return load_data(TRAIN_PATH)


def prepare_data():
	word_dict = build_word_dict()
	embeddings = EmbeddingLayer(word_dict)

	return embeddings


@timer
def fit_and_predict(load_test_data, 
					train_data, 
					test_feature_matrics, 
					train_label, 
					test_label_OR_test_data,
					if_plt_roc):
	

	model.fit(train_data, train_label)

	prediction = model.predict(test_feature_matrics)

	if load_test_data:
		del test_label_OR_test_data['question_text']
		
		test_label_OR_test_data.insert(1, 'prediction', prediction)
		test_label_OR_test_data.to_csv('submission.csv', index=False)
		return prediction
	else:
		if if_plt_roc:
			plt_roc(test_label_OR_test_data, prediction)

		print(f'accuracy is: {accuracy_score(test_label_OR_test_data, prediction)}')
		print(f'f1 score is: {f1_score(test_label_OR_test_data, prediction)}')
		print(f'confusion_matrix score is: {confusion_matrix(test_label_OR_test_data, prediction)}')


if __name__ == "__main__":
	load_test_data = False
    
	embeddings = prepare_data()
	train_test_split(embeddings, pd_data['target'], test_size=0.2, shuffle=False)
	fit_and_predict(load_test_data, train_data, test_data, train_label, test_label, if_plt_roc=True)
