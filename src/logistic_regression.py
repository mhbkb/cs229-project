# System import
import os

# External import
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Internal import
from data_pre_processing import load_data

# dev score is: 0.9506938463010814
# TRAIN_PATH = 'preprocess_without_punctuation.csv'  

# dev score is: 0.9521179060197148
# TRAIN_PATH = 'preprocess_without_tokenize.csv'

# dev score is: 0.952619389415255
# TRAIN_PATH = 'preprocess_without_removing_stopwords.csv' 

# dev score is: 0.9503646281940855
# TRAIN_PATH = 'preprocess_without_stem.csv'  

# dev score is: 0.952822279643985
# TRAIN_PATH = 'preprocess_just_stem.csv'

# dev score is: 0.9530175136376686
TRAIN_PATH = 'preprocess_tokenize_and_stem.csv'

# dev score is: 0.9504986123073978
# TRAIN_PATH = 'preprocess_all.csv' 

TEST_PATH = 'test.csv'


def prepare_data(load_test_data=False):
	pd_data = load_data(TRAIN_PATH)
	vector = TfidfVectorizer("english")

	feature_matrics = vector.fit_transform(pd_data['question_text'].values.astype('U'))

	# shuffle=False means pick the last 20% as dev data set.
	if load_test_data:
		test_data = load_data(TEST_PATH)
		test_feature_matrics = vector.transform(test_data['question_text'].values.astype('U'))
		return feature_matrics, test_feature_matrics, pd_data['target'], test_data
	else:
		return train_test_split(feature_matrics, pd_data['target'], test_size=0.2, shuffle=False)


def fit_and_predict(load_test_data, train_data, test_feature_matrics, train_label, test_label_OR_test_data):
	model = LogisticRegression(solver='liblinear', penalty='l2')
	model.fit(train_data, train_label)
	prediction = model.predict(test_feature_matrics)

	if load_test_data:
		del test_label_OR_test_data['question_text']
		# import pdb; pdb.set_trace()
		test_label_OR_test_data.insert(1, 'prediction', prediction)
		test_label_OR_test_data.to_csv('submission.csv', index=False)
		return prediction
	else:
		return accuracy_score(test_label_OR_test_data, prediction)

if __name__ == "__main__":
	load_test_data = True
    
	train_data, test_data, train_label, test_label = prepare_data(load_test_data)
	score = fit_and_predict(load_test_data, train_data, test_data, train_label, test_label)
	# print(f'dev score is: {score}')
