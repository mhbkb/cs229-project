# System import
import os

# External import
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Internal import
from data_pre_processing import load_data
from utils import plt_roc


# TRAIN_PATH = 'preprocess_without_punctuation.csv'  

# TRAIN_PATH = 'preprocess_without_tokenize.csv'

# TRAIN_PATH = 'preprocess_without_removing_stopwords.csv' 

# TRAIN_PATH = 'preprocess_without_stem.csv'  

# TRAIN_PATH = 'preprocess_just_stem.csv'

# accuracy is: 0.9442855775672313
# f1 score is: 0.23745153515665934
TRAIN_PATH = 'preprocess_tokenize_and_stem.csv'

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


def fit_and_predict(load_test_data, 
					train_data, 
					test_feature_matrics, 
					train_label, 
					test_label_OR_test_data,
					if_plt_roc):
	model = XGBClassifier()
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


if __name__ == "__main__":
	load_test_data = False
    
	train_data, test_data, train_label, test_label = prepare_data(load_test_data)
	fit_and_predict(load_test_data, train_data, test_data, train_label, test_label, if_plt_roc=True)

