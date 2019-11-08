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


def prepare_data():
	pd_data = load_data(TRAIN_PATH)

	feature_matrics = TfidfVectorizer("english").fit_transform(pd_data['question_text'].values.astype('U'))

	# shuffle=False means pick the last 20% as dev data set.
	return train_test_split(feature_matrics, pd_data['target'], test_size=0.2, shuffle=False)


def fit_and_predict(train_data, test_data, train_label, test_label):
	model = LogisticRegression(solver='liblinear', penalty='l2')
	model.fit(train_data, train_label)
	prediction = model.predict(test_data)
	return accuracy_score(test_label, prediction)


if __name__ == "__main__":
	score = fit_and_predict(*prepare_data())
	print(f'dev score is: {score}')
