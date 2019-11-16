# System import
import os

# External import
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

# Internal import
from data_pre_processing import load_data
from utils import plt_roc

# accuracy is: 0.9507168150062207
# f1 score is: 0.4766241157817708
# TRAIN_PATH = 'preprocess_without_punctuation.csv'  

# accuracy is: 0.9521179060197148
# f1 score is: 0.5007982120051084
# TRAIN_PATH = 'preprocess_without_tokenize.csv'

# accuracy is: 0.9526232175327782
# f1 score is: 0.511640754478731
# TRAIN_PATH = 'preprocess_without_removing_stopwords.csv' 

# accuracy is: 0.9503684563116087
# f1 score is: 0.4719156042523726
# TRAIN_PATH = 'preprocess_without_stem.csv'  

# accuracy is: 0.9528184515264618
# f1 score is: 0.5173669577475819
# TRAIN_PATH = 'preprocess_just_stem.csv'

# accuracy is: 0.9530022011675758
# f1 score: 0.5177736753211045
# test f1 score: 0.53762
TRAIN_PATH = 'preprocess_tokenize_and_stem.csv'

# accuracy is: 0.9504986123073978
# f1 score is: 0.4712328767123288
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
		if if_plt_roc:
			plt_roc(test_label_OR_test_data, prediction)

		print(f'accuracy is: {accuracy_score(test_label_OR_test_data, prediction)}')
		print(f'f1 score is: {f1_score(test_label_OR_test_data, prediction)}')


if __name__ == "__main__":
	load_test_data = False
    
	train_data, test_data, train_label, test_label = prepare_data(load_test_data)
	fit_and_predict(load_test_data, train_data, test_data, train_label, test_label, if_plt_roc=True)

