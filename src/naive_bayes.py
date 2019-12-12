# System import
import os

import pdb
import numpy as np

# External import
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, log_loss, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Internal import
from data_pre_processing import load_data
from utils import plt_roc

#accuracy is: 0.9405455067470572
#f1 score is: 0.11093937832732269 #best tfidf
#vectorized
#accuracy is: 0.9332835678055317
#f1 score is: 0.5288711072664359
#TRAIN_PATH = 'preprocess_without_punctuation.csv'  

#TRAIN_PATH = 'preprocess_without_punctuation_new.csv'
#training loss:  0.16490205440536118
#accuracy is: 0.9398602737104029
#f1 score is: 0.10197782096718873
#[[244623  15223]
#[   487    892]]
#testing loss:  0.17232970043286167
#precision:  0.6468455402465555
#recall:  0.05535215637604716

#accuracy is: 0.9400746482917025
#f1 score is: 0.08209217778820219 
#vectorized
#accuracy is: 0.9367862953392669
#f1 score is: 0.5501157880397766
#TRAIN_PATH = 'preprocess_without_tokenize.csv'

#TRAIN_PATH = 'preprocess_without_tokenize_new.csv'
#training loss:  0.16409026205662797
#accuracy is: 0.9403234759307111
#f1 score is: 0.09947432268499798
#[[244775  15254]
#[   335    861]]
#testing loss:  0.17298740947302232
#precision:  0.7198996655518395
#recall:  0.05342848278001862

#accuracy is: 0.9395808211312088
#f1 score is: 0.06460024891839034
#vectorized
#accuracy is: 0.9355077040865154
#f1 score is: 0.5522630026310894 #best vectorized
#TRAIN_PATH = 'preprocess_without_removing_stopwords.csv' 

#TRAIN_PATH = 'preprocess_without_removing_stopwords_new.csv'
#training loss:  0.16406945563400122
#accuracy is: 0.9399942578237152
#f1 score is: 0.09356386977389695
#[[244741  15306]
#[   369    809]]
#testing loss:  0.1704290152697438
#precision:  0.6867572156196944
#recall:  0.05020167545764816

#accuracy is: 0.9401703512297828
#f1 score is: 0.08948441596271482 
#vectorized
#accuracy is: 0.9348760646951861
#f1 score is: 0.5286228872263785
#TRAIN_PATH = 'preprocess_without_stem.csv'

#accuracy is: 0.9406526940377069
#f1 score is: 0.10856189983324707
#vectorized
#accuracy is: 0.9317944300890038
#f1 score is: 0.5517623084857481
#TRAIN_PATH = 'preprocess_just_stem.csv'

#TRAIN_PATH = 'preprocess_just_stem_new.csv'
#training loss:  0.16794367373392913
#accuracy is: 0.9399827734711456
#f1 score is: 0.08519080406115066
#[[244817  15385]
#[   293    730]]
#testing loss:  0.17614428024533574
#precision:  0.7135874877810362
#recall:  0.0452994104871238

#accuracy is: 0.9403234759307111
#f1 score is: 0.09738868623704475
#vectorized
#accuracy is: 0.9326059910039238
#f1 score is: 0.5510417463596257
#TRAIN_PATH = 'preprocess_tokenize_and_stem.csv'

#TRAIN_PATH = 'preprocess_tokenize_and_stem_new.csv'
#training loss:  0.16406945563400122
#accuracy is: 0.9399942578237152
#f1 score is: 0.09356386977389695
#[[244741  15306]
#[   369    809]]
#testing loss:  0.1704290152697438
#precision:  0.6867572156196944
#recall:  0.05020167545764816

#accuracy is: 0.9399253517082975
#f1 score is: 0.08007503370654787
#vectorized
#accuracy is: 0.9353622356206336
#f1 score is: 0.524031007751938
#TRAIN_PATH = 'preprocess_all.csv' 

#TRAIN_PATH = 'preprocess_all_new.csv'
#training loss:  0.16490205440536118
#accuracy is: 0.9398602737104029
#f1 score is: 0.10197782096718873
#[[244623  15223]
#[   487    892]]
#testing loss:  0.17232970043286167
#precision:  0.6468455402465555
#recall:  0.05535215637604716

TEST_PATH = 'test.csv'


def prepare_data(load_test_data=False):

	pd_data = load_data(TRAIN_PATH)
	vector = TfidfVectorizer("english")
	#vector = CountVectorizer()

	feature_matrics = vector.fit_transform(pd_data['question_text'].values.astype('U'))
	print('prepped train')
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
	model = MultinomialNB()
	print("model")
	model.fit(train_data, train_label)
	print("fitting")
	prediction = model.predict(test_feature_matrics)
	print("#training loss: ", log_loss(train_label, model.predict_proba(train_data)))

	if load_test_data:
		del test_label_OR_test_data['question_text']
		# import pdb; pdb.set_trace()
		test_label_OR_test_data.insert(1, 'prediction', prediction)
		test_label_OR_test_data.to_csv('submission.csv', index=False)
		return prediction
	else:
		if if_plt_roc:
			plt_roc(test_label_OR_test_data, prediction)

		print(f'#accuracy is: {accuracy_score(test_label_OR_test_data, prediction)}')
		#print('naive accuracy is: ', 1-np.sum(test_label_OR_test_data)/len(test_label_OR_test_data))
		print(f'#f1 score is: {f1_score(test_label_OR_test_data, prediction)}')
		print(confusion_matrix(test_label_OR_test_data, prediction).T)
		print('#testing loss: ', log_loss(test_label_OR_test_data, model.predict_proba(test_feature_matrics)))
		print('#precision: ', precision_score(test_label_OR_test_data, prediction))
		print('#recall: ', recall_score(test_label_OR_test_data, prediction))

if __name__ == "__main__":
	load_test_data = False
    
	train_data, test_data, train_label, test_label = prepare_data(load_test_data)
	fit_and_predict(load_test_data, train_data, test_data, train_label, test_label, if_plt_roc=False)
