# System import
import os

# External import
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

# Internal import
from data_pre_processing import load_data
from utils import plt_roc

# accuracy is: 0.9507168150062207
# f1 score is: 0.4766241157817708
#Best regularization factor:  30
#With 80% train:
#accuracy is: 0.9513446262800268
#f1 score is: 0.5245754469963343
#With 60% train:
#accuracy is: 0.9503072064312375
#f1 score is: 0.5141295804169629
#TRAIN_PATH = 'preprocess_without_punctuation.csv'  

# accuracy is: 0.9521179060197148
# f1 score is: 0.5007982120051084
#Best regularization factor:  30
#With 80% train:
#accuracy is: 0.9519226720260312
#f1 score is: 0.5323403463042264
#With 60% train:
#Accuracy is: 0.9511149392286343
#f1 score is: 0.522866537139441
#TRAIN_PATH = 'preprocess_without_tokenize.csv'

# accuracy is: 0.9526232175327782
# f1 score is: 0.511640754478731
# best reg: 30
# With 80% train:
# accuracy is: 0.9528146234089386
# f1 score is: 0.5450317436881736
# With 60% train:
# accuracy is: 0.9519341563786008
# f1 score is: 0.534962962962963
#TRAIN_PATH = 'preprocess_without_removing_stopwords.csv' 

# accuracy is: 0.9503684563116087
# f1 score is: 0.4719156042523726
# best reg: 30
#With 80% train:
#accuracy is: 0.951237438989377
#f1 score is: 0.5217391304347825
#With 60% train:
#accuracy is: 0.9502918939611447
#f1 score is: 0.5116401519425327
#TRAIN_PATH = 'preprocess_without_stem.csv'  

# accuracy is: 0.9528184515264618
# f1 score is: 0.5173669577475819
# best reg: 30
#With 80% train:
#accuracy is: 0.9536032156187195
#f1 score is: 0.556693489392831
#With 60% train:
#accuracy is: 0.952783998468753
#f1 score is: 0.5479401847236476
#TRAIN_PATH = 'preprocess_just_stem.csv'

# accuracy is: 0.9530022011675758
# f1 score: 0.5177736753211045
# kaggle test f1 score: 0.53762
# best reg: 30
#With 80% train:
#accuracy is: 0.9531553258685042
#f1 score is: 0.5508533675903835
#With 60% train:
#accuracy is: 0.9523093118958752
#f1 score is: 0.5420863044916562
#TRAIN_PATH = 'preprocess_tokenize_and_stem.csv'

# accuracy is: 0.9504986123073978
# f1 score is: 0.4712328767123288
# best reg: 30
#With 80% train:
#accuracy is: 0.951072829935879
#f1 score is: 0.5183705769303237
#With 60% train:
#accuracy is: 0.9500698631447986
#f1 score is: 0.5083864158908447
#TRAIN_PATH = 'preprocess_all.csv' 

#TRAIN_PATH = 'preprocess_all_new.csv'
#Best regularization factor:  30
#With 80% train:
#accuracy is: 0.9511991578141449
#f1 score is: 0.5190885770333484
#confusion_matrix score is: [[241597   3513]
#[  9235   6880]]
#With 60% train:
#accuracy is: 0.9506708775959422
#f1 score is: 0.5130007558578987
#confusion_matrix score is: [[241552   3558]
#[  9328   6787]]

#TRAIN_PATH = 'preprocess_just_stem_new.csv'
#Kaggle Test score: 0.54742
#Best regularization factor:  30
#With 80% train:
#accuracy is: 0.9528988419944492
#f1 score is: 0.5461118489006935
#confusion_matrix score is: [[241519   3591]
#[  8713   7402]]
#With 60% train:
#accuracy is: 0.9524050148339555
#f1 score is: 0.5402847106674062
#confusion_matrix score is: [[241486   3624]
#[  8809   7306]]

#TRAIN_PATH = 'preprocess_tokenize_and_stem_new.csv'
#Kaggle test score: 0.54874
#Best regularization factor:  30
#With 80% train:
#accuracy is: 0.9537027466743229
#f1 score is: 0.5548111610100861
#confusion_matrix score is: [[241595   3515]
#[  8579   7536]]
#With 60% train:
#accuracy is: 0.9529332950521581
#f1 score is: 0.5458910433979686
#confusion_matrix score is: [[241540   3570]
#[  8725   7390]]

#TRAIN_PATH = 'preprocess_without_punctuation_new.csv'
#Best regularization factor:  30
#With 80% train:
#accuracy is: 0.9511991578141449
#f1 score is: 0.5190885770333484
#confusion_matrix score is: [[241597   3513]
#[  9235   6880]]
#With 60% train:
#accuracy is: 0.9506708775959422
#f1 score is: 0.5130007558578987
#confusion_matrix score is: [[241552   3558]
#[  9328   6787]]

#TRAIN_PATH = 'preprocess_without_removing_stopwords_new.csv'
#Best regularization factor:  30
#With 80% train:
#accuracy is: 0.9537027466743229
#f1 score is: 0.5548111610100861
#confusion_matrix score is: [[241595   3515]
#[  8579   7536]]
#With 60% train:
#accuracy is: 0.9529332950521581
#f1 score is: 0.5458910433979686
#confusion_matrix score is: [[241540   3570]
#[  8725   7390]]

#TRAIN_PATH = 'preprocess_without_tokenize_new.csv'
#Best regularization factor:  30
#With 80% train:
#accuracy is: 0.9520681404919131
#f1 score is: 0.5341369944562264
#confusion_matrix score is: [[241526   3584]
#[  8937   7178]]
#With 60% train:
#accuracy is: 0.9514709541582926
#f1 score is: 0.5280869597587761
#confusion_matrix score is: [[241455   3655]
#[  9022   7093]]

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
	C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
	best_C = 1
	best_f1 = 0
	train_x, valid_x, train_y, valid_y = train_test_split(train_data, train_label, test_size=0.25, shuffle=False)
	if not load_test_data:
		for c in C:
			model = LogisticRegression(solver='liblinear', penalty='l2', C=c)
			model.fit(train_x, train_y)
			prediction = model.predict(valid_x)
			f1 = f1_score(valid_y, prediction)
			print("Regularization: ", c)
			print("F1 score: ",f1)
			if f1>best_f1:
				best_f1=f1
				best_C = c
	print("#Best regularization factor: ", best_C)
	model = LogisticRegression(solver='liblinear', penalty='l2', C=best_C)
	model.fit(train_data, train_label)
	prediction80 = model.predict(test_feature_matrics)

	model = LogisticRegression(solver='liblinear', penalty='l2', C=best_C)
	model.fit(train_x, train_y)
	prediction60 = model.predict(test_feature_matrics)
	if load_test_data:
		del test_label_OR_test_data['question_text']
		# import pdb; pdb.set_trace()
		test_label_OR_test_data.insert(1, 'prediction', prediction80)
		test_label_OR_test_data.to_csv('submission.csv', index=False)
		return prediction
	else:
		if if_plt_roc:
			plt_roc(test_label_OR_test_data, prediction)

		print("#With 80% train:")
		print(f'#accuracy is: {accuracy_score(test_label_OR_test_data, prediction80)}')
		print(f'#f1 score is: {f1_score(test_label_OR_test_data, prediction80)}')
		print(f'#confusion_matrix score is: {confusion_matrix(test_label_OR_test_data, prediction80)}')
		print("#With 60% train:")
		print(f'#accuracy is: {accuracy_score(test_label_OR_test_data, prediction60)}')
		print(f'#f1 score is: {f1_score(test_label_OR_test_data, prediction60)}')
		print(f'#confusion_matrix score is: {confusion_matrix(test_label_OR_test_data, prediction60)}')


if __name__ == "__main__":
	load_test_data = False
	print("hello there")
	train_data, test_data, train_label, test_label = prepare_data(load_test_data)
	fit_and_predict(load_test_data, train_data, test_data, train_label, test_label, if_plt_roc=True)

