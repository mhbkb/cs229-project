# System import
import os
import time
import string
from collections import defaultdict

# External import
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm


tqdm.pandas()

# Always run with train_small.csv to how outputs look like when modifying data pre-processing.
TRAIN_PATH = 'train_small.csv'
#TRAIN_PATH = '../train_shuffle1.csv'


def load_data(file_path):
	return pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path))

# data = load_data(TRAIN_PATH)
# print(data.shape)


def remove_punctuations(text):
	return ''.join([t for t in text if t not in string.punctuation])

# print(remove_punctuations('ab#//#adfaacd# adf #/a'))


def init_tokenizer():
	return RegexpTokenizer(r'\w+')


def remove_stopwords(words):
	"""
		Make sure to run:
			```
			import nltk
			nltk.download()
			```
		in the virtual venv.
	"""
	return ' '.join([word for word in words if word not in stopwords.words('english')])

# print(remove_stopwords('I see you a monster'))


def init_stemmer():
	return SnowballStemmer('english')

#print(init_stemmer().stem('I am fishing'))


def clean_data(pd_data, opt_punctuation=True, opt_tokenize=True, opt_remove_stopwords=True, opt_stemming=True):
	if opt_punctuation:
		print('Running remove_punctuations...')
		pd_data['question_text'] = pd_data['question_text'].apply(lambda x: remove_punctuations(x))
		print(pd_data['question_text'].head(15))

	if opt_tokenize:
		print('Running tokenize...')
		pd_data['question_text'] = pd_data['question_text'].apply(lambda x: ' '.join(init_tokenizer().tokenize(x.lower())))
		print(pd_data['question_text'].head(15))

	if opt_remove_stopwords:
		# Removing stopwords takes about 30 mins in Hao's machine.
		print('Removing stopwords...')
		pd_data['question_text'] = pd_data['question_text'].apply(lambda x: remove_stopwords(x.split(' ')))
		print(pd_data['question_text'].head(15))

	if opt_stemming:
		print('Stemming...')
		pd_data['question_text'] = pd_data['question_text'].apply(lambda x: init_stemmer().stem(x))
		print(pd_data['question_text'].head(15))

	#print(pd_data['question_text'].head(15))


def load_and_clean_data(data_path=TRAIN_PATH, 
						output_file='preprocess_{}.csv', 
						opt_punctuation=True, 
						opt_tokenize=True,
						opt_remove_stopwords=True,
						opt_stemming=True):
	pd_data = load_data(data_path)
	clean_data(pd_data, opt_punctuation, opt_tokenize, opt_remove_stopwords, opt_stemming)

	# Since it may take some time to preprocess data, so let's save the pd_data to a file.
	if output_file:
		pd_data.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 
									output_file.format(time.strftime("%Y%m%d-%H%M%S"))), 
					   index=False)

	return pd_data


# pd_data = load_and_clean_data(output_file='preprocess_without_punctuation_{}.csv', opt_punctuation=False)
# pd_data = load_and_clean_data(output_file='preprocess_without_tokenize_{}.csv', opt_tokenize=False)
# pd_data = load_and_clean_data(output_file='preprocess_without_removing_stopwords_{}.csv', opt_remove_stopwords=False)
# pd_data = load_and_clean_data(output_file='preprocess_without_stem_{}.csv', opt_stemming=False)
# pd_data = load_and_clean_data(output_file='preprocess_just_stem_{}.csv', opt_punctuation=False, opt_tokenize=False, opt_remove_stopwords=False)
# pd_data = load_and_clean_data(output_file='preprocess_tokenize_and_stem_{}.csv', opt_punctuation=False, opt_remove_stopwords=False)
# pd_data = load_and_clean_data()

#print("just punctuation")
#load_and_clean_data(output_file='preprocess_only_punctuation_{}.csv', opt_punctuation=True, opt_tokenize=False, opt_remove_stopwords=False, opt_stemming=False)
#print("just tokenize")
#load_and_clean_data(output_file='preprocess_only_tokenize_{}.csv', opt_punctuation=False, opt_tokenize=True, opt_remove_stopwords=False, opt_stemming=False)
#print("just remove stop")
#load_and_clean_data(output_file='preprocess_only_stop_{}.csv', opt_punctuation=False, opt_tokenize=False, opt_remove_stopwords=True, opt_stemming=False)
#print("just stem")
#load_and_clean_data(output_file='preprocess_only_stem_{}.csv', opt_punctuation=False, opt_tokenize=False, opt_remove_stopwords=False, opt_stemming=True)



# def build_word_dict(pd_data):
# 	word_dict = defaultdict(int)
# 	pd_data = load_and_clean_data()
	
# 	for word in tqdm(pd_data['question_text'].values, disable=False):
# 		word_dict[word] += 1
		
# 	return word_dict
