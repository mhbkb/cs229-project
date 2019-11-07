# System import
import string
from collections import defaultdict

# External import
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm


tqdm.pandas

TRAIN_PATH = '/Users/hmao/Documents/cs229-project/train.csv'


def load_data(file_path):
	return pd.read_csv(file_path)

data = load_data(TRAIN_PATH)
print(data.shape)


def remove_punctuations(text):
	return ''.join([t for t in text if t not in string.punctuation])

# print(remove_punctuations('ab#//#adfaacd# adf #/a'))


def init_tokenizer():
	return RegexpTokenizer(r'\w+')


def remove_stopwords(text):
    """
		Make sure to run:
			```
			import nltk
			nltk.download()
			```
		in the virtual venv.
    """
    return [t for t in text if t not in stopwords.words('english')]

# print(remove_stopwords(['I', 'see', 'you', 'a', 'monster']))

def clean_data(pd_data):
    pd_data['question_text'] = pd_data['question_text'].apply(lambda x: remove_punctuations(x))
    pd_data['question_text'] = pd_data['question_text'].apply(lambda x: init_tokenizer().tokenize(x.lower()))
    pd_data['question_text'] = pd_data['question_text'].apply(lambda x: remove_stopwords(x))

    print(pd_data['question_text'].head(15))


def load_and_clean_data(data_path=TRAIN_PATH):
	pd_data = load_data(data_path)
	clean_data(pd_data)
	return pd_data


def build_word_dict(pd_data):
	word_dict = defaultdict(int)
	pd_data = load_and_clean_data()

	for word in tqdm(pd_data['question_text'].values):
		word_dict[word] += 1

	return word_dict


# pd_data = load_and_clean_data()