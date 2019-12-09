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
from utils import timer


tqdm.pandas()

# Always run with train_small.csv to how outputs look like when modifying data pre-processing.
#TRAIN_PATH = 'train_small.csv'
TRAIN_PATH = 'train_shuffle1.csv'

# https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/74518#latest-456642
puncts=['☹', 'Ź', 'Ż', 'ἰ', 'ή', 'Š', '＞', 'ξ','ฉ', 'ั', 'น', 'จ', 'ะ', 'ท', 'ำ', 'ใ', 'ห', '้', 'ด', 'ี', 
'่', 'ส', 'ุ', 'Π', 'प', 'ऊ', 'Ö', 'خ', 'ب', 'ஜ', 'ோ', 'ட', '', '「', 'ẽ', '½', '△', 'É', 'ķ', 'ï', '¿', 
'ł', '북', '한', '¼', '∆', '≥', '⇒', '¬', '∨', 'č', 'š', '∫', 'ḥ', 'ā', 'ī', 'Ñ', 'à', '▾', 'Ω', '＾', 'ý', 
'µ', '?', '!', '.', ',', '"', '#', '$', '%', '\\', "'", '(', ')', '*', '+', '-', '/', ':', ';', '<', '=', 
'>', '@', '[', ']', '^', '_', '`', '{', '|', '}', '~', ' ', '“', '”', '’', 'é', 'á', '′', '…', 'ɾ', '̃', 
'ɖ', 'ö', '–', '‘', 'ऋ', 'ॠ', 'ऌ', 'ॡ', 'ò', 'è', 'ù', 'â', 'ğ', 'म', 'ि', 'ल', 'ग', 'ई', 'क', 'े', 'ज', 
'ो', 'ठ', 'ं', 'ड', 'Ž', 'ž', 'ó', '®', 'ê', 'ạ', 'ệ', '°', 'ص', 'و', 'ر', 'ü', '²', '₹', 'ú', '√', 'α', 
'→', 'ū', '—', '£', 'ä', '️', 'ø', '´', '×', 'í', 'ō', 'π', '÷', 'ʿ', '€', 'ñ', 'ç', 'へ', 'の', 'と', 'も',
'↑', '∞', 'ʻ', '℅''ι', '•', 'ì', '−', 'л', 'я', 'д', 'ل', 'ك', 'م', 'ق', 'ا', '∈', '∩', '⊆', 'ã', 'अ', 'न',
'ु', 'स', '्', 'व', 'ा', 'र', 'त', '§', '℃', 'θ', '±', '≤', 'उ', 'द', 'य', 'ब', 'ट', '͡', '͜', 'ʖ', '⁴', 
'™', 'ć', 'ô', 'с', 'п', 'и', 'б', 'о', 'г', '≠', '∂', 'आ', 'ह', 'भ', 'ी', '³', 'च', '...', '⌚', '⟨', '⟩',
'∖', '˂', 'ⁿ', '⅔', 'న', 'ీ', 'క', 'ె', 'ం', 'ద', 'ు', 'ా', 'గ', 'ర', 'ి', 'చ', 'র', 'ড়', 'ঢ়', 'સ',
'ં', 'ઘ', 'ર', 'ા', 'જ', '્', 'ય', 'ε', 'ν', 'τ', 'σ', 'ş', 'ś', 'س', 'ت', 'ط', 'ي', 'ع', 'ة', 'د', 'Å',
'☺', 'ℇ', '❤', '♨', '✌', 'ﬁ', 'て', '„', 'Ā', 'ត', 'ើ', 'ប', 'ង', '្', 'អ', 'ូ', 'ន', 'ម', 'ា', 'ធ', 'យ',
'វ', 'ី', 'ខ', 'ល', 'ះ', 'ដ', 'រ', 'ក', 'ឃ', 'ញ', 'ឯ', 'ស', 'ំ', 'ព', 'ិ', 'ៃ', 'ទ', 'គ', '¢', 'つ', 'や', 
'ค', 'ณ', 'ก', 'ล', 'ง', 'อ', 'ไ', 'ร', 'į', 'ی', 'ю', 'ʌ', 'ʊ', 'י', 'ה', 'ו', 'ד', 'ת', 'ᠠ', 'ᡳ', 'ᠰ', 
'ᠨ', 'ᡤ', 'ᡠ', 'ᡵ', 'ṭ', 'ế', 'ध', 'ड़', 'ß', '¸', 'ч',  'ễ', 'ộ', 'फ', 'μ', '⧼', '⧽', 'ম', 'হ', 'া', 'ব', 
'ি', 'শ', '্', 'প', 'ত', 'ন', 'য়', 'স', 'চ', 'ছ', 'ে', 'ষ', 'য', '়', 'ট', 'উ', 'থ', 'ক', 'ῥ', 'ζ', 'ὤ', 'Ü', 
'Δ', '내', '제', 'ʃ', 'ɸ', 'ợ', 'ĺ', 'º', 'ष', '♭', '़', '✅', '✓', 'ě', '∘', '¨', '″', 'İ', '⃗', '̂', 'æ', 'ɔ', 
'∑', '¾', 'Я', 'х', 'О', 'з', 'ف', 'ن', 'ḵ', 'Č', 'П', 'ь', 'В', 'Φ', 'ỵ', 'ɦ', 'ʏ', 'ɨ', 'ɛ', 'ʀ', 'ċ', 'օ', 
'ʍ', 'ռ', 'ք', 'ʋ', '兰', 'ϵ', 'δ', 'Ľ', 'ɒ', 'î', 'Ἀ', 'χ', 'ῆ', 'ύ', 'ኤ', 'ል', 'ሮ', 'ኢ', 'የ', 'ኝ', 'ን', 
'አ', 'ሁ', '≅', 'ϕ', '‑', 'ả', '￼', 'ֿ', 'か', 'く', 'れ', 'ő', '－', 'ș', 'ן', 'Γ', '∪', 'φ', 'ψ', '⊨', 'β', '∠', 
'Ó', '«', '»', 'Í', 'க', 'வ', 'ா', 'ம', '≈', '⁰', '⁷', 'ấ', 'ũ', '눈', '치', 'ụ', 'å', '،', '＝', '（', '）', 
'ə', 'ਨ', 'ਾ', 'ਮ', 'ੁ', '︠', '︡', 'ɑ', 'ː', 'λ', '∧', '∀', 'Ō', 'ㅜ', 'Ο', 'ς', 'ο', 'η', 'Σ', 'ण', '大','能', 
'化', '生', '水', '谷', '精', '微', 'ル', 'ー', 'ジ', 'ュ', '支', '那', '¹', 'マ', 'リ', '仲', '直', 'り', 'し', 'た', 
'主', '席', '血', '⅓', '漢', '髪', '金', '茶', '訓', '読', '黒', 'ř', 'あ', 'わ', 'る', '胡', '南', '수', '능', '广', 
'电', '总', 'ί', '서', '로', '가', '를', '행', '복', '하', '게', '기', '乡', '故', '爾', '汝', '言', '得', '理', '让', 
'骂', '野', '比', 'び', '太', '後', '宮', '甄', '嬛', '傳', '做', '莫', '你', '酱', '紫', '甲', '骨', '陳', '宗', '陈', 
'什', '么', '说', '伊', '藤', '長', 'ﷺ', '僕', 'だ', 'け', 'が', '街', '◦', '火', '团', '表',  '看', '他', '顺', '眼', 
'中', '華', '民', '國', '許', '自', '東', '儿', '臣', '惶', '恐', 'っ', '木', 'ホ', 'ج', '教', '官', '국', '고', '등', 
'학', '교', '는', '몇', '시', '간', '업', '니', '本', '語', '上', '手', 'で', 'ね', '台', '湾', '最', '美', '风', '景', 
'Î', '≡', '皎', '滢', '杨', '∛', '簡', '訊', '短', '送', '發', 'お', '早', 'う', '朝', 'ش', 'ه', '饭', '乱', '吃', 
'话', '讲', '男', '女', '授', '受', '亲', '好', '心', '没', '报', '攻', '克', '禮', '儀', '統', '已', '經', '失', '存', 
'٨', '八', '‛', '字', '：', '别', '高', '兴', '还', '几', '个', '条', '件', '呢', '觀', '《', '》', '記', '宋', '楚', 
'瑜', '孫', '瀛', '枚', '无', '挑', '剔', '聖', '部', '頭', '合', '約', 'ρ', '油', '腻', '邋', '遢', 'ٌ', 'Ä', '射', '籍', 
'贯', '老', '常', '谈', '族', '伟', '复', '平', '天', '下', '悠', '堵', '阻', '愛', '过', '会', '俄', '罗', '斯', '茹', 
'西', '亚', '싱', '관', '없', '어', '나', '이', '키', '夢', '彩', '蛋', '鰹', '節', '狐', '狸', '鳳', '凰', '露', '王', 
'晓', '菲', '恋', 'に', '落', 'ち', 'ら', 'よ', '悲', '反', '清', '復', '明', '肉', '希', '望', '沒', '公', '病', '配', 
'信', '開', '始', '日', '商', '品', '発', '売', '分', '子', '创', '意', '梦', '工', '坊', 'ک', 'پ', 'ڤ', '蘭', '花', '羡', 
'慕', '和', '嫉', '妒', '是', '样', 'ご', 'め', 'な', 'さ', 'い', 'す', 'み', 'ま', 'せ', 'ん', '音', '红', '宝', '书', 
'封', '柏', '荣', '江', '青', '鸡', '汤', '文', '粵', '拼', '寧', '可', '錯', '殺', '千', '絕', '放', '過', '」', '之', 
'勢', '请', '国', '知', '识', '产', '权', '局', '標', '點', '符', '號', '新', '年', '快', '乐', '学', '业', '进', '步', '身', 
'体', '健', '康', '们', '读', '我', '的', '翻', '译', '篇', '章', '欢', '迎', '入', '坑', '有', '毒', '黎', '氏', '玉', '英', 
'啧', '您', '这', '口', '味', '奇', '特', '也', '就', '罢', '了', '非', '要', '以', '此', '为', '依', '据', '对', '人', '家', 
'批', '判', '一', '番', '不', '地', '道', '啊', '谢', '六', '佬']


def load_data(file_path):
	return pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path))

# data = load_data(TRAIN_PATH)
# print(data.shape)


def remove_punctuations(text):
	return ' '.join([t for t in text.split(' ') if t not in puncts])

# print(remove_punctuations('ab#//#adfaacd# adf #/a'))


def init_tokenizer():
	return RegexpTokenizer(r'\w+')

cached_stopwords = stopwords.words('english')

def remove_stopwords(words):
	"""
		Make sure to run:
			```
			import nltk
			nltk.download()
			```
		in the virtual venv.
	"""
	return ' '.join([word for word in words if word not in cached_stopwords])

# print(remove_stopwords('I see you a monster'))


def init_stemmer():
	return SnowballStemmer('english')

# print(init_stemmer().stem('I am fishing?'))

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
		stemmer = init_stemmer()
		pd_data['question_text'] = pd_data['question_text'].apply(lambda x: ' '.join([stemmer.stem(w) for w in x.split(' ')]))
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


#pd_data = load_and_clean_data(output_file='preprocess_without_punctuation_new_{}.csv', opt_punctuation=False)
#pd_data = load_and_clean_data(output_file='preprocess_without_tokenize_new_{}.csv', opt_tokenize=False)
#pd_data = load_and_clean_data(output_file='preprocess_without_removing_stopwords_new_{}.csv', opt_remove_stopwords=False)
#pd_data = load_and_clean_data(output_file='preprocess_without_stem_{}.csv', opt_stemming=False)
#pd_data = load_and_clean_data(output_file='preprocess_just_stem_new_{}.csv', opt_punctuation=False, opt_tokenize=False, opt_remove_stopwords=False)
#pd_data = load_and_clean_data(output_file='preprocess_tokenize_and_stem_new_{}.csv', opt_punctuation=False, opt_remove_stopwords=False)
#pd_data = load_and_clean_data(output_file='preprocess_all_new_{}')

#print("just punctuation")
#load_and_clean_data(output_file='preprocess_only_punctuation_{}.csv', opt_punctuation=True, opt_tokenize=False, opt_remove_stopwords=False, opt_stemming=False)
#print("just tokenize")
#load_and_clean_data(output_file='preprocess_only_tokenize_{}.csv', opt_punctuation=False, opt_tokenize=True, opt_remove_stopwords=False, opt_stemming=False)
#print("just remove stop")
#load_and_clean_data(output_file='preprocess_only_stop_{}.csv', opt_punctuation=False, opt_tokenize=False, opt_remove_stopwords=True, opt_stemming=False)
#print("just stem")
#load_and_clean_data(output_file='preprocess_only_stem_{}.csv', opt_punctuation=False, opt_tokenize=False, opt_remove_stopwords=False, opt_stemming=True)

# pd_data = load_and_clean_data(opt_remove_stopwords=False, opt_stemming=False)