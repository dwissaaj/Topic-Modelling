
import nltk
nltk.download('wordnet')
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pyLDAvis
from nltk.corpus import stopwords
import pandas as pd
import warnings
import pyLDAvis.gensim_models
import spacy
import openpyxl

model = spacy.load("xx_ent_wiki_sm", disable=['parser', 'ner'])
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

warnings.filterwarnings("ignore", category=DeprecationWarning)
stopwords = stopwords.words("indonesian")

data = pd.read_excel('bersih.xlsx')

data1 = data.copy()
data1 = data1.drop_duplicates(subset='content')

data1['content'] = data1['content'].str.lstrip()
data1['content'] = data1['content'].str.lower()

data1['content'] = data1['content'].apply(str)
data1['content'] = data1['content'].fillna("pos")

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

data2 = data1.copy()
data2['lemma'] = data1.content.apply(lemmatize_text)
