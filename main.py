import tes_json
import nltk
import pyLDAvis
import pyLDAvis.gensim_models
import spacy
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

model = spacy.load('xx_ent_wiki_sm', disable=['parser', 'ner'])
from nltk.corpus import stopwords
import pandas as pd

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import tes_json

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

stopwords = stopwords.words("indonesian")


def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = tes_json.load(f)
    return data


def write_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        tes_json.dump(data, f, indent=4)


#data = load_data("file.json")

encoding = 'unicode_escape'
data = pd.read_csv("tweet.csv", encoding= 'unicode_escape', sep='delimiter', header=None)
data = data.rename(columns={0: 'teks'})

data['cols_to_check'] = data['teks'].replace({
                                              '"':'',
                                              '\d+':'',
                                              ':':'',
                                              ';':'',
                                              '#':'',
                                              '@':'',
                                              '_':'',
                                                ',': '',
                                                "'": '',
                                              }, regex=True)
data['check'] = data['cols_to_check'].str.replace(r'[https]+[?://]+[^\s<>"]+|www\.[^\s<>"]+[@?()]+[(??)]+[)*]+[(\xa0]+[-&gt...]', "",regex=True)

https = "https"
clean = []
def cleaner():
    for x in data['cols_to_check'].iteritems():
        x.str.find("https")
        print(x)

cleaner()
check = print(data.check[1])
'''def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

data['lemma'] = data.check.apply(lemmatize_text)'''




