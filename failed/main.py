
import nltk
import gensim
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
#data1['content'] = data1['content'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

def remove_stopword(text):
    textarr = text.split(' ')
    rem_text = " ".join([i for i in textarr if i not in stopwords])
    return rem_text

data1['content'] = data1['content'].apply(remove_stopword)

data2 = data1.copy()

column = data2.columns
data2 = data2.drop(columns=['Unnamed: 0', 'reviewId', 'userName', 'userImage', 'score',
       'thumbsUpCount', 'reviewCreatedVersion', 'at', 'replyContent',
       'repliedAt'],axis=0)



def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]


data2['lemma'] = data1.content.apply(lemmatize_text)
lemma = data2.copy()
lemma = data2['lemma'].drop(columns=['content'],axis=0)

print(lemma[0])

id2word = corpora.Dictionary(lemma)

corpus = []
for text in lemma:
    new = id2word.doc2bow(text)
    corpus.append(new)


lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=20,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto')

py = pyLDAvis.gensim_models.prepare(lda_model,corpus,id2word,mds='mmds',R=30)

#extract = pyLDAvis.save_html(py,"done.html")
