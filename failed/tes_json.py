import pandas as pd
import spacy

encoding = 'unicode_escape'
list_data = pd.read_csv("tweet.csv", encoding= 'unicode_escape', sep='delimiter', header=None,engine='python')
list_data['teks'] = list_data
data1 = list_data.drop(labels=0,axis=1)
data1 = data1.to_json('/home/levi/PycharmProjects/file.json')



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

data['clean'] = data['check'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

data['lemma'] = data.clean.apply(lemmatize_text)

lemma = data['lemma']


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

extract = pyLDAvis.save_html(py,"done.html")



def lemmatization(texts,allowed_postags=['NOUN','ADJ']):
    nlp = spacy.load("xx_ent_wiki_sm", disable=['parser', 'ner'])
    texts_out = []
    for text in texts:
        doc = nlp(text)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_postags:
                new_text.append(token)

textlist = data1['content'].tolist()
tokenizex = lemmatization(textlist)
def lemmatization(texts,allowed_postags=['NOUN','ADJ']):
    nlp = spacy.load("xx_ent_wiki_sm", disable=['parser', 'ner'])
    texts_out = []
    for text in texts:
        doc = nlp(text)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_postags:
                new_text.append(token.lemma_)
        final = " ".join(new_text)
        texts_out.append(final)
    return (texts_out)


lemmatized = lemmatization(data2)
