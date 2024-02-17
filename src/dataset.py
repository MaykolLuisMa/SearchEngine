#import nltk
import spacy
import gensim
import ir_datasets
import os
import json

language = 'english'
nlp = spacy.load("en_core_web_sm")

class Document:
    def __init__(self,title,data):
        self.title = title
        self.data = data
        
    def process(self):
        self.tokenize()
        self.remuve_noise()
        self.remuve_stopwords()
        self.morphological_reduce()
        
    def tokenize(self):
        self.data = [token for token in nlp(self.data)]
    
    def remuve_noise(self):
        self.data = [token for token in self.data if token.is_alpha]

    def remuve_stopwords(self):
        stop_words = spacy.lang.en.stop_words.STOP_WORDS
        self.data = [token for token in self.data if token.text not in stop_words]
    
    def morphological_reduce(self):
        self.data = [token.lemma_ for token in self.data]

    def filter_by_occurrence(self,vocabulary):
        self.data = [word for word in self.data if word in vocabulary]

    def to_doc2bow(self,dictionary):
        self.data = dictionary.doc2bow(self.data)
    
    def to_vector(self,tfidf):
        self.data = tfidf[self.data]

def get_dataset(data = None):
    if data is None:
        content = os.listdir('data/')
        return [Document(doc,readfile('data/' + doc)) for doc in content] 
    else:
        dataset = ir_datasets.load(data)
        return [Document(doc.title, doc.text) for doc in dataset.docs_iter()[:10]]

def readfile(path):
    f = open(path,'r')
    text = f.read()
    f.close()
    return text    