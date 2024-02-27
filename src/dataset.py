import nltk
import spacy
import gensim
import ir_datasets
import os
import json
import nltk

nlp = spacy.load("en_core_web_sm")

class Corpus:
    def __init__(self,document_vectors,word_vectors) -> None:
        self.document_vectors = document_vectors
        self.word_vectors = word_vectors
        self.dictionary = load_dictionary()
        self.coo_matrix = load_co_matrix()
    def vocabulary(self):
        return get_vocabulary(self.dictionary)

class Document:
    def __init__(self,id,title,data):
        self.title = title
        self.data = data
        self.id = id
        self.tags = []
    def process(self):
        self.tokenize()
        self.remuve_noise()
        self.remuve_stopwords()
        self.get_tags()
        self.morphological_reduce()
        
    def tokenize(self):
        self.data = [token for token in nlp(self.data)]

    def remuve_noise(self):
        self.data = [token for token in self.data if token.is_alpha]

    def remuve_stopwords(self):
        stop_words = spacy.lang.en.stop_words.STOP_WORDS
        self.data = [token for token in self.data if token.text not in stop_words]

    def get_tags(self):
        self.tags = [token.pos_ for token in self.data]

    def morphological_reduce(self):
        self.data = [token.lemma_ for token in self.data]

    def filter_by_occurrence(self,vocabulary):
        self.data, _ = filter_by_occurrence(self.data,vocabulary)
        

    def to_doc2bow(self,dictionary):
        self.data = dictionary.doc2bow(self.data)
    
    def to_vector(self,tfidf):
        self.data = tfidf[self.data]

    def add_words(self,words):
        self.data.extend(words)

    def replace_word(self,original,new):
        self.data = [new if word == original else word for word in self.data]

def get_dataset_from_external_files(data = None):
    if data is None:
        content = os.listdir('./data/external_data/')
        return [Document(i,content[i],readfile('./data/external_data/' + content[i])) for i in range(0,len(content))] 
    else:
        dataset = ir_datasets.load(data)
        return [Document(i, doc.title, doc.text) for i,doc in enumerate(dataset.docs_iter())]

def filter_by_occurrence(data,vocabulary):
        new_data = []
        not_founded =[]
        for word in data:
            if word in vocabulary:
                new_data.append(word)
            else:
                not_founded.append(word)
        return new_data,not_founded

def readfile(path):
    f = open(path,'r')
    text = f.read()
    f.close()
    return text  
def load_dictionary():
        return gensim.corpora.Dictionary.load("./data/program_data/dictionary")  

def get_word_vectors(document_vectors,dictionary):
    word_vectors = {i: [word,[]] for i,word in dictionary.iteritems()}
    for doc in document_vectors.keys():
        for word in document_vectors[doc]:
            word_vectors[word[0]][1].append(doc)
    return word_vectors


def save_dates(document_vectors,dictionary):
    with open('./data/document_vectors.json','w') as file1:
        json.dump(document_vectors,file1)
    with open('./data/word_vectors.json','w') as file2:
        json.dump(get_word_vectors(document_vectors,dictionary),file2)

def load_co_matrix():
    with open('./data/co_matrix.json','r') as file:
        matrix = json.load(file)
    return dict(matrix)

def get_corpus():
    with open('./data/document_vectors.json','r') as file1:
        document_vectors = json.load(file1)
    with open('./data/word_vectors.json','r') as file2:
        word_vectors = json.load(file2)
    return Corpus(document_vectors,word_vectors)

def get_vocabulary(dictionary):
  vocabulary = list(dictionary.token2id.keys())
  return vocabulary