import spacy
import gensim
import ir_datasets
import os
import json
import nltk

nlp = spacy.load("en_core_web_sm")

class Corpus:
    def __init__(self) -> None:
        self.document_vectors = self.load_document_vectors()
        self.word_vectors = self.load_word_vectors()
        self.dictionary = self.load_dictionary()
        self.coo_matrix = self.load_co_matrix()
    
    def vocabulary(self):
        return get_vocabulary(self.dictionary)

    def load_dictionary(self):
        return gensim.corpora.Dictionary.load("./data/program_data/dictionary")      
    
    def load_co_matrix(self):
        with open('./data/co_matrix.json','r') as file:
            matrix = json.load(file)
        matrix = {tuple(key): value for (key, value) in matrix}
        return dict(matrix)

    def load_document_vectors(self):
        with open('./data/document_vectors.json','r') as file1:
            document_vectors = json.load(file1)
        return {id: Document(id,title,data) for id, title, data in document_vectors}
    
    def load_word_vectors(self):
        with open('./data/word_vectors.json','r') as file2:
            word_vectors = json.load(file2)
        return word_vectors 

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
        self.data,self.tags ,not_founded = filter_by_occurrence(self.data,self.tags,vocabulary)
        return not_founded

    def to_doc2bow(self,dictionary):
        self.data = dictionary.doc2bow(self.data)
    
    def to_vector(self,tfidf):
        self.data = {id: frec for id,frec in tfidf[self.data]}

    def add_words(self,words):
        self.data.extend(words)
        self.tags.extend(["" for word in words])

    def replace_word(self,original,new):
        self.data = [new if word == original else word for word in self.data]

def get_dataset_from_external_files(data = None):
    if data is None:
        content = os.listdir('./data/external_data/')
        return [Document(i,content[i],readfile('./data/external_data/' + content[i])) for i in range(0,len(content))] 
    else:
        dataset = ir_datasets.load(data)
        return [Document(doc.doc_id, doc.title, doc.text) for doc in dataset.docs_iter()]

def filter_by_occurrence(data,tags,vocabulary):
        new_data = []
        new_tags = []
        not_founded =[]
        for i,word in enumerate(data):
            if word in vocabulary:
                new_data.append(word)
                new_tags.append(tags[i])
            else:
                not_founded.append(word)
        return new_data,new_tags ,not_founded

def readfile(path):
    f = open(path,'r')
    text = f.read()
    f.close()
    return text  

def get_word_vectors(document_vectors,dictionary):
    word_vectors = {i: [] for i,word in dictionary.iteritems()}
    for doc in document_vectors:
        for word in doc.data.keys():
            word_vectors[word].append(doc.id)
    return word_vectors


def save_dates(dataset,dictionary):
    document_vectors = [(doc.id,doc.title,doc.data) for doc in dataset]
    with open('./data/document_vectors.json','w') as file1:
        json.dump(document_vectors,file1)
    with open('./data/word_vectors.json','w') as file2:
        json.dump(get_word_vectors(dataset,dictionary),file2)

def get_vocabulary(dictionary):
  vocabulary = list(dictionary.token2id.keys())
  return vocabulary