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
        self.authors = self.load_authors()
    
    def vocabulary(self):
        return get_vocabulary(self.dictionary)

    def load_dictionary(self):
        return gensim.corpora.Dictionary.load("./data/program_data/dictionary")      
    
    def load_co_matrix(self):
        with open('./data/co_matrix.json','r') as file:
            matrix = json.load(file)
        matrix = {tuple(key): value for (key, value) in matrix}
        return dict(matrix)

    def load_authors(self):
        with open('./data/authors.json','r') as file3:
            authors = json.load(file3)
        return authors

    def load_document_vectors(self):
        with open('./data/document_vectors.json','r') as file1:
            document_vectors = json.load(file1)
        return {id: Document(id,title,data,author,bib) for id, title, data, author ,bib in document_vectors}
    
    def load_word_vectors(self):
        with open('./data/word_vectors.json','r') as file2:
            word_vectors = json.load(file2)
        return {id:Word(id,word,in_documents,idf) for id,word,in_documents,idf in word_vectors}
        

class Word:
    def __init__(self,id,word,in_documents,idf):
        self.id = id
        self.word = word
        self.in_documents = in_documents
        self.idf = idf

class Document:
    def __init__(self,id,title,data,author = "User",bib = "None"):
        self.title = title
        self.data = data
        self.id = id
        self.author = author
        self.bib = bib
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
        self.data,self.tags,not_founded = filter_by_occurrence(self.data,self.tags,vocabulary)
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

    def find_authors(self):
        authors = [author.text for author in nlp(self.authors)]
        if "and" not in authors:
            return
        self.author = authors_parser(authors)
def authors_parser(authors_tokens):
    authors = []
    author = ""
    
    i = 0
    while i <  len(authors_tokens):
        if authors_tokens[i] == "and":
            authors.append(author)
            author = ""
            for j in range(i+1,len(authors_tokens)):
                author += authors_tokens[j]
            authors.append(author)
            break
        if is_word(authors_tokens[i]):
            if(i-1>=0):
                if((authors_tokens[i-1] == ',') | (authors_tokens[i-1][len(authors_tokens[i-1])-1] == ',')):
                    author = author[:-1]
                    authors.append(author)
                    author = ""
                elif ((authors_tokens[i-1] != '-') & (authors_tokens[i-1][len(authors_tokens[i-1])-1] != '-')):
                    author += " "
        author += authors_tokens[i]
        i += 1
    return authors

def is_word(token):
    if (',' in token) | ('.' in token):
        return False
    return len(token)>=2

def get_dataset_from_external_files(data = None):
    if data is None:
        content = os.listdir('./data/external_data/')
        return [Document(i,content[i],readfile('./data/external_data/' + content[i])) for i in range(0,len(content))] 
    else:
        dataset = ir_datasets.load(data)
        return [Document(doc.doc_id, doc.title, doc.text, doc.author, doc.bib) for doc in dataset.docs_iter()]

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

def get_word_vectors(documents,idfs,dictionary):
    word_vectors = {id:Word(id,word,[],0) for id,word in list(dictionary.items())}
    for doc in documents:
        for id in doc.data.keys():
            word_vectors[id].in_documents.append(doc.id)
            word_vectors[id].idf = idfs[id]
    print("words")
    for i,word in enumerate(word_vectors.values()):
        print(word.id,word.word)
        if i == 10:
            break
    return word_vectors


def build_authors_db(dataset):
    authors = [(doc.author) for doc in dataset]
    all_authors = []
    for items in authors:
        for item in items:
            if item not in all_authors:
                all_authors.append(item)
    return all_authors

def save_dates(documents,idfs,dictionary):
    document_vectors = [(doc.id,doc.title,doc.data,doc.author,doc.bib) for doc in documents]
    authors = build_authors_db(documents)
    word_vectors = [(w.id,w.word,w.in_documents,w.idf) for w in get_word_vectors(documents,idfs,dictionary).values()]
    with open('./data/authors.json','w') as file3:
        json.dump({author:0 for author in authors},file3)
    with open('./data/document_vectors.json','w') as file1:
        json.dump(document_vectors,file1)
    with open('./data/word_vectors.json','w') as file2:
        json.dump(word_vectors,file2)

def get_vocabulary(dictionary):
  vocabulary = list(dictionary.token2id.keys())
  return vocabulary