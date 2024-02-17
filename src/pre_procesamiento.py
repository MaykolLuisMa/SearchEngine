from dataset import *
    
def generate_corpus(dataset):
    dictionary = process_dataset(dataset)
    for doc in dataset:
        doc.to_doc2bow(dictionary)
    
    model = calculate_tfidf(dataset)
    save_dates(model,dictionary)

def calculate_tfidf(dataset):
    corpus = [doc.data for doc in dataset]
    tfidf = gensim.models.TfidfModel(corpus, normalize = True)
    for doc in dataset:
        doc.to_vector(tfidf)
    return {doc.title: doc.data for doc in dataset}

def process_dataset(dataset):
    for doc in dataset:
        doc.process()
    dictionary = filter_by_occurrence(dataset)
    return dictionary
    
def filter_by_occurrence(dataset,no_below=5, no_above=0.5):
    dictionary = gensim.corpora.Dictionary([doc.data for doc in dataset])
    #dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    vocabulary = [word for _, word in dictionary.iteritems()]
    for doc in dataset:
        doc.filter_by_occurrence(vocabulary)
    return dictionary

def save_dates(model,dictionary):
    with open('./data/model.json','w') as file:
        json.dump(model,file)
    with open('./data/dictionary.json','w') as file:
        json.dump({x: y for x,y in dictionary.iteritems()},file)
    
def get_corpus():
    with open('./data/model.json','r') as file:
        model = json.load(file)
    with open('./data/dictionary.json','r') as file:
        dictionary = json.load(file)
    return model, dictionary
    