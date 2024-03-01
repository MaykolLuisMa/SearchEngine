from dataset import *

def build_dataset(dataset_name):
    dataset = get_dataset_from_external_files(dataset_name)
    generate_corpus(dataset)

def generate_corpus(dataset):
    dictionary = process_dataset(dataset)
    build_co_occurrence_matrix(dataset,dictionary)
    for doc in dataset:
        doc.to_doc2bow(dictionary)

    document_vectors = calculate_tfidf(dataset)
    save_dates(document_vectors,dictionary)

def calculate_tfidf(dataset):
    corpus = [doc.data for doc in dataset]
    tfidf = gensim.models.TfidfModel(corpus, normalize = True)
    for doc in dataset:
        doc.to_vector(tfidf)
    return {(doc.title, doc.id): {id: frec for id,frec in doc.data} for doc in dataset}

def process_dataset(dataset):
    for doc in dataset:
        doc.process()
    dictionary = build_dictionary(dataset)
    return dictionary
    
def build_dictionary(dataset,no_below=5, no_above=0.5):
    dictionary = gensim.corpora.Dictionary([doc.data for doc in dataset])
    #dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    dictionary.save("./data/program_data/dictionary")
    vocabulary = get_vocabulary(dictionary)
    for doc in dataset:
        doc.filter_by_occurrence(vocabulary)
    return dictionary


def build_co_occurrence_matrix(documents,dictionary, windows_size = 2):
    matrix = {}
    for doc in documents:
        for i,token in enumerate(doc.data):
            start = max(0,i-windows_size)
            end = min(len(doc.data),i+windows_size+1)
            for i2 in range(start,end):
                if i2 == i:
                    continue
                key = tuple(sorted([dictionary.token2id[doc.data[i2]],dictionary.token2id[token]]))
                if key in matrix.keys():
                    matrix[key] += 1/2
                else:
                    matrix[key] = 1/2
    
    with open('./data/co_matrix.json','w') as file:
        json.dump(list(matrix.items()),file)
                 



    