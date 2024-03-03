from lexer import *

def process_query(query,dictionary):
    proceced_query = morphological_reduction_spacy(remove_stopwords_spacy(remove_noise_spacy(tokenization_spacy([(0,query)]))), True)
    proceced_query = filter_tokens_by_occurrence(proceced_query,dictionary)
    return [id for (id,sec) in vector_representation(proceced_query, dictionary, [],True)[0][1]]

def search(query,corpus):
    docs = {id:[doc,True] for id, doc in corpus}
    if len(query) == 0:
        return []
    for id in query:
        for key in docs.keys():
            if id not in [k for k,_ in docs[key][0]]:
                docs[key][1] = False
    return [k for k,(l,b) in list(docs.items()) if b == True]
