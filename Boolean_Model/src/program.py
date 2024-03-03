from model import *

def ejecute():
    documents = load_dataset("cranfield")

    tokenized_docs = morphological_reduction_spacy(remove_stopwords_spacy(remove_noise_spacy(tokenization_spacy(documents))), True)

    dictionary = get_dictionary(tokenized_docs)

    tokenized_docs = filter_tokens_by_occurrence(tokenized_docs,dictionary)

    corpus = vector_representation(tokenized_docs, dictionary, [],True)

    while(True):
        print("Introduce your query:")
        query = input()
        procesed_query = process_query(query,dictionary)
        docs = search(procesed_query,corpus)
        print(docs)

#docs = ir_datasets.load("cranfield")
