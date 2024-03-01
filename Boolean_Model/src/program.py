from model import *

documents = load_dataset()
tokenized_docs = morphological_reduction_spacy(remove_stopwords_spacy(remove_noise_spacy(tokenization_spacy(documents))), True)
tokenized_docs = filter_tokens_by_occurrence(tokenized_docs)

vocabulary = build_vocabulary(dictionary)


corpus = vector_representation(tokenized_docs, dictionary, [],True)

consulta = "A AND (B OR NOT C)"
while(True):
    consulta = input()
    consulta_dnf = query_to_dnf(consulta,dictionary,vocabulary)



